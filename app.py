#!/usr/bin/env python3
"""
Architecture AI — Web App
Run: python3 app.py
Open: http://localhost:5000
"""

import base64
import io
import json
import os
import re
import sys

from flask import Flask, jsonify, render_template, request
from groq import Groq
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload

# ── Config ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import arch_config
    API_KEY = getattr(arch_config, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
except ImportError:
    API_KEY = os.environ.get("GROQ_API_KEY")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT = """
You are an expert architect and 3D modeling specialist. Carefully analyze the architectural image provided.

STEP 1 — Output a JSON block (between ```json and ```) with exact values extracted from this specific image:

```json
{
  "style": "exact style name e.g. Art Deco, Brutalist, Gothic, Modern Glass Tower",
  "width": 40,
  "depth": 30,
  "height": 80,
  "floors": 20,
  "floor_height": 4,
  "roof": "flat",
  "facade_material": "glass",
  "facade_color": "#4a90d9",
  "window_cols": 6,
  "window_rows": 20,
  "has_balconies": false,
  "has_entrance_canopy": true,
  "has_podium": false,
  "podium_floors": 0,
  "setbacks": 0
}
```

Rules for the JSON values:
- "style": name the EXACT architectural style you see (e.g. "Victorian Gothic", "Bauhaus", "Art Deco Skyscraper", "Brutalist Concrete", "Modern Glass Curtain Wall")
- "width", "depth", "height": real estimated meters based on what you see — vary these significantly per building
- "floors": count actual visible floors
- "roof": choose one — "flat", "pitched", "dome", "setback", "spire"
- "facade_material": choose one — "glass", "concrete", "brick", "stone", "mixed"
- "facade_color": hex color that MATCHES the actual building color in the image
- "window_cols": count actual window columns visible on the front facade
- "has_balconies": true only if balconies are clearly visible
- "has_podium": true if the building has a wider base section
- "setbacks": count visible stepped setbacks on upper floors

STEP 2 — Write a detailed architecture report:

## Executive Summary
## Architectural Style
## Spatial Layout
## Structural Elements
## Facade & Materials
## Environmental Context
## Strengths
## Verdict
Score: X/10 — justification.

Be specific to THIS building. Every building is different — reflect that in the numbers and descriptions.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(file_bytes: bytes, filename: str) -> tuple[str, str]:
    ext = os.path.splitext(filename)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        mime = "image/jpeg"
    max_dim = 1568
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if "jpeg" in mime else "PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), mime


def try_parse_json(raw: str) -> dict:
    """Try multiple strategies to extract valid JSON."""
    raw = raw.strip()
    # Remove JS-style comments
    raw = re.sub(r'//.*', '', raw)
    # Fix trailing commas
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def parse_response(text: str) -> tuple[dict, str]:
    """Extract 3D spec JSON and report text from AI response."""
    spec = {}
    report = text

    # Strategy 1: ```json ... ``` markdown block
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        spec = try_parse_json(match.group(1))

    # Strategy 2: ``` ... ``` plain block
    if not spec:
        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            spec = try_parse_json(match.group(1))

    # Strategy 3: <3D_SPEC> tags (legacy)
    if not spec:
        match = re.search(r"<3D_SPEC>(.*?)</3D_SPEC>", text, re.DOTALL)
        if match:
            spec = try_parse_json(match.group(1))

    # Strategy 4: bare JSON object anywhere in the text
    if not spec:
        match = re.search(r"\{[^{}]*\"style\"[^{}]*\}", text, re.DOTALL)
        if match:
            spec = try_parse_json(match.group(0))

    # Extract report — everything after the first ```...``` block
    report_match = re.search(r"```.*?```\s*(.*)", text, re.DOTALL)
    if report_match:
        report = report_match.group(1).strip()
    elif "## Executive Summary" in text:
        report = text[text.index("## Executive Summary"):]
    elif "<REPORT>" in text:
        r = re.search(r"<REPORT>(.*?)</REPORT>", text, re.DOTALL)
        if r:
            report = r.group(1).strip()

    # Validate / clamp spec values
    def clamp(val, lo, hi, default):
        try:
            v = float(val)
            return max(lo, min(hi, v))
        except (TypeError, ValueError):
            return default

    spec["style"]             = str(spec.get("style", "Contemporary"))
    spec["width"]             = clamp(spec.get("width"), 5, 500, 35)
    spec["depth"]             = clamp(spec.get("depth"), 5, 500, 28)
    spec["height"]            = clamp(spec.get("height"), 3, 600, 55)
    spec["floors"]            = int(clamp(spec.get("floors"), 1, 150, 14))
    spec["floor_height"]      = clamp(spec.get("floor_height"), 2, 10, 4)
    spec["roof"]              = spec.get("roof", "flat") if spec.get("roof") in ("flat","pitched","dome","setback","spire","complex") else "flat"
    spec["facade_material"]   = spec.get("facade_material", "glass") if spec.get("facade_material") in ("glass","concrete","brick","stone","mixed") else "glass"
    spec["window_cols"]       = int(clamp(spec.get("window_cols"), 1, 30, 5))
    spec["window_rows"]       = int(clamp(spec.get("window_rows"), 1, 150, spec["floors"]))
    spec["has_balconies"] = bool(spec.get("has_balconies", False))
    spec["has_entrance_canopy"] = bool(spec.get("has_entrance_canopy", True))
    spec["has_podium"] = bool(spec.get("has_podium", False))
    spec["podium_floors"] = int(clamp(spec.get("podium_floors"), 0, 20, 0))
    spec["setbacks"] = int(clamp(spec.get("setbacks"), 0, 5, 0))

    # Facade color — validate hex
    color = str(spec.get("facade_color", "#6fa8c8"))
    if not re.match(r"^#[0-9a-fA-F]{6}$", color):
        material_colors = {
            "glass": "#4a90d9", "concrete": "#8a9bb0",
            "brick": "#b5643c", "stone": "#9a8870", "mixed": "#7a8fa8"
        }
        color = material_colors.get(spec["facade_material"], "#6fa8c8")
    spec["facade_color"] = color

    return spec, report


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = file.read()
        b64, mime = image_to_base64(file_bytes, file.filename)
        image_data_url = f"data:{mime};base64,{b64}"
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    if not API_KEY:
        return jsonify({"error": "No API key configured in arch_config.py"}), 500

    try:
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            max_tokens=4096,
        )
        raw_text = response.choices[0].message.content
        spec, report = parse_response(raw_text)
        return jsonify({"spec": spec, "report": report, "image": image_data_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = 5000

    public_url = None
    try:
        from pyngrok import ngrok, conf
        ngrok_token = getattr(arch_config, "NGROK_TOKEN", None) if 'arch_config' in sys.modules else None
        if ngrok_token and ngrok_token != "YOUR_NGROK_TOKEN_HERE":
            conf.get_default().auth_token = ngrok_token
        tunnel = ngrok.connect(port, bind_tls=True)
        public_url = tunnel.public_url
    except Exception as e:
        print(f"   [!] ngrok tunnel failed: {e}")

    print("\n🏛  Architecture AI Web App")
    print(f"   Local  : http://localhost:{port}")
    if public_url:
        print(f"   Public : {public_url}  <-- share this link")
    print()

    app.run(debug=False, port=port)
