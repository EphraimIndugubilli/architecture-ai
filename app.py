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
import time

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from groq import Groq
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/energy-class": {"origins": "*"}, r"/health": {"origins": "*"}, r"/models": {"origins": "*"}, r"/accessibility": {"origins": "*"}})

# ── Config ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import arch_config
    API_KEY = getattr(arch_config, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
except ImportError:
    API_KEY = os.environ.get("GROQ_API_KEY")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Per-request model selection — 2026 LLMOps pattern: callers can override the
# model tier without a code change. Scout is the fast default; Maverick offers
# higher reasoning capacity (128 experts vs 16) for complex multi-building
# comparisons; llama-3.3-70b-versatile is the text-only fallback.
AVAILABLE_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct":     "Llama 4 Scout — fast, vision-capable (default)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick — high-capacity, vision-capable",
    "llama-3.3-70b-versatile":                       "Llama 3.3 70B — text-only fallback",
}


def resolve_model(requested: str | None) -> str:
    """Return the requested model if it's in the allow-list, otherwise MODEL."""
    if requested and requested in AVAILABLE_MODELS:
        return requested
    return MODEL

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
  "setbacks": 0,
  "extraction_confidence": 85
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
- "extraction_confidence": 0-100 — how confident you are in the extracted values above, based on image clarity, angle, occlusion, and resolution. Lower this for blurry, distant, heavily cropped, or partially obscured buildings.

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


def _groq_call(client, model: str, messages: list, max_tokens: int, max_retries: int = 3):
    """Groq API call with exponential-backoff retry on transient errors (rate limits, timeouts)."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
            )
        except Exception as e:
            err = str(e).lower()
            is_transient = any(k in err for k in ("rate", "timeout", "connection", "429", "502", "503", "500"))
            if attempt < max_retries - 1 and is_transient:
                time.sleep(2 ** attempt)
                continue
            raise


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


def _clamp(val, lo, hi, default):
    try:
        v = float(val)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def validate_spec(spec: dict) -> dict:
    """Clamp and type-coerce a raw extracted spec dict into safe, bounded values."""
    spec["style"]               = str(spec.get("style", "Contemporary"))
    spec["width"]               = _clamp(spec.get("width"), 5, 500, 35)
    spec["depth"]               = _clamp(spec.get("depth"), 5, 500, 28)
    spec["height"]              = _clamp(spec.get("height"), 3, 600, 55)
    spec["floors"]              = int(_clamp(spec.get("floors"), 1, 150, 14))
    spec["floor_height"]        = _clamp(spec.get("floor_height"), 2, 10, 4)
    spec["roof"]                = spec.get("roof", "flat") if spec.get("roof") in ("flat","pitched","dome","setback","spire","complex") else "flat"
    spec["facade_material"]     = spec.get("facade_material", "glass") if spec.get("facade_material") in ("glass","concrete","brick","stone","mixed") else "glass"
    spec["window_cols"]         = int(_clamp(spec.get("window_cols"), 1, 30, 5))
    spec["window_rows"]         = int(_clamp(spec.get("window_rows"), 1, 150, spec["floors"]))
    spec["has_balconies"]       = bool(spec.get("has_balconies", False))
    spec["has_entrance_canopy"] = bool(spec.get("has_entrance_canopy", True))
    spec["has_podium"]          = bool(spec.get("has_podium", False))
    spec["podium_floors"]       = int(_clamp(spec.get("podium_floors"), 0, 20, 0))
    spec["setbacks"]            = int(_clamp(spec.get("setbacks"), 0, 5, 0))
    spec["extraction_confidence"] = int(_clamp(spec.get("extraction_confidence"), 0, 100, 60))

    color = str(spec.get("facade_color", "#6fa8c8"))
    if not re.match(r"^#[0-9a-fA-F]{6}$", color):
        material_colors = {
            "glass": "#4a90d9", "concrete": "#8a9bb0",
            "brick": "#b5643c", "stone": "#9a8870", "mixed": "#7a8fa8"
        }
        color = material_colors.get(spec["facade_material"], "#6fa8c8")
    spec["facade_color"] = color
    return spec


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

    spec = validate_spec(spec)
    return spec, report


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "api_key_set": bool(API_KEY),
        "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
    })


@app.route("/models")
def models():
    return jsonify({
        "default": MODEL,
        "available": [
            {"id": mid, "description": desc}
            for mid, desc in AVAILABLE_MODELS.items()
        ],
    })


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

    chosen_model = resolve_model(request.form.get("model"))
    try:
        client = Groq(api_key=API_KEY)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": PROMPT},
            ],
        }]
        response = _groq_call(client, chosen_model, messages, 4096)
        raw_text = response.choices[0].message.content
        spec, report = parse_response(raw_text)
        return jsonify({"spec": spec, "report": report, "image": image_data_url, "model_used": chosen_model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/stream", methods=["POST"])
def analyze_stream():
    """SSE-streaming version of /analyze — tokens arrive word-by-word."""
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

    chosen_model = resolve_model(request.form.get("model"))

    @stream_with_context
    def generate():
        full_text = []
        try:
            client = Groq(api_key=API_KEY)
            stream = client.chat.completions.create(
                model=chosen_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                max_tokens=4096,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_text.append(delta)
                    yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"

            raw_text = "".join(full_text)
            spec, report = parse_response(raw_text)
            yield f"data: {json.dumps({'type': 'done', 'spec': spec, 'report': report, 'image': image_data_url, 'model_used': chosen_model})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


COMPARE_PROMPT = """
You are an expert architect comparing two buildings side by side.

For each building output a JSON block labeled ```json-a``` and ```json-b``` with the same fields as the standard spec.
Then write a detailed comparison report covering:

## Side-by-Side Comparison
## Architectural Style Contrast
## Scale & Massing
## Facade & Materials
## Strengths of Each
## Verdict — Which is More Architecturally Significant and Why?

Be specific, precise, and reference what you actually see.
"""


SUSTAINABILITY_PROMPT = """
You are a certified green building consultant (LEED AP, BREEAM assessor). Analyze this architectural image and evaluate the building's environmental credentials.

Output a JSON block (between ```json and ```) with these sustainability metrics:

```json
{
  "overall_rating": "B",
  "energy_efficiency_score": 65,
  "solar_potential": "high",
  "green_roof_potential": "medium",
  "facade_heat_gain": "high",
  "estimated_carbon_class": "D",
  "natural_ventilation_potential": "low",
  "embodied_carbon_estimate": "medium",
  "leed_likely_category": "Silver",
  "sustainability_strengths": ["compact massing reduces heat loss", "large south-facing glazing for passive solar"],
  "sustainability_concerns": ["all-glass curtain wall increases cooling load", "no visible green infrastructure"],
  "retrofit_recommendations": ["add external solar shading fins", "install green roof on podium", "photovoltaic cladding on south face"]
}
```

Rating scale: A+ (best) to F (worst)
Solar potential: high/medium/low based on visible roof area and glazing orientation
Facade heat gain: high/medium/low (glass = high, masonry = low)
Carbon class: A–G (like EU EPC labels)
LEED likely: Platinum / Gold / Silver / Certified / Unlikely

Then write a sustainability report:

## Green Building Overview
## Energy Performance
## Carbon & Materials Assessment
## Passive Design Strategies
## Retrofit Potential
## Sustainability Score: X/10

Be specific to what you actually see in this building. Identify the architectural style's inherent sustainability characteristics.
"""


@app.route("/sustainability", methods=["POST"])
def sustainability():
    """Analyze a building image for green/sustainability metrics using LLaMA vision."""
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
                    {"type": "text", "text": SUSTAINABILITY_PROMPT},
                ],
            }],
            max_tokens=2048,
        )
        raw_text = response.choices[0].message.content

        # Extract JSON metrics
        metrics: dict = {}
        match = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
        if match:
            metrics = try_parse_json(match.group(1))

        # Extract report text
        report = raw_text
        report_match = re.search(r"```.*?```\s*(.*)", raw_text, re.DOTALL)
        if report_match:
            report = report_match.group(1).strip()
        elif "## Green Building" in raw_text:
            report = raw_text[raw_text.index("## Green Building"):]

        # Validate and clamp fields
        valid_ratings = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "E", "F"]
        metrics["overall_rating"] = (
            metrics.get("overall_rating", "C")
            if metrics.get("overall_rating") in valid_ratings else "C"
        )
        try:
            score = int(metrics.get("energy_efficiency_score", 50) or 50)
            metrics["energy_efficiency_score"] = max(0, min(100, score))
        except (TypeError, ValueError):
            metrics["energy_efficiency_score"] = 50

        for field in ["solar_potential", "green_roof_potential", "facade_heat_gain",
                      "natural_ventilation_potential", "embodied_carbon_estimate"]:
            metrics[field] = (
                metrics.get(field, "medium")
                if metrics.get(field) in ("high", "medium", "low") else "medium"
            )

        valid_carbon = list("ABCDEFG")
        metrics["estimated_carbon_class"] = (
            metrics.get("estimated_carbon_class", "D")
            if metrics.get("estimated_carbon_class") in valid_carbon else "D"
        )

        valid_leed = ["Platinum", "Gold", "Silver", "Certified", "Unlikely"]
        metrics["leed_likely_category"] = (
            metrics.get("leed_likely_category", "Unlikely")
            if metrics.get("leed_likely_category") in valid_leed else "Unlikely"
        )

        for field in ["sustainability_strengths", "sustainability_concerns", "retrofit_recommendations"]:
            val = metrics.get(field, [])
            metrics[field] = val if isinstance(val, list) else []

        return jsonify({"metrics": metrics, "report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    if "image_a" not in request.files or "image_b" not in request.files:
        return jsonify({"error": "Two images required: image_a and image_b"}), 400

    try:
        fa, fb = request.files["image_a"], request.files["image_b"]
        b64_a, mime_a = image_to_base64(fa.read(), fa.filename)
        b64_b, mime_b = image_to_base64(fb.read(), fb.filename)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    if not API_KEY:
        return jsonify({"error": "No API key configured"}), 500

    try:
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_a};base64,{b64_a}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_b};base64,{b64_b}"}},
                    {"type": "text", "text": COMPARE_PROMPT},
                ],
            }],
            max_tokens=4096,
        )
        raw = response.choices[0].message.content

        # Extract and validate specs for both buildings
        m_a = re.search(r"```json-a\s*(.*?)```", raw, re.DOTALL)
        spec_a = validate_spec(try_parse_json(m_a.group(1)) if m_a else {})
        m_b = re.search(r"```json-b\s*(.*?)```", raw, re.DOTALL)
        spec_b = validate_spec(try_parse_json(m_b.group(1)) if m_b else {})

        report = raw
        for pat in [r"```json-a.*?```", r"```json-b.*?```"]:
            report = re.sub(pat, "", report, flags=re.DOTALL).strip()

        return jsonify({
            "spec_a": spec_a, "spec_b": spec_b,
            "report": report,
            "image_a": f"data:{mime_a};base64,{b64_a}",
            "image_b": f"data:{mime_b};base64,{b64_b}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/energy-class", methods=["POST"])
def energy_class():
    """Estimate EPC energy-efficiency class (A++ → G) from a building spec.

    Accepts the same JSON spec that /analyze returns. The scoring is
    rule-based (deterministic), not AI-generated, so it is cheap to call
    and suitable for real-time UI display alongside the spec extraction.

    2026 context: European and Indian GBC regulations increasingly require
    an energy performance estimate at design time. Integrating a quick
    EPC estimate directly into the analysis flow reduces friction for
    architects using the tool professionally.

    Request body (JSON): same spec dict as returned by /analyze.
    Returns: { "class": "B", "score": 68, "label": "Good", "factors": [...], "tips": [...] }
    """
    spec = request.get_json(silent=True) or {}

    score = 70  # baseline for an average modern building

    # Facade material: glass is worst insulator; stone/brick best
    material_delta = {
        "glass":    -20,
        "concrete":  -5,
        "mixed":     -5,
        "brick":     +8,
        "stone":    +12,
    }
    material = str(spec.get("facade_material", "concrete")).lower()
    score += material_delta.get(material, 0)

    # Window density: window_cols * window_rows / floors gives glazing intensity
    cols    = int(spec.get("window_cols", 4))
    rows    = int(spec.get("window_rows", 10))
    floors  = max(1, int(spec.get("floors", 10)))
    glazing = (cols * rows) / (floors * 4)  # normalised 0–~5
    score -= min(15, int(glazing * 6))       # max –15 for very high glazing

    # Balconies: thermal bridging
    if spec.get("has_balconies"):
        score -= 5

    # Entrance canopy shades ground-floor glazing
    if spec.get("has_entrance_canopy"):
        score += 2

    # Podium base reduces exposed envelope per floor above it
    if spec.get("has_podium") and int(spec.get("podium_floors", 0)) >= 2:
        score += 3

    # Roof type: pitched traps more air; dome is well-insulating
    roof_delta = {"pitched": +8, "dome": +5, "flat": 0, "setback": +3, "spire": +4}
    roof = str(spec.get("roof", "flat")).lower()
    score += roof_delta.get(roof, 0)

    # Setbacks reduce wind exposure on upper floors
    setbacks = int(spec.get("setbacks", 0))
    score += min(6, setbacks * 2)

    # Tall buildings lose more heat through external surface area
    height = int(spec.get("height", 30))
    if height > 60:
        score -= 8
    elif height > 30:
        score -= 4

    score = max(0, min(100, score))

    # Map score → EPC class
    if   score >= 95: cls, label = "A++", "Net Zero / Passive"
    elif score >= 85: cls, label = "A+",  "Excellent"
    elif score >= 75: cls, label = "A",   "Very Good"
    elif score >= 65: cls, label = "B",   "Good"
    elif score >= 55: cls, label = "C",   "Average"
    elif score >= 45: cls, label = "D",   "Below Average"
    elif score >= 35: cls, label = "E",   "Poor"
    elif score >= 25: cls, label = "F",   "Very Poor"
    else:             cls, label = "G",   "Non-Compliant"

    factors: list[str] = []
    tips: list[str] = []

    if material == "glass":
        factors.append("All-glass facade has very low insulation value (U-value ~2–6 W/m²K).")
        tips.append("Specify triple-glazed unitised curtain wall with low-e coating to recover 10–15 points.")
    elif material in ("brick", "stone"):
        factors.append(f"{material.title()} facade provides good thermal mass and moderate insulation.")
    if glazing > 1.5:
        factors.append(f"High window density ({cols}×{rows} grid) increases solar gain and heat loss.")
        tips.append("Add external shading fins or electrochromic glass to cut solar gain by 30–40%.")
    if spec.get("has_balconies"):
        factors.append("Balconies create thermal bridging at every slab edge.")
        tips.append("Use thermally broken balcony connectors to reduce bridging penalty.")
    if roof == "flat":
        tips.append("Green roof or white reflective membrane on flat roof can add 5–8 points.")
    if height > 60:
        factors.append(f"High-rise ({height} m) has large exposed envelope-to-floor-area ratio.")
        tips.append("Increase insulation thickness on curtain wall spandrel panels.")

    if not factors:
        factors.append("No major negative factors detected for this spec.")
    if not tips:
        tips.append("Building appears reasonably efficient for its typology — focus on HVAC optimisation.")

    return jsonify({
        "class": cls,
        "score": score,
        "label": label,
        "factors": factors,
        "tips": tips,
        "note": "Estimate based on extracted spec only — commission a certified EPC assessment for regulatory purposes.",
    })


SHADOW_HOURS = [8, 10, 12, 14, 16, 18]


@app.route("/shadow-analysis", methods=["POST"])
def shadow_analysis():
    """Deterministic solar shadow analysis from a building spec.

    Accepts the same JSON spec returned by /analyze and computes approximate
    shadow length and cardinal bearing at six times of day (8 am → 6 pm) for
    any latitude/longitude. No AI cost — pure trigonometry.

    2026 green-building context: shadow-casting data is increasingly required
    at design time for daylight-access compliance (UK BRE 209, Indian NBC
    Part 8). Integrating it directly into the analysis flow saves architects
    a separate tool switch.

    Request body: { ...spec, latitude: 28.6, longitude: 77.2, month: 6 }
    latitude and longitude default to New Delhi if omitted.
    month (1-12) determines solar declination; defaults to 6 (summer solstice).
    """
    import math

    body = request.get_json(silent=True) or {}
    height = float(body.get("height", 30))
    lat_deg = float(body.get("latitude", 28.6))
    month = int(body.get("month", 6))
    height = max(1.0, min(600.0, height))
    lat = math.radians(max(-90.0, min(90.0, lat_deg)))

    # Solar declination (°) via Spencer approximation — max ≈ ±23.45° at solstices
    day_of_year = {1: 15, 2: 46, 3: 75, 4: 106, 5: 136, 6: 167,
                   7: 197, 8: 228, 9: 259, 10: 289, 11: 320, 12: 350}.get(month, 167)
    B = math.radians((360 / 365) * (day_of_year - 81))
    decl = math.radians(23.45 * math.sin(B))

    timeline = []
    for hour in SHADOW_HOURS:
        # Hour angle: 0 at solar noon; negative morning, positive afternoon
        hour_angle = math.radians((hour - 12) * 15)

        # Solar altitude angle
        sin_alt = (math.sin(lat) * math.sin(decl)
                   + math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
        altitude_rad = math.asin(max(-1.0, min(1.0, sin_alt)))
        altitude_deg = math.degrees(altitude_rad)

        if altitude_deg <= 0:
            # Sun below horizon — no shadow length computable
            timeline.append({
                "hour": hour,
                "label": f"{hour:02d}:00",
                "sun_altitude_deg": round(altitude_deg, 1),
                "shadow_length_m": None,
                "shadow_bearing": None,
                "above_horizon": False,
            })
            continue

        shadow_length = height / math.tan(altitude_rad)

        # Solar azimuth (bearing of the sun from North, clockwise)
        cos_az = ((math.sin(decl) - math.sin(lat) * sin_alt)
                  / (math.cos(lat) * math.cos(altitude_rad) + 1e-9))
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth = math.degrees(math.acos(cos_az))
        if hour > 12:
            azimuth = 360 - azimuth  # afternoon: sun moves to west

        # Shadow falls opposite the sun (180° flip)
        shadow_bearing = (azimuth + 180) % 360

        timeline.append({
            "hour": hour,
            "label": f"{hour:02d}:00",
            "sun_altitude_deg": round(altitude_deg, 1),
            "sun_azimuth_deg": round(azimuth, 1),
            "shadow_length_m": round(shadow_length, 1),
            "shadow_bearing": round(shadow_bearing, 1),
            "shadow_direction": _bearing_to_cardinal(shadow_bearing),
            "above_horizon": True,
        })

    above = [t for t in timeline if t["above_horizon"] and t["shadow_length_m"] is not None]
    max_shadow = max((t["shadow_length_m"] for t in above), default=None)
    min_shadow = min((t["shadow_length_m"] for t in above), default=None)

    return jsonify({
        "building_height_m": height,
        "latitude": lat_deg,
        "month": month,
        "timeline": timeline,
        "max_shadow_length_m": round(max_shadow, 1) if max_shadow else None,
        "min_shadow_length_m": round(min_shadow, 1) if min_shadow else None,
        "note": "Approximate shadow analysis based on solar geometry. Commission a certified daylight study for planning/compliance purposes.",
    })


@app.route("/wind-analysis", methods=["POST"])
def wind_analysis():
    """Deterministic wind load estimate from a building spec.

    Accepts the same JSON spec returned by /analyze and computes approximate
    wind pressure and base shear at six reference wind speeds (10–90 m/s).
    Uses simplified ASCE 7 / IS:875 Part 3 methodology — suitable for
    early-stage feasibility; commission a full CFD study for regulatory use.

    2026 context: India's National Building Code Part 6 now mandates wind load
    documentation for buildings taller than 15 m. Integrating a quick estimate
    here saves architects a separate specialist tool at the sketch stage.

    Request body: { ...spec, wind_speed_ms: 30, latitude: 28.6 }
    wind_speed_ms: design wind speed in m/s (defaults to 33 m/s — IS:875 Zone III)
    latitude: used to infer exposure category (coastal vs inland)
    """
    import math

    body = request.get_json(silent=True) or {}
    height   = float(body.get("height", 30))
    width    = float(body.get("width", 20))
    depth    = float(body.get("depth", 20))
    floors   = int(body.get("floors", 8))
    material = str(body.get("facade_material", "concrete")).lower()
    lat_deg  = float(body.get("latitude", 28.6))
    vb       = float(body.get("wind_speed_ms", 33))

    height  = max(1.0,  min(600.0, height))
    width   = max(1.0,  min(500.0, width))
    depth   = max(1.0,  min(500.0, depth))
    vb      = max(10.0, min(90.0,  vb))

    # Exposure category based on height and coastal proximity (|lat| < 12 = tropical coast)
    coastal = abs(lat_deg) < 12
    if height > 100 or coastal:
        exposure = "A"   # open terrain / coastal — highest wind loads
        k2_factor = 1.10
    elif height > 40:
        exposure = "B"   # open terrain, suburban outskirts
        k2_factor = 1.00
    else:
        exposure = "C"   # urban / suburban — reduced by surrounding buildings
        k2_factor = 0.88

    # Risk / importance factor (IS:875): residential=1.0, commercial=1.07, essential=1.15
    k1 = 1.07  # commercial/office default — most likely use for this tool

    # Topography factor — assume flat terrain
    k3 = 1.0

    # Design wind speed
    vz = vb * k1 * k2_factor * k3

    # Basic wind pressure: pz = 0.6 × Vz² (IS:875 formula, result in N/m²)
    pz = 0.6 * vz ** 2  # N/m²

    # External pressure coefficients (Cp) for a rectangular building
    # Front face: Cp = +0.8 (positive pressure); rear: -0.5 (suction)
    # Side walls: -0.7 (suction); roof flat: -0.9 (uplift)
    cp_windward  =  0.8
    cp_leeward   = -0.5
    cp_side      = -0.7
    cp_roof      = -0.9 if body.get("roof", "flat") == "flat" else -0.6

    # Net wind pressure on windward face
    p_windward = round(pz * cp_windward / 1000, 3)     # kN/m²
    p_leeward  = round(pz * abs(cp_leeward) / 1000, 3) # kN/m² (suction)
    p_side     = round(pz * abs(cp_side) / 1000, 3)
    p_roof     = round(pz * abs(cp_roof) / 1000, 3)

    # Total lateral wind force on windward face (simplified: uniform pressure × area)
    windward_area = width * height  # m²
    total_lateral_kN = round(pz * cp_windward * windward_area / 1000, 1)

    # Base shear from wind — triangular distribution assumed (peak at top)
    # V_base ≈ 0.6 × p_total × A for a triangular distribution over height
    base_shear_kN = round(total_lateral_kN * 0.6, 1)

    # Overturning moment about base = force × (H × 2/3) for triangular load
    overturning_moment_kNm = round(base_shear_kN * (height * 2 / 3), 1)

    # Facade cladding pressure (max of windward + suction)
    facade_design_pressure_kPa = round((pz * (cp_windward - cp_leeward)) / 1000, 3)

    # Risk category based on base shear
    if base_shear_kN < 200:
        risk = "low"
        risk_note = "Wind loads manageable with standard structural framing."
    elif base_shear_kN < 800:
        risk = "moderate"
        risk_note = "Wind governs some member design — engage structural engineer early."
    elif base_shear_kN < 2000:
        risk = "high"
        risk_note = "Wind is a primary structural load — dedicated wind study required."
    else:
        risk = "very high"
        risk_note = "Supertall / high-exposure building — full CFD wind tunnel study essential."

    # Material-specific notes
    material_notes = {
        "glass":    "All-glass curtain wall: facade panels must resist local cladding pressure; glazing bite depth critical.",
        "concrete": "Concrete frame provides good mass damping; check shear walls for lateral load path.",
        "brick":    "Brick is brittle under cyclic wind load; ensure adequate ties and cavity wall design.",
        "stone":    "Heavy stone cladding increases seismic demand; ensure positive mechanical fixings.",
        "mixed":    "Mixed facade: verify each panel type meets local pressure independently.",
    }.get(material, "Verify facade system meets local pressure requirements.")

    # Deflection check guidance (H/500 serviceability drift limit is common)
    allowable_drift_mm = round((height * 1000) / 500, 0)

    return jsonify({
        "building_height_m":       round(height, 1),
        "building_width_m":        round(width, 1),
        "windward_area_m2":        round(windward_area, 1),
        "design_wind_speed_ms":    round(vz, 1),
        "basic_wind_speed_ms":     round(vb, 1),
        "exposure_category":       exposure,
        "wind_pressure_kPa": {
            "windward":  p_windward,
            "leeward_suction": p_leeward,
            "side_suction":    p_side,
            "roof_uplift":     p_roof,
            "facade_net_design": facade_design_pressure_kPa,
        },
        "lateral_loads": {
            "total_lateral_force_kN":    total_lateral_kN,
            "base_shear_kN":             base_shear_kN,
            "overturning_moment_kNm":    overturning_moment_kNm,
            "allowable_drift_limit_mm":  allowable_drift_mm,
        },
        "risk_level":  risk,
        "risk_note":   risk_note,
        "material_note": material_notes,
        "note": "Simplified ASCE 7 / IS:875 Part 3 estimate. Commission a certified structural wind study for planning and building permit purposes.",
    })


def _bearing_to_cardinal(bearing: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(bearing / 22.5) % 16
    return dirs[idx]



@app.route("/material-estimator", methods=["POST"])
def material_estimator():
    """Rule-based construction material quantity estimator from a building spec.

    Takes the same JSON spec returned by /analyze and computes approximate
    quantities of the four primary structural/envelope materials: concrete,
    reinforcing steel, facade glazing, and cladding area. Includes a rough
    cost-per-m² range and a note on embodied carbon class.

    This is pure arithmetic — no AI cost — and makes the analysis actionable
    for early-stage design decision-making. Aligns with the 2026 AI app UX
    trend of extending AI outputs with deterministic, transparent calculations
    that architects can immediately apply in feasibility studies.

    Request body: same spec dict as returned by /analyze
    Returns: { concrete_m3, steel_kg, glazing_m2, cladding_m2, cost_range_usd_m2,
               gross_floor_area_m2, embodied_carbon_class, breakdown: [...] }
    """
    spec = request.get_json(silent=True) or {}

    width   = float(spec.get("width", 35))
    depth   = float(spec.get("depth", 28))
    height  = float(spec.get("height", 55))
    floors  = max(1, int(spec.get("floors", 14)))
    material = str(spec.get("facade_material", "concrete")).lower()
    window_cols = int(spec.get("window_cols", 5))
    window_rows = int(spec.get("window_rows", floors))
    has_podium  = bool(spec.get("has_podium", False))
    podium_floors = max(0, int(spec.get("podium_floors", 0)))
    setbacks = int(spec.get("setbacks", 0))

    # ── Floor areas ──────────────────────────────────────────────────
    gross_floor_area = width * depth * floors   # m²
    podium_area = (width * 1.3) * (depth * 1.3) * podium_floors if has_podium else 0.0

    # ── Concrete (structural frame + slabs) ──────────────────────────
    # Industry rule of thumb: ~0.30–0.45 m³ of concrete per m² of floor area
    # depending on height / seismic zone. Glass towers use more due to larger
    # spans; masonry/brick uses less (walls carry load).
    concrete_factor = {
        "glass": 0.42, "concrete": 0.40, "mixed": 0.38,
        "brick": 0.32, "stone": 0.30,
    }.get(material, 0.38)
    concrete_m3 = round((gross_floor_area + podium_area) * concrete_factor)

    # ── Reinforcing steel (rebar in slabs and columns) ──────────────
    # Typical: 100–150 kg per m³ of concrete for high-rise; 70–100 for low-rise.
    rebar_per_m3 = 140 if floors > 20 else 110 if floors > 10 else 85
    steel_kg = round(concrete_m3 * rebar_per_m3)

    # ── Facade areas ─────────────────────────────────────────────────
    perimeter = 2 * (width + depth)
    facade_total_m2 = round(perimeter * height)

    # Window grid: cols × rows windows, each assumed ~1.5 m × 1.8 m = 2.7 m²
    window_area_each = 1.5 * 1.8
    glazing_m2 = round(window_cols * window_rows * window_area_each)
    glazing_m2 = min(glazing_m2, int(facade_total_m2 * 0.85))  # cap at 85% of facade

    cladding_m2 = max(0, facade_total_m2 - glazing_m2)

    # ── Cost range (USD / m² of gross floor area, 2026 benchmarks) ──
    # Source: Turner & Townsend International Construction Market Survey 2026
    base_cost = {
        "glass":    3800,  # curtain-wall high-rise
        "concrete": 2600,  # reinforced concrete frame
        "mixed":    3000,
        "brick":    2200,  # brick/masonry load-bearing
        "stone":    3400,  # natural stone cladding premium
    }.get(material, 2800)

    height_premium = 1.15 if floors > 30 else 1.08 if floors > 15 else 1.0
    cost_low  = round(gross_floor_area * base_cost * height_premium * 0.85 / 1000) * 1000
    cost_high = round(gross_floor_area * base_cost * height_premium * 1.15 / 1000) * 1000

    # ── Embodied carbon class (rough estimate) ───────────────────────
    # Glass curtain walls have highest embodied carbon; timber/stone lowest.
    carbon_score = {
        "glass": "high", "mixed": "medium-high",
        "concrete": "medium", "brick": "medium-low", "stone": "low",
    }.get(material, "medium")

    breakdown = [
        f"Structural concrete: ~{concrete_m3:,} m³ ({concrete_factor:.2f} m³/m² gross floor area)",
        f"Reinforcing steel: ~{steel_kg:,} kg ({rebar_per_m3} kg per m³ concrete)",
        f"Glazing (window grid {window_cols}×{window_rows}): ~{glazing_m2:,} m²",
        f"Solid facade cladding: ~{cladding_m2:,} m² ({material})",
        f"Estimated project cost range: USD {cost_low:,}–{cost_high:,}",
        f"Cost basis: {base_cost} USD/m² × {height_premium:.2f} height factor × {gross_floor_area:.0f} m² GFA",
    ]

    return jsonify({
        "gross_floor_area_m2": round(gross_floor_area),
        "concrete_m3": concrete_m3,
        "steel_kg": steel_kg,
        "glazing_m2": glazing_m2,
        "cladding_m2": cladding_m2,
        "cost_range_usd": {"low": cost_low, "high": cost_high},
        "embodied_carbon_class": carbon_score,
        "breakdown": breakdown,
        "note": (
            "Quantities are indicative early-stage estimates only. "
            "Commission a quantity surveyor for tender-grade BOQ."
        ),
    })


# ── Accessibility Assessment ──────────────────────────────────────────────────

ACCESSIBILITY_PROMPT = """
You are an accessibility consultant and architect specialising in ADA (Americans with Disabilities Act), WCAG spatial equivalents, and universal design principles.

Carefully examine the building image provided and assess it for physical accessibility.

STEP 1 — Output a JSON block (between ```json and ```) with your accessibility assessment:

```json
{
  "overall_score": 65,
  "wheelchair_access": "partial",
  "step_free_entrance": false,
  "visible_ramp": true,
  "parking_accessible": "unknown",
  "elevator_likely": true,
  "tactile_paving": false,
  "wide_doorways_likely": true,
  "accessible_signage": false,
  "barriers": ["steps at main entrance", "narrow side path"],
  "positives": ["ramp visible on north side", "flush threshold at service entrance"],
  "assessment_confidence": 55
}
```

Field rules:
- "overall_score": 0–100 universal design score based on visible features
- "wheelchair_access": "full", "partial", "likely_limited", or "unknown"
- "step_free_entrance": true only if you can confirm no steps at the main entrance
- "visible_ramp": true if a ramp or sloped approach is clearly visible
- "parking_accessible": "yes", "no", or "unknown" — only "yes" if accessible bays are visible
- "elevator_likely": true for multi-storey buildings; false for single-storey without lift
- "tactile_paving": true if yellow/textured tactile guide strips are visible
- "wide_doorways_likely": true if entrances appear ≥ 900mm wide based on proportions
- "accessible_signage": true only if high-contrast or Braille signage is clearly visible
- "barriers": list up to 5 specific physical barriers you can observe
- "positives": list up to 5 positive accessibility features you can observe
- "assessment_confidence": 0–100 — lower for distant/occluded/partial views

STEP 2 — Write a concise accessibility report:

## Accessibility Overview
## Entrance & Approach
## Vertical Circulation
## Key Barriers
## Recommendations
## Accessibility Rating: X/10 — brief justification.

Be specific to what you actually see. Do not invent features not visible in the image.
"""


@app.route("/accessibility", methods=["POST"])
def accessibility():
    """Assess a building image for universal design and ADA-style accessibility.

    2026 AI UX trend: "transparent AI" means surfacing concrete, actionable
    insight — not just aesthetics. Accessibility scoring fills a genuine gap in
    architectural AI tools: most analyse style and structure but ignore whether
    the building is physically usable by everyone.

    Accepts: multipart/form-data with field 'image' (JPEG/PNG/WEBP).
    Returns: JSON with structured accessibility metrics + a written report.
    """
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
        return jsonify({"error": "No API key configured"}), 500

    try:
        client = Groq(api_key=API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text",      "text": ACCESSIBILITY_PROMPT},
                ],
            }],
            max_tokens=1500,
            temperature=0.2,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as e:
        return jsonify({"error": f"AI request failed: {str(e)}"}), 502

    metrics = {}
    match = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
    if match:
        metrics = try_parse_json(match.group(1))

    # Normalise key fields
    score = metrics.get("overall_score")
    try:
        score = max(0, min(100, int(score)))
    except (TypeError, ValueError):
        score = 50
    metrics["overall_score"] = score

    wheelchair = metrics.get("wheelchair_access", "unknown")
    if wheelchair not in ("full", "partial", "likely_limited", "unknown"):
        metrics["wheelchair_access"] = "unknown"

    for bool_field in ("step_free_entrance", "visible_ramp", "elevator_likely",
                       "tactile_paving", "wide_doorways_likely", "accessible_signage"):
        metrics[bool_field] = bool(metrics.get(bool_field, False))

    for list_field in ("barriers", "positives"):
        if not isinstance(metrics.get(list_field), list):
            metrics[list_field] = []

    confidence = metrics.get("assessment_confidence")
    try:
        metrics["assessment_confidence"] = max(0, min(100, int(confidence)))
    except (TypeError, ValueError):
        metrics["assessment_confidence"] = 50

    # Extract report text
    report = ""
    report_match = re.search(r"```.*?```\s*(.*)", raw_text, re.DOTALL)
    if report_match:
        report = report_match.group(1).strip()
    elif "## Accessibility Overview" in raw_text:
        report = raw_text[raw_text.index("## Accessibility Overview"):].strip()

    return jsonify({
        "metrics": metrics,
        "report": report,
        "model": MODEL,
    })


@app.route("/seismic-risk", methods=["POST"])
def seismic_risk():
    """Deterministic seismic vulnerability estimate from a building spec.

    Accepts the same JSON spec returned by /analyze and computes approximate
    seismic base shear and vulnerability rating using simplified IS:1893 /
    ASCE 7 methodology. Pure arithmetic — no AI cost.

    2026 context: India's National Building Code Part 6 (Rev. 2026) and ASCE
    7-22 both require seismic documentation earlier in the design process.
    Integrating a quick seismic feasibility check alongside wind and shadow
    analysis gives architects a complete structural triage at the sketch stage.

    Request body: { ...spec, latitude: 28.6, wind_speed_ms: 33 }
    latitude used to infer IS:1893 seismic zone (default: Delhi-NCR, Zone IV).
    """
    import math

    body = request.get_json(silent=True) or {}
    height   = max(1.0,  min(600.0, float(body.get("height",   30))))
    width    = max(1.0,  min(500.0, float(body.get("width",    35))))
    depth    = max(1.0,  min(500.0, float(body.get("depth",    28))))
    floors   = max(1,    int(body.get("floors",   8)))
    material = str(body.get("facade_material", "concrete")).lower()
    has_podium  = bool(body.get("has_podium",  False))
    setbacks    = max(0, int(body.get("setbacks",  0)))
    lat_deg  = float(body.get("latitude", 28.6))

    # ── IS:1893 Seismic Zone ──────────────────────────────────────────
    # Proxy: latitude ranges covering major Indian seismic zones.
    # Zone V (Z=0.36): Himalayas, Andaman; Zone IV (Z=0.24): Delhi, Punjab;
    # Zone III (Z=0.16): Mumbai, Kolkata; Zone II (Z=0.10): South Deccan.
    if lat_deg >= 31 or lat_deg <= 10:
        zone, Z = "V",   0.36
    elif 28 <= lat_deg < 31 or (22 <= lat_deg < 28 and abs(lat_deg - 26) < 2):
        zone, Z = "IV",  0.24
    elif 18 <= lat_deg < 28:
        zone, Z = "III", 0.16
    else:
        zone, Z = "II",  0.10

    # ── Response Reduction Factor R (IS:1893 Table 9) ─────────────────
    # Higher R = more ductile = lower design force requirement.
    r_map = {
        "concrete": 5.0,  # ductile RC frame
        "glass":    4.5,  # steel moment frame / curtain wall
        "mixed":    4.0,
        "brick":    3.0,  # ordinary unreinforced masonry
        "stone":    2.5,  # stone masonry — least ductile
    }
    R = r_map.get(material, 4.0)

    # ── Importance Factor I ───────────────────────────────────────────
    I = 1.2  # commercial/office (residential=1.0, essential=1.5)

    # ── Approximate Fundamental Period T (IS:1893 Cl. 7.6.1) ─────────
    # T = 0.075 × H^0.75  (RC frame);  T = 0.085 × H^0.75  (steel)
    ct = 0.085 if material in ("glass",) else 0.075
    T = round(ct * (height ** 0.75), 3)

    # ── Spectral Acceleration Sa/g (IS:1893 Fig. 2, medium soil) ──────
    if T <= 0.1:
        Sa_g = 2.5
    elif T <= 0.55:
        Sa_g = 2.5
    elif T <= 4.0:
        Sa_g = round(1.36 / T, 4)
    else:
        Sa_g = 0.34

    # ── Seismic Base Shear Coefficient Ah ────────────────────────────
    Ah = round((Z / 2) * (I / R) * Sa_g, 5)

    # ── Seismic Weight W (simplified: 12 kN/m² per floor) ────────────
    floor_area = width * depth
    W_kN = round(floor_area * floors * 12.0, 1)
    Vb_kN = round(Ah * W_kN, 1)

    # ── Slenderness Ratio ─────────────────────────────────────────────
    min_plan = min(width, depth)
    slenderness = round(height / max(min_plan, 1), 2)

    # ── Vulnerability Score (0-100; higher = safer) ───────────────────
    score = 75  # baseline for a typical concrete mid-rise

    # Material ductility
    mat_adj = {"concrete": 0, "glass": -3, "mixed": -8, "brick": -22, "stone": -28}
    score += mat_adj.get(material, 0)

    # Height penalty (taller = more seismic demand, harder to control drift)
    if height > 100: score -= 20
    elif height > 60: score -= 12
    elif height > 30: score -= 6

    # Slenderness (H/B): slender towers prone to higher base moment and P-delta
    if slenderness > 5:  score -= 18
    elif slenderness > 3: score -= 9

    # Podium: creates soft-story discontinuity — weak story failure mechanism
    if has_podium: score -= 12

    # Setbacks reduce plan irregularity and torsional eccentricity
    score += min(12, setbacks * 4)

    # Zone penalty — higher zone = higher spectral demand
    zone_adj = {"V": -15, "IV": -8, "III": 0, "II": 8}
    score += zone_adj.get(zone, 0)

    score = max(0, min(100, score))

    if   score >= 75: risk_level = "low";       risk_label = "Low — structure likely to meet IS:1893 code requirements with standard detailing."
    elif score >= 55: risk_level = "moderate";  risk_label = "Moderate — some vulnerabilities; seismic detailing (IS:13920) review recommended."
    elif score >= 35: risk_level = "high";      risk_label = "High — significant vulnerabilities; structural engineering review and capacity design required."
    else:             risk_level = "very_high"; risk_label = "Very High — critical vulnerabilities; retrofit assessment or redesign essential before proceeding."

    recommendations: list[str] = []
    if material in ("brick", "stone"):
        recommendations.append("Unreinforced masonry is highly seismically vulnerable — add RC jacketing, tie columns, or replace with confined masonry.")
    if has_podium:
        recommendations.append("Podium creates a soft-story mechanism at the transition level — ensure podium columns are stronger than upper-floor columns (capacity design principle).")
    if slenderness > 3:
        recommendations.append(f"Slender building (H/B ≈ {slenderness}) — check lateral drift (serviceability limit H/250) and P-delta amplification under seismic load.")
    if height > 60:
        recommendations.append("High-rise requires performance-based seismic design (PBSD) per IS:1893 Part 1 Cl. 12 — linear static analysis alone is insufficient.")
    if zone in ("IV", "V") and material != "concrete":
        recommendations.append(f"Zone {zone} (Z={Z}) with non-concrete structure — consider switching to ductile RC frame or steel braced frame to achieve R≥5.")
    if not recommendations:
        recommendations.append("No critical geometric vulnerabilities detected — verify reinforcement detailing meets IS:13920 ductile detailing requirements.")

    return jsonify({
        "seismic_zone":               zone,
        "zone_factor_Z":              Z,
        "importance_factor_I":        I,
        "response_reduction_R":       R,
        "fundamental_period_sec":     T,
        "spectral_acceleration_Sa_g": Sa_g,
        "base_shear_coefficient_Ah":  Ah,
        "seismic_weight_kN":          W_kN,
        "seismic_base_shear_kN":      Vb_kN,
        "slenderness_ratio":          slenderness,
        "vulnerability_score":        score,
        "risk_level":                 risk_level,
        "risk_label":                 risk_label,
        "recommendations":            recommendations,
        "note": (
            "Simplified IS:1893 / ASCE 7 seismic estimate for early-stage feasibility only. "
            "Site-specific soil classification (Type I/II/III), dynamic analysis, and ductile "
            "detailing per IS:13920 are mandatory for regulatory submission."
        ),
    })


@app.route("/acoustic-assessment", methods=["POST"])
def acoustic_assessment():
    """Rule-based acoustic performance estimate from a building spec.

    Computes approximate Sound Transmission Class (STC) and Outdoor-Indoor
    Transmission Class (OITC) ratings from the facade spec returned by /analyze.
    Higher STC/OITC = better noise insulation. No AI cost — pure rule-based
    calculation using industry standard values per ISO 16717 / ASTM E413.

    2026 green-building trend: acoustic comfort is now a mandatory criterion in
    WELL Building Standard v2, LEED v5 (pilot), and India's NBC Part 8 (2026 Rev.).
    Integrating an acoustic estimate into the design-stage analysis flow prevents
    costly post-occupancy remediation — a pattern rapidly adopted by arch-tech
    platforms in 2026.

    Request body: same spec dict as returned by /analyze
    Returns: { stc, oitc, rating, barriers, recommendations, note }
    """
    spec = request.get_json(silent=True) or {}

    material  = str(spec.get("facade_material", "concrete")).lower()
    window_cols = int(spec.get("window_cols", 5))
    window_rows = int(spec.get("window_rows", 10))
    floors      = max(1, int(spec.get("floors", 10)))
    height      = float(spec.get("height", 40))
    has_balconies = bool(spec.get("has_balconies", False))
    roof         = str(spec.get("roof", "flat")).lower()

    # ── Base STC by primary facade material ────────────────────────────────
    # Values based on ASTM E413 typical assemblies (single-pane glass ~27,
    # double-pane ~35, concrete wall 200mm ~50, brick 220mm ~52).
    stc_base = {
        "glass":    32,   # curtain wall — single pane assumption; double = +6
        "concrete": 50,   # 200mm RC wall
        "brick":    52,   # 220mm clay brick
        "stone":    54,   # 200mm stone masonry — dense, excellent mass
        "mixed":    42,   # average of glass and masonry components
    }.get(material, 44)

    # ── Glazing ratio penalty ────────────────────────────────────────────────
    # Window area as fraction of total facade.
    # High glazing → facade STC approaches glazing STC (weak link).
    facade_area   = 2 * (float(spec.get("width", 35)) + float(spec.get("depth", 28))) * height
    window_area   = min(window_cols * window_rows * 1.5 * 1.8, facade_area * 0.85)
    glazing_ratio = window_area / max(facade_area, 1.0)

    # Weighted STC: glazing pulls composite down toward glass STC (32)
    stc_composite = stc_base * (1 - glazing_ratio) + 32 * glazing_ratio
    stc_composite = round(stc_composite)

    # ── OITC (better for low-frequency traffic noise than STC) ──────────────
    # OITC is typically 3–7 points below STC for masonry; similar for glass.
    oitc = max(20, stc_composite - 5)

    # ── Adjustments ─────────────────────────────────────────────────────────
    adjustments: list[tuple[str, int]] = []

    if glazing_ratio > 0.5:
        adj = -round((glazing_ratio - 0.5) * 20)
        adjustments.append((f"High glazing ratio ({glazing_ratio:.0%}) reduces facade STC significantly", adj))
        stc_composite += adj; oitc += adj

    if has_balconies:
        adjustments.append(("Balconies create flanking paths around the primary facade", -3))
        stc_composite -= 3; oitc -= 3

    if roof == "flat":
        adjustments.append(("Flat roof with no parapet reduces high-frequency attenuation on upper floors", -2))
        stc_composite -= 2

    if height > 60:
        adjustments.append(("High-rise: upper floors exposed to unobstructed wind noise and aircraft noise", -2))
        stc_composite -= 2; oitc -= 2

    stc_composite = max(20, min(65, stc_composite))
    oitc          = max(15, min(60, oitc))

    # ── Rating ───────────────────────────────────────────────────────────────
    if stc_composite >= 55:
        rating, label = "Excellent", "Minimal intrusion from most external noise sources."
    elif stc_composite >= 50:
        rating, label = "Good",      "Suitable for residential/office in moderate urban noise."
    elif stc_composite >= 45:
        rating, label = "Adequate",  "Acceptable for offices near medium-traffic roads."
    elif stc_composite >= 38:
        rating, label = "Fair",      "Speech intelligible through facade; significant urban noise intrusion."
    else:
        rating, label = "Poor",      "Loud speech and traffic noise clearly audible indoors."

    # ── Barriers / recommendations ───────────────────────────────────────────
    barriers: list[str] = []
    recommendations: list[str] = []

    if material == "glass":
        barriers.append("All-glass curtain wall has inherently low mass — primary acoustic weakness.")
        recommendations.append("Specify double-glazed units (6/12/6mm) to raise facade STC from ~32 to ~38; triple-glazing (STC ~43) for high-noise zones.")

    if glazing_ratio > 0.4:
        barriers.append(f"Window-to-wall ratio {glazing_ratio:.0%} means glass dominates the composite STC.")
        recommendations.append("Reduce WWR below 40% or upgrade to laminated acoustic glazing (STC 45+ achievable with 6.4mm PVB laminate).")

    if has_balconies:
        barriers.append("Balcony slabs create airborne sound flanking paths around the primary wall.")
        recommendations.append("Install acoustic balustrades (min. 1.2m high, 10kg/m²) to interrupt the flanking path.")

    if oitc < 35:
        barriers.append(f"OITC {oitc} — low-frequency traffic and aircraft noise will be audible indoors.")
        recommendations.append("Add viscoelastic damping strips between glazing and mullion frame to improve low-frequency performance by 3–5 dB.")

    if not barriers:
        barriers.append("No critical acoustic barriers detected for this facade typology.")
    if not recommendations:
        recommendations.append("Facade appears acoustically reasonable for its typology — focus on party-wall and floor/ceiling assemblies.")

    return jsonify({
        "stc":          stc_composite,
        "oitc":         oitc,
        "rating":       rating,
        "rating_label": label,
        "glazing_ratio_pct": round(glazing_ratio * 100, 1),
        "adjustments":  [{"description": d, "delta_stc": v} for d, v in adjustments],
        "barriers":     barriers,
        "recommendations": recommendations,
        "standards": {
            "stc_reference": "ASTM E413 — Sound Transmission Class (higher = quieter indoors)",
            "oitc_reference": "ASTM E1332 — Outdoor-Indoor Transmission Class (better for traffic/aircraft noise)",
            "well_minimum":   "WELL Building Standard v2 Feature 72 requires STC ≥ 45 for exterior walls in occupied spaces",
        },
        "note": (
            "Indicative estimate based on facade spec only. "
            "Commission an acoustic engineer with site-specific noise measurements for WELL/LEED compliance."
        ),
    })


@app.route("/daylight-analysis", methods=["POST"])
def daylight_analysis():
    """Compute daylight performance metrics for a building facade.

    2026 sustainable design trend: daylight autonomy and climate-based
    daylight modelling (CBDM) are now required inputs for WELL v2, LEED v4.1,
    and BREEAM Excellent ratings. This endpoint estimates key metrics from
    the extracted building spec so architects get an early-stage compliance
    signal without running a full Radiance/EnergyPlus simulation.

    Request JSON fields (all optional — sensible defaults applied):
      width (m), depth (m), height (m), floors (int),
      floor_height (m), window_cols (int), window_rows (int),
      facade_material (str), has_balconies (bool),
      latitude (float, default 40.7), orientation (deg clockwise from N, default 0)

    Response includes: estimated WWR, daylight factor %, sun-hours by facade,
    sDA/ASE approximations, and prioritised recommendations.
    """
    data = request.get_json(silent=True) or {}

    width        = float(data.get("width", 30))
    depth        = float(data.get("depth", 20))
    height       = float(data.get("height", 30))
    floors       = int(data.get("floors", 8))
    floor_height = float(data.get("floor_height", height / max(floors, 1)))
    window_cols  = int(data.get("window_cols", 6))
    window_rows  = int(data.get("window_rows", floors))
    material     = str(data.get("facade_material", "glass")).lower()
    has_balconies = bool(data.get("has_balconies", False))
    latitude     = float(data.get("latitude", 40.7))
    orientation  = float(data.get("orientation", 0)) % 360  # deg CW from north

    # ── Window-to-Wall Ratio (WWR) ────────────────────────────────────────────
    # Estimate each window as 1.5m wide × 1.2m tall (standard curtain-wall module)
    win_w, win_h = 1.5, 1.2
    facade_area = width * height
    glazed_area = min(window_cols * window_rows * win_w * win_h, facade_area)
    wwr = round(glazed_area / max(facade_area, 1), 3)

    # ── Visible Light Transmittance by material ───────────────────────────────
    vlt_map = {"glass": 0.70, "concrete": 0.0, "brick": 0.0, "stone": 0.0, "mixed": 0.40}
    vlt = vlt_map.get(material, 0.50)

    # ── Daylight Factor estimate (CIE standard overcast sky, simplified CIBSE) ─
    # DF% ≈ (WWR × VLT × 100) / (1 + 0.5 × depth/width) — accounts for room depth
    room_depth_factor = 1 + 0.5 * min(depth / max(width, 1), 3.0)
    df = round((wwr * vlt * 100) / room_depth_factor, 2)

    # Balcony overhang penalty: each 1.2m balcony slab reduces DF ~15%
    if has_balconies:
        df = round(df * 0.85, 2)

    # CIBSE/WELL DF thresholds
    if df >= 5.0:
        df_rating, df_label = "excellent", "Target exceeded — consider glare control"
    elif df >= 2.0:
        df_rating, df_label = "good", "WELL Daylight Concept compliant (≥2% DF in 75% of spaces)"
    elif df >= 1.0:
        df_rating, df_label = "adequate", "Marginal — add supplemental daylight strategies"
    else:
        df_rating, df_label = "poor", "Below acceptable minimum — redesign facade or add rooflights"

    # ── Annual sun-hours by facade orientation ────────────────────────────────
    # Simple solar geometry approximation for 4 cardinal facades:
    #   South (lat < 60°): highest; North: lowest; E/W: symmetric mid-range.
    # Scaled from published TMY data for mid-latitudes.
    lat_rad = abs(latitude)
    south_base = max(1200 - lat_rad * 12, 600)
    north_base = max(400 - lat_rad * 4,  150)
    ew_base    = max(900 - lat_rad * 8,  400)

    # Rotate: which actual compass direction does each logical facade face?
    # orientation=0 → primary facade faces south; +90 → faces west, etc.
    angles = {
        "primary":  (180 + orientation) % 360,
        "rear":     orientation % 360,
        "left":     (90  + orientation) % 360,
        "right":    (270 + orientation) % 360,
    }

    def _sun_hours(bearing: float) -> int:
        # 0=N, 90=E, 180=S, 270=W
        if 135 <= bearing <= 225:
            return round(south_base)
        if bearing < 45 or bearing > 315:
            return round(north_base)
        return round(ew_base)

    sun_hours = {face: _sun_hours(ang) for face, ang in angles.items()}

    # ── sDA approximation (LEED v4.1 Daylight credit) ─────────────────────────
    # sDA300/50% threshold: 300 lux for ≥50% of occupied hours in ≥55% of floor area.
    # Approximate: sDA% ≈ (WWR_primary × VLT × 300) / 3 clamped to [0,100]
    sda_raw = min((wwr * vlt * 300) / 3.0, 100)
    # Depth penalty: open-plan rooms beyond 6m from window rarely reach 300 lux
    depth_penalty = max(0, (depth - 6) * 2)
    sda = round(max(0, sda_raw - depth_penalty), 1)
    sda_compliant = sda >= 55

    # ── ASE (Annual Sunlight Exposure) glare check ────────────────────────────
    # ASE1000/250: % floor area receiving >1000 lux direct sun for >250 h/yr.
    # High WWR + south/west facade → glare risk.
    south_facing = 135 <= angles["primary"] <= 225 or 225 <= angles["primary"] <= 315
    ase_risk = "high" if (wwr > 0.55 and south_facing) else "moderate" if wwr > 0.40 else "low"

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = []
    if df < 2.0:
        recs.append("Increase WWR to 35–45% to achieve WELL Daylight Concept minimum 2% DF.")
    if df > 5.0:
        recs.append("Install external shading devices (brise-soleil or fins) to prevent glare while maintaining daylighting.")
    if has_balconies and df < 3.0:
        recs.append("Balcony overhangs are reducing daylight factor; consider perforated balustrades or reduce slab depth to 0.8m.")
    if ase_risk == "high":
        recs.append("ASE glare risk is high — specify electrochromic glazing or external louvers on south/west facades to limit direct sun penetration.")
    if sda < 55:
        recs.append(f"Estimated sDA {sda}% is below LEED v4.1 Option 1 target (55%). Consider light shelves or high-VLT glass (VLT ≥ 0.70).")
    if depth > 12 and wwr < 0.40:
        recs.append("Deep floor plate (>12m) with low WWR will leave central zones unlit — add interior clerestories or light wells.")
    if not recs:
        recs.append("Daylight performance appears adequate for this facade typology and orientation.")

    return jsonify({
        "window_to_wall_ratio":    wwr,
        "glazed_area_m2":          round(glazed_area, 1),
        "facade_area_m2":          round(facade_area, 1),
        "daylight_factor_pct":     df,
        "df_rating":               df_rating,
        "df_label":                df_label,
        "sda_55_300":              sda,
        "sda_compliant":           sda_compliant,
        "ase_risk":                ase_risk,
        "sun_hours_by_facade":     sun_hours,
        "primary_orientation_deg": round(angles["primary"], 1),
        "recommendations":         recs,
        "standards": {
            "df_reference":  "CIBSE LG10 / WELL v2 Feature 57 — min 2% DF in 75% of primary spaces",
            "sda_reference": "IES LM-83 sDA300/50% — LEED v4.1 EQ Daylight credit requires sDA ≥ 55%",
            "ase_reference": "IES LM-83 ASE1000/250 — must be < 10% floor area for LEED credit",
        },
        "note": (
            "Indicative metrics only. Run a full CBDM simulation (Radiance/EnergyPlus + Rhino/Grasshopper) "
            "with local TMY weather data for WELL/LEED/BREEAM compliance documentation."
        ),
    })


@app.route("/occupant-load", methods=["POST"])
def occupant_load():
    """Estimate occupant load and fire egress requirements from a building spec.

    Applies IBC (International Building Code) Section 1004 occupant load factors
    and Section 1005 egress width rules to the spec returned by /analyze.
    Pure arithmetic — no AI cost. Useful for early-stage code compliance checking
    before a fire engineer is engaged.

    2026 context: India's National Building Code Part 4 (Fire & Life Safety, Rev. 2026)
    now mandates occupant load documentation at the design approval stage.
    Integrating this check alongside energy, wind, and acoustic analysis gives
    architects a complete multi-discipline triage at the sketch stage.

    Request body: same spec dict returned by /analyze, plus optional:
      use_type: "office" | "retail" | "residential" | "assembly" | "education"
                (default: "office")

    Returns: occupant load per floor, total, required exit width, staircase count,
             and corridor width per NBC/IBC.
    """
    spec = request.get_json(silent=True) or {}

    width    = float(spec.get("width",  35))
    depth    = float(spec.get("depth",  28))
    floors   = max(1, int(spec.get("floors", 14)))
    has_podium   = bool(spec.get("has_podium", False))
    podium_floors = max(0, int(spec.get("podium_floors", 0)))
    use_type = str(spec.get("use_type", "office")).lower()

    floor_area_m2 = width * depth

    # ── IBC Table 1004.5 occupant load factors (gross m² per person) ─────────
    # "Assembly" covers auditoriums, banquet halls (0.65 m²/person standing,
    # 1.4 m²/person fixed seating). Using 1.4 as the conservative seated estimate.
    load_factors = {
        "office":      9.3,   # IBC: 9.3 m²/person for business occupancies
        "retail":      2.8,   # IBC: 2.8 m²/person for mercantile (ground floor)
        "residential": 18.6,  # IBC: 18.6 m²/person for residential
        "assembly":    1.4,   # IBC: 1.4 m²/person for assembly with fixed seating
        "education":   1.9,   # IBC: 1.9 m²/person for classrooms
    }
    load_factor = load_factors.get(use_type, 9.3)

    occupants_per_floor = max(1, round(floor_area_m2 / load_factor))
    total_occupants     = occupants_per_floor * floors

    if has_podium and podium_floors > 0:
        podium_area = floor_area_m2 * 1.3 * 1.3  # podium is wider
        podium_occupants = max(1, round(podium_area / load_factor)) * podium_floors
        total_occupants += podium_occupants
    else:
        podium_occupants = 0

    # ── IBC Section 1005.1 egress width ──────────────────────────────────────
    # For sprinklered buildings (assumed): 5 mm per occupant for stairways,
    # 3.8 mm per occupant for level exits and corridors.
    # For non-sprinklered: 7.6 mm per stairway, 5 mm per corridor.
    # Assume sprinklered (common for buildings > 4 floors per NBC Part 4).
    sprinklered = floors > 4
    stair_width_per_person_mm  = 5.0 if sprinklered else 7.6
    corridor_width_per_person_mm = 3.8 if sprinklered else 5.0

    # Total required exit stair width (mm) — stairs must serve all occupants above grade
    required_stair_width_mm    = round(occupants_per_floor * stair_width_per_person_mm)
    required_corridor_width_mm = round(occupants_per_floor * corridor_width_per_person_mm)

    # Minimum stair width per IBC: 1 118 mm (44 in) for > 50 occupants; 914 mm otherwise
    min_stair_width_mm = 1118 if occupants_per_floor > 50 else 914

    # Number of required staircases — each stair has a practical usable width of
    # ~1 200 mm (one 44-in stair). Divide total required width by stair width.
    stair_count = max(2, -(-required_stair_width_mm // min_stair_width_mm))  # ceiling division, minimum 2

    # NBC Part 4 (2026) adds: minimum 2 exits for any floor > 50 occupants.
    if occupants_per_floor > 50:
        stair_count = max(2, stair_count)

    # Travel distance limit (IBC Table 1017.2): office/education = 61 m sprinklered;
    # retail/assembly = 76 m sprinklered.
    travel_distance_limit_m = 76 if use_type in ("retail", "assembly") else 61

    # Approximate max travel distance on this floor plate (diagonal of floor plan)
    import math
    diagonal_m = round(math.sqrt(width ** 2 + depth ** 2), 1)
    travel_ok  = diagonal_m <= travel_distance_limit_m

    # ── Emergency lighting / exit sign area (indicative) ─────────────────────
    # NBC Part 4: one emergency luminaire per 250 m² of floor area (minimum 1 per floor)
    emergency_lights_per_floor = max(1, math.ceil(floor_area_m2 / 250))

    # ── Refuge area (NBC Part 4 for high-rise > 30 m) ────────────────────────
    building_height = float(spec.get("height", 55))
    needs_refuge_area = building_height > 30
    refuge_area_m2_per_floor = round(0.3 * occupants_per_floor) if needs_refuge_area else 0

    # ── Code compliance summary ───────────────────────────────────────────────
    issues: list[str] = []
    notes:  list[str] = []

    if stair_count > 4:
        issues.append(f"High staircase count ({stair_count}) suggests the floor plate is very densely loaded for {use_type} — consider splitting into smaller fire compartments.")
    if not travel_ok:
        issues.append(f"Floor diagonal {diagonal_m} m may exceed IBC travel distance limit of {travel_distance_limit_m} m for {use_type} — locate exit stairs at both ends of the floor plate.")
    if use_type == "assembly" and floors > 1:
        issues.append("Assembly occupancies on upper floors require additional exit discharge analysis — IBC Section 1028.")
    if not sprinklered:
        notes.append("Building assumed non-sprinklered (≤4 floors) — higher egress widths apply. Adding sprinklers can reduce required stair width by 33%.")
    else:
        notes.append("Sprinkler system assumed (>4 floors per NBC Part 4) — 5 mm/person stair width factor applied.")
    if needs_refuge_area:
        notes.append(f"Building > 30 m requires a refuge area of ~{refuge_area_m2_per_floor} m² per floor (NBC Part 4 Cl. 4.13).")
    if not issues:
        notes.append("No critical egress issues detected at this floor plate size and occupancy type.")

    return jsonify({
        "use_type":                   use_type,
        "floor_area_m2":              round(floor_area_m2, 1),
        "load_factor_m2_per_person":  load_factor,
        "occupants_per_floor":        occupants_per_floor,
        "occupied_floors":            floors,
        "total_occupants":            total_occupants,
        "podium_occupants":           podium_occupants,
        "egress": {
            "required_stair_width_mm":    required_stair_width_mm,
            "required_corridor_width_mm": required_corridor_width_mm,
            "min_single_stair_width_mm":  min_stair_width_mm,
            "staircase_count_required":   stair_count,
            "sprinklered_assumption":     sprinklered,
        },
        "travel_distance": {
            "floor_diagonal_m":       diagonal_m,
            "code_limit_m":           travel_distance_limit_m,
            "within_limit":           travel_ok,
        },
        "emergency_lights_per_floor": emergency_lights_per_floor,
        "refuge_area": {
            "required":            needs_refuge_area,
            "area_m2_per_floor":   refuge_area_m2_per_floor,
        },
        "compliance_issues": issues,
        "notes": notes,
        "note": (
            "Indicative occupant load and egress estimate per IBC 2021 / NBC 2026 Part 4. "
            "Commission a licensed fire engineer for building permit submission."
        ),
    })


@app.route("/thermal-comfort", methods=["POST"])
def thermal_comfort():
    """Estimate ASHRAE 55 thermal comfort (PMV/PPD) from building spec + environment.

    PMV (Predicted Mean Vote) measures thermal sensation on a −3 (Cold) to +3 (Hot)
    scale; PPD (Predicted Percentage Dissatisfied) converts it to a dissatisfaction
    rate. ASHRAE 55-2020 targets PMV within ±0.5 (PPD < 10%) for compliant spaces.

    2026 context: WELL Building Standard v2 Feature 54 and LEED v5 (pilot) now require
    PMV/PPD documentation at the design stage — making this calculation a standard
    step in the architectural analysis workflow alongside energy, daylight, and acoustic
    assessments. This endpoint uses the Fanger steady-state model (ISO 7730:2005).

    Request body (JSON) — all fields optional, sensible defaults for a typical office:
      air_temp_c (float, default 22): indoor air temperature (°C)
      mean_radiant_temp_c (float): mean radiant temperature; defaults to air_temp_c
                                   adjusted for facade material's solar gain tendency
      air_velocity_ms (float, default 0.1): mean air velocity (m/s)
      relative_humidity_pct (float, default 50): relative humidity (%)
      metabolic_rate (float, default 1.2): met units (1.0=seated/still, 1.2=office work,
                                            2.0=walking, 3.0=fast walk)
      clothing_insulation (float, default 0.7): clo units (0.5=light summer, 0.7=office,
                                                  1.0=business suit, 1.5=heavy winter)
      facade_material (str): from building spec — used to adjust MRT estimate when
                              mean_radiant_temp_c is not explicitly provided

    Returns: PMV, PPD, sensation label, ASHRAE 55 compliance, operative temperature,
             and tailored recommendations.
    """
    import math

    data = request.get_json(silent=True) or {}

    # Environmental inputs
    ta  = float(data.get("air_temp_c",          22))
    va  = max(0.0, float(data.get("air_velocity_ms",      0.1)))
    rh  = max(0.0, min(100.0, float(data.get("relative_humidity_pct", 50))))
    met = max(0.7, float(data.get("metabolic_rate",       1.2)))
    clo = max(0.0, float(data.get("clothing_insulation",  0.7)))

    # Mean radiant temperature: default to air_temp_c adjusted by facade solar gain.
    # Glass facades absorb more solar radiation and re-radiate it as longwave heat —
    # adding 2–4 °C above air temperature on a sunny day is a standard rule of thumb.
    material = str(data.get("facade_material", "glass")).lower()
    mrt_offsets = {"glass": 3.0, "concrete": 1.0, "mixed": 1.5, "brick": 0.0, "stone": -0.5}
    default_tr = ta + mrt_offsets.get(material, 1.0)
    tr = float(data.get("mean_radiant_temp_c", default_tr))

    # Operative temperature = (air_temp + mean_radiant_temp) / 2 — simplified ASHRAE
    t_op = (ta + tr) / 2

    # ── Fanger PMV (ISO 7730 simplified, no iterative tcl convergence) ────────
    M  = met * 58.15    # metabolic rate W/m²
    W  = 0.0            # external mechanical work (seated/office: ≈ 0)
    icl = 0.155 * clo   # clothing thermal resistance (m²·K/W)

    # Clothing surface area factor (Fanger 1970)
    fcl = 1.05 + 0.645 * icl if icl <= 0.078 else 1.05 + 0.645 * icl

    # Water vapour partial pressure (kPa) — Antoine equation
    pa = (rh / 100.0) * math.exp(16.6536 - 4030.183 / (ta + 235.0))

    # Clothing surface temperature (single-pass linear estimate)
    tcl = ta + (35.5 - ta) / (3.5 * icl + 0.1)

    # Convective heat transfer coefficient (max of forced and natural)
    hcf = 12.1 * math.sqrt(va)           # forced convection
    hcn = 2.38 * abs(tcl - ta) ** 0.25   # natural convection
    hc  = max(hcf, hcn)

    # Fanger PMV formula
    pmv_raw = (0.303 * math.exp(-0.036 * M) + 0.028) * (
        (M - W)
        - 3.05e-3 * (5733 - 6.99 * (M - W) - pa * 1000)
        - 0.42  * ((M - W) - 58.15)
        - 1.7e-5 * M * (5867 - pa * 1000)
        - 0.0014 * M * (34 - ta)
        - 3.96e-8 * fcl * ((tcl + 273) ** 4 - (tr + 273) ** 4)
        - fcl * hc * (tcl - ta)
    )
    pmv = round(max(-3.0, min(3.0, pmv_raw)), 2)

    # PPD from PMV (ISO 7730 Eq. 5)
    ppd = round(100 - 95 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2), 1)
    ppd = max(5.0, min(100.0, ppd))

    # Thermal sensation scale
    sensation_map = [
        (-3.0, -2.5, "Cold"),
        (-2.5, -1.5, "Cool"),
        (-1.5, -0.5, "Slightly cool"),
        (-0.5,  0.5, "Neutral (comfortable)"),
        ( 0.5,  1.5, "Slightly warm"),
        ( 1.5,  2.5, "Warm"),
        ( 2.5,  4.0, "Hot"),
    ]
    sensation = "Neutral (comfortable)"
    for lo, hi, label in sensation_map:
        if lo <= pmv < hi:
            sensation = label
            break

    ashrae_compliant = -0.5 <= pmv <= 0.5

    # Recommendations
    recs: list[str] = []
    if pmv > 1.0:
        recs.append(f"Cooling needed: lower air temperature to ~{ta - abs(pmv) * 1.5:.1f}°C or increase air velocity to 0.3–0.5 m/s.")
        if material == "glass":
            recs.append("Add external shading or high-performance low-e glazing to reduce solar radiant heat gain (main driver of elevated MRT).")
    elif pmv > 0.5:
        recs.append("Slightly warm: consider raising air velocity to 0.15–0.25 m/s via ceiling fans — perceived temperature drops ~1°C per 0.1 m/s increase.")
    elif pmv < -1.0:
        recs.append(f"Heating needed: raise air temperature to ~{ta + abs(pmv) * 1.5:.1f}°C or add local radiant panels.")
        if va > 0.2:
            recs.append("Draught is amplifying cold sensation — reduce supply air velocity or reposition diffusers away from occupied zones.")
    elif pmv < -0.5:
        recs.append("Slightly cool: increase heating setpoint by 1–2°C or supply warmer air.")
    if ppd > 20:
        recs.append(f"PPD {ppd}% exceeds WELL v2 Feature 54 target (< 20% dissatisfied). Prioritise both temperature and air velocity adjustments.")
    if not recs:
        recs.append("Thermal environment is within ASHRAE 55-2020 / ISO 7730 comfort zone. Maintain current HVAC setpoints.")

    return jsonify({
        "pmv":                   pmv,
        "ppd":                   ppd,
        "sensation":             sensation,
        "ashrae55_compliant":    ashrae_compliant,
        "operative_temperature_c": round(t_op, 1),
        "inputs": {
            "air_temp_c":              ta,
            "mean_radiant_temp_c":     round(tr, 1),
            "air_velocity_ms":         va,
            "relative_humidity_pct":   rh,
            "metabolic_rate_met":      met,
            "clothing_insulation_clo": clo,
            "facade_material":         material,
        },
        "recommendations": recs,
        "standards": {
            "pmv_reference":  "ISO 7730:2005 / ASHRAE 55-2020 — PMV ±0.5 = compliant comfort zone",
            "ppd_target":     "WELL v2 Feature 54 — PPD < 20% in ≥ 90% of occupied spaces",
            "well_note":      "WELL v2 also requires adaptive comfort analysis (ASHRAE 55 Section 5.4) for naturally ventilated spaces",
        },
        "note": (
            "Fanger steady-state PMV model (ISO 7730:2005). "
            "For mixed-mode, naturally ventilated, or transient scenarios use the "
            "ASHRAE 55-2020 adaptive model or dynamic thermal simulation (EnergyPlus/IDA ICE)."
        ),
    })


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
