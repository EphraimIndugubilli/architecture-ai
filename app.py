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

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from groq import Groq
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/energy-class": {"origins": "*"}, r"/health": {"origins": "*"}, r"/models": {"origins": "*"}})

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
        "current": MODEL,
        "available": [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.3-70b-versatile",
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

    @stream_with_context
    def generate():
        full_text = []
        try:
            client = Groq(api_key=API_KEY)
            stream = client.chat.completions.create(
                model=MODEL,
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
            yield f"data: {json.dumps({'type': 'done', 'spec': spec, 'report': report, 'image': image_data_url})}\n\n"
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
