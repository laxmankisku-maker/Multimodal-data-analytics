import streamlit as st
import google.generativeai as genai
import json
import re
from PIL import Image
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image EDA — Visual Insights",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background: #f8f9fb; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8eaf0;
}

/* Cards */
.insight-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.insight-card h4 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #f0f1f4;
}

/* Metric tiles */
.metric-tile {
    background: #f3f4f8;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.metric-label {
    font-size: 11px;
    color: #9ca3af;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 3px;
}
.metric-value {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
}
.metric-sub {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
}

/* Badges */
.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px 3px 2px 0;
}
.badge-teal   { background:#d1fae5; color:#065f46; }
.badge-blue   { background:#dbeafe; color:#1e40af; }
.badge-amber  { background:#fef3c7; color:#92400e; }
.badge-purple { background:#ede9fe; color:#5b21b6; }
.badge-coral  { background:#fee2e2; color:#991b1b; }
.badge-green  { background:#dcfce7; color:#166534; }
.badge-gray   { background:#f3f4f6; color:#374151; }

/* Progress bars */
.bar-wrap { margin-bottom: 8px; }
.bar-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #374151;
    margin-bottom: 3px;
    font-weight: 500;
}
.bar-track {
    height: 8px;
    background: #f0f1f4;
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* Color swatches */
.swatch-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.swatch-box {
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 600;
    min-width: 70px;
    text-align: center;
}

/* Spatial grid */
.spatial-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 5px;
    margin-top: 6px;
}
.spatial-cell {
    background: #f3f4f8;
    border-radius: 7px;
    padding: 8px 10px;
    font-size: 11px;
    color: #6b7280;
    text-align: center;
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.spatial-cell.focus {
    background: #d1fae5;
    color: #065f46;
    font-weight: 600;
}

/* Tags */
.tag-cloud { display: flex; flex-wrap: wrap; gap: 6px; }
.tag {
    background: #f3f4f8;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #374151;
}

/* Use case items */
.use-item {
    display: flex;
    gap: 12px;
    background: #f9fafb;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    align-items: flex-start;
}
.use-num {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: #3b82f6;
    color: white;
    font-size: 12px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.use-title { font-size: 13px; font-weight: 600; color: #111827; }
.use-reason { font-size: 12px; color: #6b7280; margin-top: 2px; }

/* Section header */
.section-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 50%;
    background: #3b82f6;
    color: white;
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
    flex-shrink: 0;
}

/* Query result box */
.query-result {
    background: #ffffff;
    border: 1.5px solid #3b82f6;
    border-radius: 14px;
    padding: 20px 22px;
    margin-top: 16px;
    box-shadow: 0 2px 8px rgba(59,130,246,0.08);
}
.query-result h4 {
    color: #1e40af;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}

/* Upload area */
.upload-hint {
    background: #eff6ff;
    border: 2px dashed #93c5fd;
    border-radius: 14px;
    padding: 32px 20px;
    text-align: center;
    color: #1e40af;
    font-size: 15px;
    font-weight: 500;
}

/* Hero title */
.hero { text-align: center; padding: 8px 0 24px; }
.hero h1 { font-size: 28px; font-weight: 800; color: #111827; margin-bottom: 6px; }
.hero p { font-size: 15px; color: #6b7280; }

/* Confidence bar */
.conf-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    background: #d1fae5;
    color: #065f46;
}
</style>
""", unsafe_allow_html=True)


# ── Gemini setup ──────────────────────────────────────────────────────────────
def get_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Prompts ───────────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """
You are an expert visual analysis engine. Analyze this image across all 10 dimensions.
Return ONLY a valid raw JSON object — no markdown fences, no explanation, no preamble.

{
  "scene_detection": {
    "primary_subject": "main subject name",
    "scientific_name": "latin/scientific name or null",
    "secondary_objects": ["list", "of", "other", "objects"],
    "scene_type": "Indoor / Outdoor / Studio / Abstract",
    "confidence": 95
  },
  "color_palette": {
    "swatches": [
      { "hex": "#5DCAA5", "name": "Teal", "percent": 38 },
      { "hex": "#7DBB6E", "name": "Muted green", "percent": 28 },
      { "hex": "#8B5E3C", "name": "Russet brown", "percent": 18 },
      { "hex": "#F2C200", "name": "Gold yellow", "percent": 10 },
      { "hex": "#3D2B1A", "name": "Dark wood", "percent": 6 }
    ]
  },
  "composition": {
    "rule_of_thirds": "Weak | Moderate | Strong",
    "depth_of_field": "Shallow | Medium | Deep",
    "leading_lines": "Describe any leading lines or diagonal elements",
    "balance": "Symmetric | Asymmetric | Dynamic",
    "framing_notes": "Brief note on overall framing quality"
  },
  "mood": {
    "overall": "Descriptive mood phrase",
    "color_temperature": "Warm | Cool | Neutral | Mixed",
    "energy_level": "Low | Medium | High",
    "season": "Spring | Summer | Autumn | Winter | Not applicable",
    "emotional_tone": "Descriptive emotional tone"
  },
  "quality": {
    "sharpness": "Poor | Fair | Good | Excellent",
    "noise_level": "High | Medium | Low | Very Low",
    "brightness": "Underexposed | Well-lit | Slightly overexposed | Overexposed",
    "aspect_ratio": "e.g. 3:4 portrait",
    "lighting_type": "e.g. Natural diffused",
    "camera_angle": "e.g. Eye-level",
    "estimated_focal_length": "e.g. Telephoto 300-500mm"
  },
  "subject_prominence": [
    { "subject": "name", "prominence_percent": 72 },
    { "subject": "name", "prominence_percent": 16 },
    { "subject": "name", "prominence_percent": 12 }
  ],
  "style_classification": {
    "image_type": "Photograph | Illustration | CGI | Abstract | Screenshot | Diagram",
    "sub_genre": "e.g. Wildlife photography / Street / Portrait / Product",
    "post_processing": "None | Light | Moderate | Heavy",
    "style_notes": "Brief note on photographic or artistic style"
  },
  "spatial_layout": {
    "grid": [
      { "zone": "top-left",    "content": "brief description", "is_focus": false },
      { "zone": "top-center",  "content": "brief description", "is_focus": true  },
      { "zone": "top-right",   "content": "brief description", "is_focus": false },
      { "zone": "mid-left",    "content": "brief description", "is_focus": false },
      { "zone": "mid-center",  "content": "brief description", "is_focus": true  },
      { "zone": "mid-right",   "content": "brief description", "is_focus": false },
      { "zone": "bot-left",    "content": "brief description", "is_focus": false },
      { "zone": "bot-center",  "content": "brief description", "is_focus": false },
      { "zone": "bot-right",   "content": "brief description", "is_focus": false }
    ]
  },
  "tags": ["10 to 20 descriptive keywords about the image"],
  "use_cases": [
    { "title": "Use case title", "reason": "Why this image suits this use" },
    { "title": "Use case title", "reason": "Why this image suits this use" },
    { "title": "Use case title", "reason": "Why this image suits this use" },
    { "title": "Use case title", "reason": "Why this image suits this use" },
    { "title": "Use case title", "reason": "Why this image suits this use" }
  ]
}
"""


def run_analysis(model, image: Image.Image) -> dict:
    response = model.generate_content([image, ANALYSIS_PROMPT])
    text = response.text
    clean = re.sub(r"```json|```", "", text).strip()
    return json.loads(clean)


def run_query(model, image: Image.Image, query: str) -> str:
    prompt = f"""You are an expert image analyst. The user has already seen a full automated analysis 
of this image. Now answer their specific question in detail, clearly and insightfully.

User question: {query}

Give a thorough, accurate answer based on what you see in the image."""
    response = model.generate_content([image, prompt])
    return response.text


# ── Helper renderers ──────────────────────────────────────────────────────────

def pick_text_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "#111827"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#111827" if lum > 0.5 else "#ffffff"


def bar_html(label: str, pct: float, color: str = "#3b82f6") -> str:
    pct = min(100, max(0, round(pct)))
    return f"""
<div class="bar-wrap">
  <div class="bar-label-row"><span>{label}</span><span>{pct}%</span></div>
  <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>
</div>"""


def badge_html(text: str, style: str = "gray") -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


def render_insight_card(number: int, title: str, content_html: str):
    st.markdown(f"""
<div class="insight-card">
  <h4><span class="section-num">{number}</span>{title}</h4>
  {content_html}
</div>""", unsafe_allow_html=True)


# ── Insight renderers ─────────────────────────────────────────────────────────

def render_scene(d: dict):
    conf = d.get("confidence", 0)
    sci  = d.get("scientific_name") or "—"
    html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
  <div class="metric-tile">
    <div class="metric-label">Primary subject</div>
    <div class="metric-value">{d.get('primary_subject','—')}</div>
    <div class="metric-sub"><i>{sci}</i></div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Scene type</div>
    <div class="metric-value">{d.get('scene_type','—')}</div>
    <div class="metric-sub">Confidence: <span class="conf-badge">{conf}%</span></div>
  </div>
</div>
<div class="metric-tile">
  <div class="metric-label">Secondary objects</div>
  <div style="margin-top:4px;">{' '.join(badge_html(o,'blue') for o in d.get('secondary_objects',[])[:6])}</div>
</div>"""
    render_insight_card(1, "Scene & Object Detection", html)


def render_palette(d: dict):
    swatches = d.get("swatches", [])
    swatch_html = "".join(
        f'<div class="swatch-box" style="background:{s["hex"]};color:{pick_text_color(s["hex"])};">'
        f'{s["name"]}<br>{s["hex"]}<br>{s["percent"]}%</div>'
        for s in swatches
    )
    bars_html = "".join(bar_html(s["name"], s["percent"], s["hex"]) for s in swatches)
    html = f'<div class="swatch-row">{swatch_html}</div>{bars_html}'
    render_insight_card(2, "Color Palette Analysis", html)


def render_composition(d: dict):
    rot_map   = {"Weak": "coral", "Moderate": "amber", "Strong": "green"}
    dof_map   = {"Shallow": "purple", "Medium": "blue", "Deep": "teal"}
    bal_map   = {"Symmetric": "teal", "Asymmetric": "amber", "Dynamic": "green"}
    rot = d.get("rule_of_thirds", "—")
    dof = d.get("depth_of_field", "—")
    bal = d.get("balance", "—")
    html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Rule of thirds</div>
    <div style="margin-top:6px;">{badge_html(rot, rot_map.get(rot,'gray'))}</div>
  </div>
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Depth of field</div>
    <div style="margin-top:6px;">{badge_html(dof, dof_map.get(dof,'gray'))}</div>
  </div>
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Balance</div>
    <div style="margin-top:6px;">{badge_html(bal, bal_map.get(bal,'gray'))}</div>
  </div>
</div>
<div class="metric-tile">
  <div class="metric-label">Leading lines</div>
  <div style="font-size:13px;color:#374151;margin-top:4px;">{d.get('leading_lines','—')}</div>
</div>
<div class="metric-tile" style="margin-top:8px;">
  <div class="metric-label">Framing notes</div>
  <div style="font-size:13px;color:#374151;margin-top:4px;">{d.get('framing_notes','—')}</div>
</div>"""
    render_insight_card(3, "Composition & Visual Hierarchy", html)


def render_mood(d: dict):
    temp_map = {"Warm": "amber", "Cool": "blue", "Neutral": "gray", "Mixed": "purple"}
    eng_map  = {"Low": "teal", "Medium": "amber", "High": "coral"}
    temp = d.get("color_temperature", "—")
    eng  = d.get("energy_level", "—")
    html = f"""
<div class="metric-tile" style="margin-bottom:10px;">
  <div class="metric-label">Overall mood</div>
  <div class="metric-value" style="font-size:16px;">{d.get('overall','—')}</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Color temperature</div>
    <div style="margin-top:6px;">{badge_html(temp, temp_map.get(temp,'gray'))}</div>
  </div>
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Energy level</div>
    <div style="margin-top:6px;">{badge_html(eng, eng_map.get(eng,'gray'))}</div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
  <div class="metric-tile">
    <div class="metric-label">Season feel</div>
    <div style="font-size:13px;font-weight:600;color:#374151;margin-top:4px;">{d.get('season','—')}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Emotional tone</div>
    <div style="font-size:13px;font-weight:600;color:#374151;margin-top:4px;">{d.get('emotional_tone','—')}</div>
  </div>
</div>"""
    render_insight_card(4, "Mood & Aesthetic Analysis", html)


def render_quality(d: dict):
    sharp_map = {"Poor": "coral", "Fair": "amber", "Good": "teal", "Excellent": "green"}
    s = d.get("sharpness", "—")
    html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Sharpness</div>
    <div style="margin-top:6px;">{badge_html(s, sharp_map.get(s,'gray'))}</div>
  </div>
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Noise level</div>
    <div style="margin-top:6px;">{badge_html(d.get('noise_level','—'), 'blue')}</div>
  </div>
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Brightness</div>
    <div style="margin-top:6px;">{badge_html(d.get('brightness','—'), 'amber')}</div>
  </div>
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Aspect ratio</div>
    <div style="font-size:14px;font-weight:700;color:#111827;margin-top:6px;">{d.get('aspect_ratio','—')}</div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
  <div class="metric-tile">
    <div class="metric-label">Lighting</div>
    <div style="font-size:13px;font-weight:600;color:#374151;margin-top:4px;">{d.get('lighting_type','—')}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Camera angle</div>
    <div style="font-size:13px;font-weight:600;color:#374151;margin-top:4px;">{d.get('camera_angle','—')}</div>
  </div>
  <div class="metric-tile" style="grid-column:span 2;">
    <div class="metric-label">Estimated focal length</div>
    <div style="font-size:13px;font-weight:600;color:#374151;margin-top:4px;">{d.get('estimated_focal_length','—')}</div>
  </div>
</div>"""
    render_insight_card(5, "Image Quality & Metadata", html)


def render_prominence(items: list):
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
    bars = "".join(
        bar_html(item["subject"], item["prominence_percent"], colors[i % len(colors)])
        for i, item in enumerate(items)
    )
    render_insight_card(6, "Dominant Subjects — Ranked by Visual Prominence", bars)


def render_style(d: dict):
    proc_map = {"None": "green", "Light": "teal", "Moderate": "amber", "Heavy": "coral"}
    proc = d.get("post_processing", "—")
    html = f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
  <div class="metric-tile">
    <div class="metric-label">Image type</div>
    <div style="font-size:15px;font-weight:700;color:#111827;margin-top:4px;">{d.get('image_type','—')}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Sub-genre</div>
    <div style="font-size:14px;font-weight:600;color:#374151;margin-top:4px;">{d.get('sub_genre','—')}</div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
  <div class="metric-tile" style="text-align:center;">
    <div class="metric-label">Post-processing</div>
    <div style="margin-top:6px;">{badge_html(proc, proc_map.get(proc,'gray'))}</div>
  </div>
  <div class="metric-tile">
    <div class="metric-label">Style notes</div>
    <div style="font-size:12px;color:#6b7280;margin-top:4px;">{d.get('style_notes','—')}</div>
  </div>
</div>"""
    render_insight_card(7, "Style Classification", html)


def render_spatial(d: dict):
    grid = d.get("grid", [])
    zone_order = [
        "top-left","top-center","top-right",
        "mid-left","mid-center","mid-right",
        "bot-left","bot-center","bot-right",
    ]
    zone_map = {z["zone"]: z for z in grid}
    cells = ""
    for zone in zone_order:
        z = zone_map.get(zone, {"content": "—", "is_focus": False})
        cls = "spatial-cell focus" if z.get("is_focus") else "spatial-cell"
        cells += f'<div class="{cls}">{z.get("content","—")}</div>'
    html = f"""
<p style="font-size:12px;color:#9ca3af;margin-bottom:8px;">Green cells = primary visual weight zones</p>
<div class="spatial-grid">{cells}</div>"""
    render_insight_card(8, "Spatial Layout Map (3×3 Grid)", html)


def render_tags(tags: list):
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    html = f'<div class="tag-cloud">{tag_html}</div>'
    render_insight_card(9, "Visual Keywords & Tags", html)


def render_use_cases(items: list):
    rows = "".join(
        f"""<div class="use-item">
          <div class="use-num">{i+1}</div>
          <div><div class="use-title">{item['title']}</div>
          <div class="use-reason">{item['reason']}</div></div>
        </div>"""
        for i, item in enumerate(items)
    )
    render_insight_card(10, "Use-Case & Context Suggestions", rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("Home.py", label="← Back to Home", icon="🏠")
    st.divider()
    st.markdown("## 🔍 Image EDA")
    st.markdown("Visual insights for any unlabeled image — powered by Gemini 2.5 Flash")
    st.divider()

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Get free key at aistudio.google.com/apikey"
    )
    st.caption("🔒 Your key is never stored or sent anywhere except Google's API.")

    st.divider()
    st.markdown("**How to get a free key:**")
    st.markdown("1. Go to [aistudio.google.com](https://aistudio.google.com/apikey)")
    st.markdown("2. Sign in with Google")
    st.markdown("3. Click **Create API Key**")
    st.markdown("4. Paste it above")
    st.divider()
    st.markdown("**Free tier limits:**")
    st.markdown("- ✅ 1,500 requests / day")
    st.markdown("- ✅ 15 requests / minute")
    st.markdown("- ✅ No credit card needed")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🖼 Image EDA — Auto Visual Insights</h1>
  <p>Upload any unlabeled image to get 10 deep insights instantly — like EDA for images.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
<div class="upload-hint">
  📂 Drop an image here (JPG, PNG, WEBP)<br>
  <span style="font-size:13px;opacity:0.7;">Wildlife · Food · Landscape · Product · Art · Architecture — any unlabeled image</span>
</div>""", unsafe_allow_html=True)
    st.stop()

if not api_key:
    st.warning("⬅️ Please enter your Gemini API key in the sidebar to begin analysis.")
    st.stop()

# Load image
image = Image.open(uploaded)
w, h  = image.size

# ── Image preview + run button ────────────────────────────────────────────────
col_img, col_info = st.columns([1, 1], gap="large")

with col_img:
    st.image(image, use_container_width=True, caption=uploaded.name)

with col_info:
    st.markdown("### Image details")
    st.markdown(f"""
| Property | Value |
|---|---|
| Filename | `{uploaded.name}` |
| Dimensions | {w} × {h} px |
| Format | {image.format or uploaded.type.split('/')[-1].upper()} |
| Size | {uploaded.size / 1024:.1f} KB |
| Mode | {image.mode} |
""")

    run_all = st.button("🚀 Run all 10 insights", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### Ask a custom question")
    custom_q = st.text_area(
        "Your question about this image",
        placeholder="e.g. What species of bird is this? What camera was likely used? Is this image suitable for a magazine cover?",
        height=100,
        label_visibility="collapsed",
    )
    run_query_btn = st.button("💬 Ask Gemini", use_container_width=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if run_all or "analysis" not in st.session_state or st.session_state.get("last_file") != uploaded.name:
    if run_all:
        with st.spinner("Gemini is analyzing your image across all 10 dimensions..."):
            try:
                model = get_model(api_key)
                data  = run_analysis(model, image)
                st.session_state["analysis"]  = data
                st.session_state["last_file"] = uploaded.name
                st.session_state["model"]     = model
                st.session_state["image"]     = image
            except json.JSONDecodeError:
                st.error("Gemini returned an unexpected response. Please try again.")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

# ── Render results ────────────────────────────────────────────────────────────
if "analysis" in st.session_state and st.session_state.get("last_file") == uploaded.name:
    data = st.session_state["analysis"]

    st.divider()
    st.markdown("## 📊 All 10 Visual Insights")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        render_scene(data.get("scene_detection", {}))
        render_composition(data.get("composition", {}))
        render_quality(data.get("quality", {}))
        render_style(data.get("style_classification", {}))
        render_tags(data.get("tags", []))

    with col2:
        render_palette(data.get("color_palette", {}))
        render_mood(data.get("mood", {}))
        render_prominence(data.get("subject_prominence", []))
        render_spatial(data.get("spatial_layout", {}))
        render_use_cases(data.get("use_cases", []))

    # Raw JSON expander
    with st.expander("🔧 View raw JSON response"):
        st.json(data)

# ── Custom query ──────────────────────────────────────────────────────────────
if run_query_btn:
    if not custom_q.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Gemini is thinking..."):
            try:
                mdl = st.session_state.get("model") or get_model(api_key)
                img = st.session_state.get("image", image)
                answer = run_query(mdl, img, custom_q.strip())
                st.markdown(f"""
<div class="query-result">
  <h4>💬 Answer to your question</h4>
  <p style="font-size:13px;color:#6b7280;margin-bottom:10px;"><i>"{custom_q.strip()}"</i></p>
  <div style="font-size:14px;color:#111827;line-height:1.7;">{answer.replace(chr(10),'<br>')}</div>
</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Query error: {e}")