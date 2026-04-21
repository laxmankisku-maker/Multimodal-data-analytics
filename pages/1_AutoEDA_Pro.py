import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import base64
from datetime import datetime

# ─── PDF Export ───────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔬 AutoEDA Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & Background ── */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #111827;
}
[data-testid="stHeader"] {
    background-color: #ffffff;
    border-bottom: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}

/* ── Title ── */
.main-title {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -1px;
    color: #111827;
}
.main-title span {
    background: linear-gradient(90deg, #111827, #374151);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label { color: #6b7280 !important; font-weight: 600; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 2rem !important;
    font-weight: 800;
}

/* ── Buttons ── */
.stButton > button {
    background: #1e3a5f;
    color: #ffffff;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
    width: 100%;
    padding: 0.55rem 1rem;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: #16304f;
    border-color: #16304f;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(30,58,95,0.35);
}
.stDownloadButton > button {
    background: #1e3a5f;
    color: #ffffff;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
    width: 100%;
    padding: 0.55rem 1rem;
}
.stDownloadButton > button:hover {
    background: #16304f;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(30,58,95,0.35);
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #f3f4f6;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e5e7eb;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 7px;
    color: #6b7280;
    font-weight: 600;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1e3a5f !important;
    color: #ffffff !important;
}

/* ── Divider ── */
hr { border-color: #e5e7eb; }

/* ── Code block ── */
[data-testid="stCode"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* ── Section Headers ── */
.section-header {
    background: #eef2f7;
    border-left: 4px solid #1e3a5f;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0 0.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a5f;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #9ca3af;
    font-size: 0.8rem;
    border-top: 1px solid #e5e7eb;
    margin-top: 3rem;
}

/* ── AI box ── */
.ai-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #111827;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem;
    margin: 0.5rem 0;
    color: #374151;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Success / Info ── */
[data-testid="stSuccess"] { border-radius: 8px; }
[data-testid="stInfo"]    { border-radius: 8px; }
[data-testid="stWarning"] { border-radius: 8px; }
[data-testid="stError"]   { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
for key, default in [
    ("action", None),
    ("gemini_api_key", ""),
    ("chat_history", []),
    ("figures", []),
    ("analysis_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ──────────────────────────────────────────────────────────────────
CHART_BG = "#ffffff"
AXIS_COLOR = "#374151"
GRID_COLOR = "#f3f4f6"
TEXT_COLOR = "#111827"

def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(CHART_BG)
    axes = ax_list if ax_list else fig.get_axes()
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.tick_params(labelcolor=AXIS_COLOR)
        ax.xaxis.label.set_color(AXIS_COLOR)
        ax.yaxis.label.set_color(AXIS_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d1d5db")
        ax.grid(True, color=GRID_COLOR, alpha=0.8, linewidth=0.7)
    return fig

def save_fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=CHART_BG)
    buf.seek(0)
    return buf.getvalue()

def log_analysis(title, content=""):
    st.session_state.analysis_log.append({
        "title": title, "content": content,
        "time": datetime.now().strftime("%H:%M:%S")
    })

def add_figure(fig, caption=""):
    st.session_state.figures.append({"fig": save_fig_bytes(fig), "caption": caption})

def load_dataset(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    try:
        if ext == "csv":       return pd.read_csv(uploaded_file)
        elif ext in ["xlsx","xls"]: return pd.read_excel(uploaded_file)
        elif ext == "json":    return pd.read_json(uploaded_file)
        elif ext == "xml":     return pd.read_xml(uploaded_file)
        else: st.error("❌ Unsupported format"); return None
    except Exception as e:
        st.error(f"❌ {e}"); return None

# ─── AI ───────────────────────────────────────────────────────────────────────
def get_gemini_client():
    key = st.session_state.gemini_api_key
    if not key: return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"❌ Gemini error: {e}"); return None

def generate_pandas_code(df, query, model):
    schema = "\n".join([f"  {c} ({t})" for c, t in df.dtypes.items()])
    sample = df.head(3).to_string(index=False)
    prompt = f"""You are an expert Python data analyst.

Dataset schema:
{schema}

Sample rows:
{sample}

User question: {query}

Rules:
- Output ONLY valid Python code, no markdown fences, no explanation outside code
- Use dataframe 'df' (already loaded), also import numpy as np if needed
- Store final answer in variable called 'result'
- For plots: create fig with matplotlib/seaborn, assign to 'fig', also set result=fig
- Use clean professional palettes like 'Blues', 'Greys', 'viridis', or 'tab10'
- Set figure facecolor '#ffffff', text color '#111827'
- Do NOT call plt.show()
- For summary/descriptive textual answers: result = a markdown-formatted string with bullet points, bold headings, and clear structure using **bold**, - bullets, ## headers etc.
- For numeric single answers: result = "answer string"
- For tabular data: result = dataframe or series
- If the question asks for summary, insights, or explanation — always return a rich markdown string as result
"""
    resp = model.generate_content(prompt)
    code = resp.text.strip().replace("```python","").replace("```","").strip()
    return code

def run_code(code, df):
    local = {"df": df, "pd": pd, "np": np, "sns": sns, "plt": plt,
             "fig": None, "result": None}
    try:
        exec(code, {}, local)
        return local.get("result"), local.get("fig"), None
    except Exception as e:
        return None, None, str(e)

# ─── PDF ──────────────────────────────────────────────────────────────────────
def generate_pdf(df, log, figures):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=50, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=22,
                              textColor=colors.HexColor("#6366f1"), alignment=TA_CENTER)
    h1_s = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13,
                           textColor=colors.HexColor("#4338ca"), spaceBefore=12)
    n_s = ParagraphStyle("N", parent=styles["Normal"], fontSize=9, leading=13)
    sm_s = ParagraphStyle("S", parent=styles["Normal"], fontSize=7,
                           textColor=colors.HexColor("#475569"))

    story = []
    story.append(Paragraph("AutoEDA Pro — Data Analysis Report", title_s))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}", sm_s))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#6366f1")))
    story.append(Spacer(1, 0.25*inch))

    story.append(Paragraph("Dataset Overview", h1_s))
    meta = [
        ["Metric", "Value"],
        ["Rows", str(df.shape[0])],
        ["Columns", str(df.shape[1])],
        ["Missing Values", str(df.isna().sum().sum())],
        ["Numeric Columns", str(len(df.select_dtypes("number").columns))],
        ["Categorical Columns", str(len(df.select_dtypes("object").columns))],
        ["Memory", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB"],
    ]
    t = Table(meta, colWidths=[2.5*inch, 3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#312e81")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f3ff"),colors.HexColor("#ede9fe")]),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#c4b5fd")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(t); story.append(Spacer(1, 0.25*inch))

    story.append(Paragraph("Statistical Summary", h1_s))
    desc = df.describe(include="all").round(2).reset_index()
    dd = [desc.columns.tolist()] + desc.values.tolist()
    dd = [[str(v)[:15] for v in row] for row in dd]
    cw = min(1.1*inch, 6.5*inch/max(len(dd[0]),1))
    td = Table(dd, colWidths=[cw]*len(dd[0]))
    td.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#312e81")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),6.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f5f3ff"),colors.HexColor("#ede9fe")]),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#c4b5fd")),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(td)

    if log:
        story.append(PageBreak())
        story.append(Paragraph("Analysis History", h1_s))
        for item in log:
            story.append(Paragraph(f"[{item['time']}] {item['title']}", n_s))
            if item["content"]:
                story.append(Paragraph(item["content"], sm_s))
            story.append(Spacer(1, 4))

    if figures:
        story.append(PageBreak())
        story.append(Paragraph("Charts & Visualizations", h1_s))
        for i, fd in enumerate(figures):
            img_buf = io.BytesIO(fd["fig"])
            img = RLImage(img_buf, width=6*inch, height=3.5*inch)
            story.append(img)
            if fd["caption"]:
                story.append(Paragraph(fd["caption"], sm_s))
            story.append(Spacer(1, 0.15*inch))
            if (i+1) % 2 == 0: story.append(PageBreak())

    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#c4b5fd")))
    story.append(Paragraph(
        "AutoEDA Pro  •  Powered by Gemini AI & Streamlit  •  Built with love",
        ParagraphStyle("ft", parent=sm_s, alignment=TA_CENTER)
    ))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:3rem;'>📊</div>
        <div style='font-size:1.3rem;font-weight:800;
                    background:linear-gradient(90deg,#a855f7,#6366f1);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            AutoEDA Pro
        </div>
        <div style='color:#64748b;font-size:0.8rem;'>Intelligent Data Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.page_link("Home.py", label="← Back to Home", icon="🏠")
    st.divider()

    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("API Key", value=st.session_state.gemini_api_key,
                             type="password", placeholder="AIza...",
                             label_visibility="collapsed")
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
    if st.session_state.gemini_api_key:
        st.success("✅ API Key saved")
    else:
        st.info("💡 Enter key to unlock AI features")

    st.divider()
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "CSV, Excel, JSON, XML", type=["csv","xlsx","xls","json","xml"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("""
### ℹ️ About AutoEDA Pro

**AutoEDA Pro** is a no-code intelligent data analysis platform:

- 🔍 **Auto EDA** – instant stats & distributions
- 🤖 **Gemini AI** – ask in plain English
- 📈 **Beautiful Charts** – dark themed visuals
- 📄 **PDF Export** – full analysis reports
- 🔗 **Bivariate & Multivariate** analysis

---
**Version:** 2.0 Pro  
**AI:** Gemini 2.5 Flash  
**Stack:** Streamlit · Pandas · Seaborn
    """)
    st.divider()
    st.markdown("<div style='color:#475569;font-size:0.75rem;text-align:center;'>"
                "Made with ❤️ using Python & Gemini</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🔬 AutoEDA Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">✨ Upload your dataset — analyze, visualize & export with AI in seconds</div>',
            unsafe_allow_html=True)

st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80",
         use_container_width=True,
         caption="📊 Transforming raw data into actionable insights")
st.divider()

if uploaded_file is None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style='text-align:center;padding:3rem;background:linear-gradient(135deg,#1e1b4b,#312e81);
                    border-radius:20px;border:1px dashed #6366f1;'>
            <div style='font-size:4rem;'>📂</div>
            <div style='font-size:1.3rem;color:#a5b4fc;font-weight:700;margin:1rem 0;'>
                No Dataset Loaded
            </div>
            <div style='color:#64748b;'>
                Upload a CSV, Excel, JSON, or XML file<br>from the sidebar to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    df = load_dataset(uploaded_file)

    if df is not None:
        st.success(f"✅ **{uploaded_file.name}** loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Metrics row
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("🗂️ Rows", f"{df.shape[0]:,}")
        m2.metric("📋 Columns", df.shape[1])
        m3.metric("❓ Missing", f"{df.isna().sum().sum():,}")
        m4.metric("🔢 Numeric", len(df.select_dtypes("number").columns))
        m5.metric("🔤 Categ.", len(df.select_dtypes("object").columns))
        m6.metric("💾 Memory", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
        st.divider()

        # Action buttons
        st.markdown('<div class="section-header">⚙️ Select Analysis Module</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
        btns = [(c1,"📋 Overview","overview"),(c2,"📊 Describe","describe"),
                (c3,"❓ Missing","missing"),(c4,"📉 Univariate","univariate"),
                (c5,"🔗 Bivariate","bivariate"),(c6,"🌐 Multivar.","multivariate"),
                (c7,"🔝 Top/Bot","top_bottom"),(c8,"📦 GroupBy","groupby")]
        for col,label,key in btns:
            with col:
                if st.button(label, key=f"btn_{key}"):
                    st.session_state.action = key

        c9,c10,_ = st.columns([2,2,4])
        with c9:
            if st.button("🤖 Ask AI (Gemini)", key="btn_ai"):
                st.session_state.action = "ai"
        with c10:
            if st.button("🗃️ Query History", key="btn_hist"):
                st.session_state.action = "history"

        st.divider()
        action = st.session_state.action

        # ── OVERVIEW ──────────────────────────────────────────────────────────
        if action == "overview":
            st.markdown('<div class="section-header">🔍 Dataset Overview</div>', unsafe_allow_html=True)
            tab1,tab2,tab3 = st.tabs(["📄 Sample Data","🏷️ Data Types","🧠 Stats"])
            with tab1:
                n = st.slider("Rows to preview", 5, 50, 10)
                st.dataframe(df.head(n), use_container_width=True)
            with tab2:
                dtype_df = pd.DataFrame({
                    "Column": df.dtypes.index, "Type": df.dtypes.values.astype(str),
                    "Non-Null": df.count().values, "Null": df.isna().sum().values,
                    "Unique": df.nunique().values,
                })
                st.dataframe(dtype_df, use_container_width=True)
            with tab3:
                st.dataframe(df.describe(include="all").T, use_container_width=True)
            log_analysis("Dataset Overview viewed")

        # ── DESCRIBE ──────────────────────────────────────────────────────────
        elif action == "describe":
            st.markdown('<div class="section-header">📊 Statistical Summary</div>', unsafe_allow_html=True)
            st.dataframe(df.describe(include="all").round(3), use_container_width=True)
            log_analysis("Statistical Summary viewed")

        # ── MISSING ───────────────────────────────────────────────────────────
        elif action == "missing":
            st.markdown('<div class="section-header">❓ Missing Value Analysis</div>', unsafe_allow_html=True)
            miss = df.isna().sum().reset_index()
            miss.columns = ["Column","Missing Count"]
            miss["Missing %"] = (miss["Missing Count"]/len(df)*100).round(2)
            miss = miss[miss["Missing Count"]>0].sort_values("Missing Count",ascending=False)
            if miss.empty:
                st.success("🎉 No missing values found!")
            else:
                st.dataframe(miss, use_container_width=True)
                fig, ax = plt.subplots(figsize=(10,4))
                sns.barplot(data=miss, x="Column", y="Missing %", palette="magma", ax=ax)
                ax.set_title("Missing Value % by Column")
                plt.xticks(rotation=45, ha="right")
                style_fig(fig); st.pyplot(fig); add_figure(fig,"Missing Values")
            log_analysis("Missing Analysis viewed")

        # ── UNIVARIATE ────────────────────────────────────────────────────────
        elif action == "univariate":
            st.markdown('<div class="section-header">📉 Univariate Analysis</div>', unsafe_allow_html=True)
            num_cols = df.select_dtypes("number").columns.tolist()
            cat_cols = df.select_dtypes("object").columns.tolist()
            if num_cols:
                st.markdown("#### 🔢 Numerical Distributions")
                for i in range(0, len(num_cols), 3):
                    row_cols = st.columns(3)
                    for j, cn in enumerate(num_cols[i:i+3]):
                        with row_cols[j]:
                            fig, ax = plt.subplots(figsize=(5,3.5))
                            sns.histplot(df[cn].dropna(), kde=True, color="#6366f1", ax=ax, alpha=0.8)
                            ax.set_title(cn, fontsize=11)
                            style_fig(fig); st.pyplot(fig); add_figure(fig,f"Distribution: {cn}")
            if cat_cols:
                st.markdown("#### 🔤 Categorical Distributions")
                for i in range(0, len(cat_cols), 3):
                    row_cols = st.columns(3)
                    for j, cn in enumerate(cat_cols[i:i+3]):
                        with row_cols[j]:
                            top = df[cn].value_counts().nlargest(15)
                            fig, ax = plt.subplots(figsize=(5,3.5))
                            sns.barplot(x=top.values, y=top.index, palette="viridis", ax=ax)
                            ax.set_title(cn, fontsize=11)
                            style_fig(fig); st.pyplot(fig); add_figure(fig,f"Count: {cn}")
            log_analysis("Univariate Analysis completed")

        # ── BIVARIATE ─────────────────────────────────────────────────────────
        elif action == "bivariate":
            st.markdown('<div class="section-header">🔗 Bivariate Analysis</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: col1 = st.selectbox("X Axis", df.columns, key="bv1")
            with c2: col2 = st.selectbox("Y Axis", [c for c in df.columns if c!=col1], key="bv2")
            if col1 and col2:
                n1 = pd.api.types.is_numeric_dtype(df[col1])
                n2 = pd.api.types.is_numeric_dtype(df[col2])
                if n1 and n2:
                    corr = df[[col1,col2]].corr().iloc[0,1]
                    st.metric("📈 Pearson Correlation", f"{corr:.4f}")
                    tab1,tab2,tab3 = st.tabs(["Scatter","Line","Hex Density"])
                    with tab1:
                        fig,ax = plt.subplots(figsize=(8,5))
                        sns.regplot(x=df[col1],y=df[col2],ax=ax,
                                    scatter_kws={"alpha":0.5,"color":"#a855f7"},
                                    line_kws={"color":"#f59e0b"})
                        style_fig(fig); st.pyplot(fig); add_figure(fig,f"Scatter: {col1} vs {col2}")
                    with tab2:
                        fig,ax = plt.subplots(figsize=(8,5))
                        sns.lineplot(x=df[col1].sort_values(),y=df[col2],ax=ax,color="#6366f1")
                        style_fig(fig); st.pyplot(fig)
                    with tab3:
                        fig,ax = plt.subplots(figsize=(8,5))
                        ax.hexbin(df[col1],df[col2],gridsize=30,cmap="plasma")
                        style_fig(fig); st.pyplot(fig)
                else:
                    fig,ax = plt.subplots(figsize=(10,5))
                    try: sns.boxplot(x=df[col1],y=df[col2],palette="viridis",ax=ax)
                    except: sns.barplot(x=df[col1],y=df[col2],palette="viridis",ax=ax)
                    plt.xticks(rotation=45,ha="right")
                    style_fig(fig); st.pyplot(fig); add_figure(fig,f"Bivariate: {col1} vs {col2}")
            log_analysis("Bivariate Analysis", f"{col1} vs {col2}")

        # ── MULTIVARIATE ──────────────────────────────────────────────────────
        elif action == "multivariate":
            st.markdown('<div class="section-header">🌐 Multivariate Analysis</div>', unsafe_allow_html=True)
            all_num = df.select_dtypes("number").columns.tolist()
            cols = st.multiselect("Select columns (min 2)", all_num,
                                  default=all_num[:min(5,len(all_num))])
            if len(cols) >= 2:
                tab1,tab2 = st.tabs(["🔥 Correlation Heatmap","📐 Pair Plot"])
                with tab1:
                    fig,ax = plt.subplots(figsize=(10,7))
                    sns.heatmap(df[cols].corr(),annot=True,fmt=".2f",cmap="coolwarm",
                                linewidths=0.5,linecolor="#1e1b4b",ax=ax,annot_kws={"size":9})
                    ax.set_title("Correlation Matrix",fontsize=14)
                    style_fig(fig); st.pyplot(fig); add_figure(fig,"Correlation Heatmap")
                with tab2:
                    with st.spinner("Rendering pair plot…"):
                        fig = sns.pairplot(df[cols].dropna(),
                                           plot_kws={"alpha":0.5,"color":"#6366f1"},
                                           diag_kws={"color":"#a855f7"})
                        fig.fig.patch.set_facecolor("#ffffff")
                        st.pyplot(fig); add_figure(fig.fig,"Pair Plot")
            log_analysis("Multivariate Analysis")

        # ── TOP / BOTTOM ──────────────────────────────────────────────────────
        elif action == "top_bottom":
            st.markdown('<div class="section-header">🔝 Top / Bottom Values</div>', unsafe_allow_html=True)
            col = st.selectbox("Select Column", df.select_dtypes("number").columns)
            n = st.slider("N", 5, 100, 10)
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("#### ⬆️ Top Values")
                st.dataframe(df.nlargest(n,col), use_container_width=True)
            with c2:
                st.markdown("#### ⬇️ Bottom Values")
                st.dataframe(df.nsmallest(n,col), use_container_width=True)
            log_analysis("Top/Bottom Analysis", f"{col}, N={n}")

        # ── GROUPBY ───────────────────────────────────────────────────────────
        elif action == "groupby":
            st.markdown('<div class="section-header">📦 GroupBy Analysis</div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1: grp = st.selectbox("Group By", df.columns, key="gb_g")
            with c2: agg = st.selectbox("Value Column", df.select_dtypes("number").columns, key="gb_a")
            with c3: func = st.selectbox("Function", ["mean","sum","max","min","count","median","std"])
            res = df.groupby(grp)[agg].agg(func).reset_index()
            res.columns = [grp, f"{func}({agg})"]
            res = res.sort_values(res.columns[1], ascending=False)
            st.dataframe(res, use_container_width=True)
            fig,ax = plt.subplots(figsize=(10,5))
            sns.barplot(x=grp,y=res.columns[1],data=res,palette="viridis",ax=ax)
            ax.set_title(f"{func.capitalize()} of {agg} by {grp}")
            plt.xticks(rotation=45,ha="right")
            style_fig(fig); st.pyplot(fig); add_figure(fig,f"GroupBy: {grp} → {func}({agg})")
            log_analysis("GroupBy Analysis", f"{grp} → {func}({agg})")

        # ── AI ────────────────────────────────────────────────────────────────
        elif action == "ai":
            st.markdown('<div class="section-header">🤖 AI Data Analyst (Gemini)</div>', unsafe_allow_html=True)

            # ── Always show dataset preview so user knows what to ask ──
            with st.expander("📋 Your Dataset Preview (2 sample rows)", expanded=True):
                st.dataframe(df.sample(min(2, len(df))), use_container_width=True)
                col_list = " • ".join([f"`{c}`" for c in df.columns.tolist()])
                st.markdown(f"**Columns:** {col_list}")

            if not st.session_state.gemini_api_key:
                st.warning("⚠️ Enter your Gemini API key in the sidebar to use AI features.")
            else:
                # ── Dynamic tips based on actual column names ──
                num_cols = df.select_dtypes("number").columns.tolist()
                cat_cols = df.select_dtypes("object").columns.tolist()
                tips = []
                if num_cols:
                    tips.append(f"📊 *Summarize the `{num_cols[0]}` column*")
                if len(num_cols) >= 2:
                    tips.append(f"🔗 *What is the correlation between `{num_cols[0]}` and `{num_cols[1]}`?*")
                if cat_cols:
                    tips.append(f"📦 *Which `{cat_cols[0]}` has the highest average {num_cols[0] if num_cols else 'count'}?*")
                if num_cols:
                    tips.append(f"📈 *Plot the distribution of `{num_cols[0]}`*")
                if cat_cols and num_cols:
                    tips.append(f"🏆 *Show top 10 rows by `{num_cols[0]}`*")
                tips.append("📝 *Give me a full dataset summary with key insights*")

                tips_html = "  &nbsp;|&nbsp;  ".join(tips[:4])
                st.markdown(f"""
                <div class="ai-box">
                💡 <b>Ask anything about your data in plain English!</b><br><br>
                <b>💬 Suggestions for this dataset:</b><br>
                {tips_html}
                </div>
                """, unsafe_allow_html=True)

                query = st.text_input("💬 Your question", placeholder="e.g. Summarize this dataset with key insights…", key="ai_q")
                if query:
                    mdl = get_gemini_client()
                    if mdl:
                        with st.spinner("🧠 Generating analysis…"):
                            code = generate_pandas_code(df, query, mdl)
                        danger = ["import os","import sys","subprocess","__import__","open(","exec(","eval("]
                        if any(d in code for d in danger):
                            st.error("🚫 Unsafe code detected — blocked.")
                        else:
                            with st.expander("🔍 Generated Code", expanded=False):
                                st.code(code, language="python")
                            result, fig, error = run_code(code, df)
                            if error:
                                st.error(f"❌ Error: {error}")
                            else:
                                st.success("✅ Analysis complete!")
                                if fig is not None:
                                    style_fig(fig)
                                    st.pyplot(fig)
                                    add_figure(fig, f"AI: {query[:60]}")
                                elif isinstance(result, (pd.DataFrame, pd.Series)):
                                    st.dataframe(result, use_container_width=True)
                                elif isinstance(result, str):
                                    # Render as markdown to support bullet points, bold, headers
                                    st.markdown(result)
                                elif result is not None:
                                    st.write(result)
                                else:
                                    try: st.pyplot(plt.gcf())
                                    except: st.info("No output returned.")
                            st.session_state.chat_history.append({
                                "q": query, "code": code, "error": error or "",
                                "time": datetime.now().strftime("%H:%M:%S")
                            })
                            log_analysis(f"AI: {query[:60]}", error or "Success")

        # ── HISTORY ───────────────────────────────────────────────────────────
        elif action == "history":
            st.markdown('<div class="section-header">🗃️ AI Query History</div>', unsafe_allow_html=True)
            if not st.session_state.chat_history:
                st.info("No AI queries yet. Use the 'Ask AI' button to get started!")
            else:
                for i, h in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"[{h['time']}] {h['q'][:80]}", expanded=(i==0)):
                        st.code(h["code"], language="python")
                        if h["error"]: st.error(h["error"])

        # ── EXPORT ────────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-header">📤 Export Analysis</div>', unsafe_allow_html=True)
        e1,e2,e3 = st.columns(3)

        with e1:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download Dataset (CSV)", data=csv_buf.getvalue(),
                               file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")
        with e2:
            desc_buf = io.StringIO()
            df.describe(include="all").to_csv(desc_buf)
            st.download_button("⬇️ Download Stats (CSV)", data=desc_buf.getvalue(),
                               file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")
        with e3:
            if st.button("📄 Generate Full PDF Report"):
                import time as _time
                report_start = _time.time()
                progress_placeholder = st.empty()
                timer_placeholder = st.empty()

                # ── Autopilot: run all analyses and collect figures ──
                autopilot_figures = []
                autopilot_log = list(st.session_state.analysis_log)

                def _save(fig, caption):
                    autopilot_figures.append({"fig": save_fig_bytes(fig), "caption": caption})
                    plt.close(fig)

                steps = [
                    "📊 Running statistical summary…",
                    "❓ Analysing missing values…",
                    "📉 Building univariate distributions…",
                    "🌐 Computing correlation matrix…",
                    "📦 Generating GroupBy charts…",
                    "📝 Compiling PDF…",
                ]
                total = len(steps)

                # Step 1 – describe (no chart needed)
                progress_placeholder.progress(1/total, text=steps[0])
                autopilot_log.append({"title":"Statistical Summary (auto)","content":"","time":datetime.now().strftime("%H:%M:%S")})

                # Step 2 – missing
                progress_placeholder.progress(2/total, text=steps[1])
                miss = df.isna().sum().reset_index()
                miss.columns = ["Column","Missing Count"]
                miss["Missing %"] = (miss["Missing Count"]/len(df)*100).round(2)
                miss = miss[miss["Missing Count"]>0].sort_values("Missing Count",ascending=False)
                if not miss.empty:
                    fig,ax = plt.subplots(figsize=(10,4))
                    sns.barplot(data=miss, x="Column", y="Missing %", palette="Blues_d", ax=ax)
                    ax.set_title("Missing Value % by Column"); plt.xticks(rotation=45,ha="right")
                    style_fig(fig); _save(fig,"Missing Values by Column")
                autopilot_log.append({"title":"Missing Values (auto)","content":f"{miss.shape[0]} cols with missing data","time":datetime.now().strftime("%H:%M:%S")})

                # Step 3 – univariate
                progress_placeholder.progress(3/total, text=steps[2])
                num_c = df.select_dtypes("number").columns.tolist()
                cat_c = df.select_dtypes("object").columns.tolist()
                for cn in num_c[:6]:
                    fig,ax = plt.subplots(figsize=(6,3.5))
                    sns.histplot(df[cn].dropna(), kde=True, color="#1e3a5f", ax=ax, alpha=0.75)
                    ax.set_title(f"Distribution: {cn}", fontsize=11)
                    style_fig(fig); _save(fig, f"Distribution: {cn}")
                for cn in cat_c[:4]:
                    top = df[cn].value_counts().nlargest(10)
                    fig,ax = plt.subplots(figsize=(6,3.5))
                    sns.barplot(x=top.values, y=top.index, palette="Blues_d", ax=ax)
                    ax.set_title(f"Top values: {cn}", fontsize=11)
                    style_fig(fig); _save(fig, f"Count: {cn}")
                autopilot_log.append({"title":"Univariate Analysis (auto)","content":f"{len(num_c)} numeric, {len(cat_c)} categorical","time":datetime.now().strftime("%H:%M:%S")})

                # Step 4 – correlation heatmap
                progress_placeholder.progress(4/total, text=steps[3])
                if len(num_c) >= 2:
                    fig,ax = plt.subplots(figsize=(10,7))
                    sns.heatmap(df[num_c].corr(), annot=True, fmt=".2f", cmap="Blues",
                                linewidths=0.5, ax=ax, annot_kws={"size":8})
                    ax.set_title("Correlation Matrix", fontsize=14)
                    style_fig(fig); _save(fig,"Correlation Heatmap")
                autopilot_log.append({"title":"Correlation Matrix (auto)","content":"","time":datetime.now().strftime("%H:%M:%S")})

                # Step 5 – groupby charts for first cat × first num
                progress_placeholder.progress(5/total, text=steps[4])
                if cat_c and num_c:
                    grp_res = df.groupby(cat_c[0])[num_c[0]].mean().reset_index()
                    grp_res.columns = [cat_c[0], f"mean({num_c[0]})"]
                    grp_res = grp_res.sort_values(grp_res.columns[1], ascending=False).head(15)
                    fig,ax = plt.subplots(figsize=(10,5))
                    sns.barplot(x=cat_c[0], y=grp_res.columns[1], data=grp_res, palette="Blues_d", ax=ax)
                    ax.set_title(f"Mean of {num_c[0]} by {cat_c[0]}"); plt.xticks(rotation=45, ha="right")
                    style_fig(fig); _save(fig, f"GroupBy: {cat_c[0]} → mean({num_c[0]})")
                autopilot_log.append({"title":"GroupBy Analysis (auto)","content":"","time":datetime.now().strftime("%H:%M:%S")})

                # Step 6 – build PDF
                progress_placeholder.progress(6/total, text=steps[5])
                all_figs = autopilot_figures + list(st.session_state.figures)
                pdf_bytes = generate_pdf(df, autopilot_log, all_figs)

                elapsed = _time.time() - report_start
                progress_placeholder.empty()
                timer_placeholder.markdown(
                    f"<div style='color:#1e3a5f;font-size:0.85rem;font-weight:600;"
                    f"padding:0.4rem 0.8rem;background:#eef2f7;border-radius:6px;"
                    f"border-left:3px solid #1e3a5f;margin-top:0.3rem;'>"
                    f"⏱️ Report generated in <b>{elapsed:.1f} seconds</b> — "
                    f"{len(all_figs)} charts included</div>",
                    unsafe_allow_html=True
                )
                st.download_button("⬇️ Download Full PDF Report", data=pdf_bytes,
                                   file_name=f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                   mime="application/pdf")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    📊 <b>AutoEDA Pro v2.0</b> &nbsp;|&nbsp;
    Built with ❤️ using Streamlit · Pandas · Seaborn · Gemini AI
    &nbsp;|&nbsp; 2025
    <br><br>
    ⚡ <em>Tip: Enter your Gemini API key in the sidebar to unlock natural language queries</em>
</div>
""", unsafe_allow_html=True)