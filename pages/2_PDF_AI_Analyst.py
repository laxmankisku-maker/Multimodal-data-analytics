import os
import io
import re
import json
import warnings
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as sp_stats
from collections import Counter

# ── suppress gRPC noise ─────────────────────────────────────────────
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
warnings.filterwarnings("ignore")

import google.generativeai as genai
from PyPDF2 import PdfReader
import pdfplumber

st.set_page_config(
    page_title="PDF AI Analyst · Gemini",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.hdr{font-size:2.1rem;font-weight:600;text-align:center;letter-spacing:-.02em;margin-bottom:.2rem;}
.sub{text-align:center;color:#6B7280;font-size:.9rem;margin-bottom:1.4rem;}
.gbadge{display:inline-flex;align-items:center;gap:5px;
  background:linear-gradient(135deg,#4285F4,#0F9D58);color:#fff;
  font-size:11px;font-weight:500;padding:3px 12px;border-radius:20px;}
.badge{display:inline-block;font-size:10px;font-weight:600;
  padding:2px 8px;border-radius:20px;margin-bottom:5px;}
.b-ai {background:#EFF6FF;color:#1D4ED8;}
.b-ext{background:#FEF9C3;color:#854D0E;}
.b-viz{background:#F0FDF4;color:#166534;}
.b-nlp{background:#FDF4FF;color:#7E22CE;}
.b-rep{background:#ECFDF5;color:#065F46;}
.b-qa {background:#FFF7ED;color:#9A3412;}
.aibox{background:#F0F9FF;border-left:3px solid #4285F4;
  border-radius:0 10px 10px 0;padding:12px 16px;margin:10px 0;
  font-size:13px;color:#1E3A5F;line-height:1.7;}
.ailbl{font-size:10px;font-weight:700;color:#4285F4;
  text-transform:uppercase;letter-spacing:.06em;margin:0 0 4px;}
.fh{background:#FEF2F2;border-left:3px solid #EF4444;border-radius:0 8px 8px 0;
  padding:8px 12px;margin:5px 0;font-size:12px;color:#991B1B;}
.fm{background:#FFFBEB;border-left:3px solid #F59E0B;border-radius:0 8px 8px 0;
  padding:8px 12px;margin:5px 0;font-size:12px;color:#92400E;}
.fl{background:#F0FDF4;border-left:3px solid #22C55E;border-radius:0 8px 8px 0;
  padding:8px 12px;margin:5px 0;font-size:12px;color:#166534;}
.qb{background:#EFF6FF;border-radius:0 12px 12px 12px;
  padding:10px 14px;margin-bottom:4px;font-size:13px;color:#1E3A5F;}
.ab{background:#F9FAFB;border:1px solid #E5E7EB;border-radius:12px 0 12px 12px;
  padding:10px 14px;margin-bottom:12px;font-size:13px;color:#111827;}
.ql{font-size:10px;font-weight:600;color:#4285F4;
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px;}
.al{font-size:10px;font-weight:600;color:#0F9D58;
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px;}
</style>
""", unsafe_allow_html=True)

_D = dict(messages=[], processed_file=None, full_text="",
          tables=[], active_option=None, option_result=None, pdf_bytes=None)
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.page_link("Home.py", label="← Back to Home", icon="🏠")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    gemini_key = st.text_input("Gemini API Key", type="password",
                               help="Free at aistudio.google.com/app/apikey")
    if gemini_key:
        st.success("✅ Key set")
        genai.configure(api_key=gemini_key)
    st.markdown("---")
    model_choice = st.selectbox("Gemini model",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        help="2.5-flash = fastest & free")
    st.markdown("---")
    st.markdown("### 🛠️ Stack (all free)")
    st.markdown("""
- **Gemini 2.5 Flash** — AI + table extraction
- **PyPDF2** — text extraction
- **pandas + scipy** — statistics
- **Plotly** — interactive charts
    """)
    st.caption("Free tier: 15 req/min · 1M tokens/min")

st.markdown('<p class="hdr">📊 PDF AI Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload any PDF — 10 AI analyses with real charts · Powered by Gemini</p>',
            unsafe_allow_html=True)
st.markdown('<div style="text-align:center;margin-bottom:1rem">'
            '<span class="gbadge">✦ Gemini — native PDF understanding</span></div>',
            unsafe_allow_html=True)

def _model():
    return genai.GenerativeModel(model_choice)

def _cfg(temp=0.1, tokens=4096):
    return genai.types.GenerationConfig(temperature=temp, max_output_tokens=tokens)

def ask(prompt: str, temp=0.1) -> str:
    try:
        r = _model().generate_content(prompt, generation_config=_cfg(temp))
        return r.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

def ask_pdf(prompt: str, temp=0.1) -> str:
    try:
        part = {"mime_type": "application/pdf", "data": st.session_state.pdf_bytes}
        r = _model().generate_content([part, prompt], generation_config=_cfg(temp))
        return r.text.strip()
    except Exception as e:
        return ask(prompt + "\n\nDocument text:\n" + st.session_state.full_text[:6000], temp)

def smart_extract_tables() -> list:
    prompt = """Read this PDF carefully. Find ALL tables, frequency tables, data tables, and structured datasets.

For EACH table return a JSON object with:
- "title": short descriptive name for this table
- "columns": list of column name strings (use the ACTUAL names from the PDF, e.g. "Number of Trees", "Frequency", "Flavour", "Animal", "Angle")
- "rows": list of rows, each row is a list of values (use actual numbers, not text descriptions)

Return ONLY a JSON array of table objects. No markdown, no explanation, just raw JSON.

Example format:
[
  {
    "title": "Trees frequency table",
    "columns": ["Number of Trees", "Frequency"],
    "rows": [[0, 4], [1, 3], [2, 2], [3, 1], [4, 2], [5, 2], [6, 1]]
  }
]

Important rules:
- Use EXACT numbers from the PDF, do not invent any values
- Column names must match what is written in the PDF
- Each row must have the same number of values as there are columns
- If a cell contains both a number and explanatory text, keep only the number
- Return empty array [] if no tables found"""

    try:
        raw = ask_pdf(prompt, temp=0.0)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group())
        tables = []
        for obj in data:
            cols = obj.get("columns", [])
            rows = obj.get("rows", [])
            if not cols or not rows:
                continue
            df = pd.DataFrame(rows, columns=cols)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            tables.append((obj.get("title", f"Table {len(tables)+1}"), df))
        return tables
    except Exception as e:
        st.warning(f"JSON table extraction error: {e}. Falling back to text parsing.")
        return []

def fallback_text_tables(text: str) -> list:
    tables = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    block = []
    prev_had_data = False

    def flush(block):
        if len(block) < 2:
            return None
        rows = []
        for tokens in block:
            nums = []
            words = []
            for t in tokens:
                t_clean = re.sub(r'\([^)]*\)', '', t).strip()
                try:
                    nums.append(float(t_clean) if '.' in t_clean else int(t_clean))
                except ValueError:
                    if t_clean and len(t_clean) <= 30:
                        words.append(t_clean)
            label = " ".join(words).strip()
            if nums:
                rows.append([label] + nums if label else nums)
        if len(rows) < 2:
            return None
        max_c = max(len(r) for r in rows)
        has_label = any(isinstance(r[0], str) and r[0] for r in rows if len(r) > 1)
        if has_label:
            cols = ["Category"] + [f"Value{i}" for i in range(1, max_c)]
        else:
            cols = [f"Col{i}" for i in range(max_c)]
        padded = [r + [None]*(max_c - len(r)) for r in rows]
        df = pd.DataFrame(padded, columns=cols[:max_c])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        return df

    for line in lines:
        line = re.sub(r'\([\w𝑥𝑓]+\)', '', line).strip()
        tokens = line.split()
        num_count = sum(1 for t in tokens if re.match(r'^-?\d+(?:\.\d+)?°?$', t.rstrip('°')))
        if num_count >= 1 and len(tokens) >= 2:
            block.append(tokens)
            prev_had_data = True
        else:
            if prev_had_data:
                df = flush(block)
                if df is not None and len(df) >= 2 and df.shape[1] >= 2:
                    tables.append((f"Dataset {len(tables)+1}", df))
            block = []
            prev_had_data = False

    df = flush(block)
    if df is not None and len(df) >= 2 and df.shape[1] >= 2:
        tables.append((f"Dataset {len(tables)+1}", df))
    return tables

def process_pdf(uploaded_file):
    raw_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(raw_bytes))
    text = "".join(p.extract_text() or "" for p in reader.pages)
    return text, raw_bytes

def load_tables():
    tables = smart_extract_tables()
    if not tables:
        st.info("ℹ️ Using regex text parser as fallback for table extraction.")
        tables = fallback_text_tables(st.session_state.full_text)
    return tables

C = ["#4285F4","#0F9D58","#F4B400","#DB4437","#7B1FA2",
     "#00ACC1","#E65100","#2E7D32","#1565C0","#AD1457"]

def num_cols(df):
    return [c for c in df.select_dtypes(include="number").columns if df[c].nunique() >= 2]

def cat_cols(df):
    return [c for c in df.select_dtypes(exclude="number").columns
            if df[c].nunique() <= 25 and df[c].astype(str).str.len().mean() <= 40]

def make_bar(df, cat_col, val_col, title, color):
    fig = px.bar(df, x=val_col, y=cat_col, orientation="h", title=title,
                 color_discrete_sequence=[color], template="plotly_white")
    fig.update_layout(height=max(260, len(df)*42), margin=dict(t=50, b=20), showlegend=False)
    return fig

def make_hist(df, col, title, color):
    clean = pd.to_numeric(df[col], errors="coerce").dropna()
    if clean.nunique() < 2:
        return None
    fig = px.histogram(clean, x=col, nbins=min(25, max(5, len(clean)//2+1)),
                       title=title, color_discrete_sequence=[color], template="plotly_white")
    fig.update_layout(height=280, margin=dict(t=50, b=20), showlegend=False)
    return fig

def make_pie(df, cat_col, val_col, title):
    try:
        if val_col and val_col in df.columns:
            vals = pd.to_numeric(df[val_col], errors="coerce")
            data = df.copy()
            data["_v"] = vals
            data = data.dropna(subset=["_v"])
            fig = px.pie(data, values="_v", names=cat_col, title=title,
                         template="plotly_white", color_discrete_sequence=C)
        else:
            vc = df[cat_col].value_counts().head(10)
            fig = px.pie(values=vc.values, names=vc.index.astype(str),
                         title=title, template="plotly_white", color_discrete_sequence=C)
        fig.update_layout(height=340, margin=dict(t=50, b=20))
        return fig
    except Exception:
        return None

def make_scatter(df, xc, yc, color_col, title):
    fig = px.scatter(df, x=xc, y=yc, color=color_col, title=title,
                     template="plotly_white", color_discrete_sequence=C)
    fig.update_layout(height=360, margin=dict(t=50, b=20))
    return fig

def df_stats(df):
    try:
        return df.describe(include="all").round(3).to_string()
    except Exception:
        return ""

def show_ai(text):
    st.markdown('<div class="aibox"><p class="ailbl">✦ Gemini AI Insight</p></div>', unsafe_allow_html=True)
    st.markdown(text)

def show_figs(figs):
    for fig in figs:
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

def run_eda_summary():
    tables = st.session_state.tables
    figs = []
    stats_parts = [f"Table: '{t}'\n{df_stats(df)}" for t, df in tables[:4]]
    stats_str = "\n\n".join(stats_parts) or "No tables."
    prompt = f"""You are a senior data analyst. Read this PDF and produce an Auto EDA Summary.
ONLY report numbers and facts that appear IN this PDF. Do not invent any data.
Provide:
1. Document overview — topic, domain, purpose (2 sentences)
2. Data found — describe each table with its REAL column names and row counts
3. Top 5 key statistics with EXACT numbers from the document
4. Distribution highlights — skew, dominant categories, ranges
5. Top 3 analyst observations
Real extracted table stats:
{stats_str}"""
    ai = ask_pdf(prompt)
    for i, (title, df) in enumerate(tables[:5]):
        nc = num_cols(df)
        cc = cat_cols(df)
        color = C[i % len(C)]
        if cc and nc:
            fig = make_bar(df, cc[0], nc[0], f"'{title}' — {nc[0]} by {cc[0]}", color)
            figs.append(fig)
        elif nc and len(nc) >= 1:
            fig = make_hist(df, nc[0], f"'{title}' — distribution of {nc[0]}", color)
            if fig:
                figs.append(fig)
        elif cc:
            vc = df[cc[0]].value_counts().head(15).reset_index()
            vc.columns = [cc[0], "Count"]
            fig = make_bar(vc, cc[0], "Count", f"'{title}' — {cc[0]} counts", color)
            figs.append(fig)
    return ai, figs

def run_extraction():
    tables = st.session_state.tables
    figs = []
    if not tables:
        ai = ask_pdf("Extract ALL structured data from this PDF. List every table in markdown with actual values.")
        return ai, figs, []
    lines = [f"**{t}** — {df.shape[0]} rows × {df.shape[1]} cols · {df.isnull().mean().mean()*100:.1f}% nulls" for t, df in tables]
    ai = f"### ✅ {len(tables)} table(s) extracted\n\n" + "\n\n".join(lines)
    for _, (title, df) in enumerate(tables[:4]):
        nm = df.isnull().astype(int)
        if nm.values.sum() > 0 and df.shape[1] <= 20:
            fig = px.imshow(nm, title=f"'{title}' · Missing value map",
                            color_continuous_scale=["#4ADE80","#EF4444"],
                            template="plotly_white", aspect="auto")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=min(300, 22*df.shape[0]+80), margin=dict(t=50,b=20))
            figs.append(fig)
    return ai, figs, tables

def run_report():
    tables = st.session_state.tables
    stats = "\n\n".join(f"'{t}':\n{df_stats(df)}" for t, df in tables[:4]) or "No tables."
    prompt = f"""Write a professional EDA report for this PDF with these 6 sections:
# EDA Report
## 1. Executive Summary
## 2. Document & Data Overview
## 3. Data Quality Assessment
## 4. Key Findings & Patterns
## 5. Statistical Highlights
## 6. Recommendations & Next Steps
Rules:
- Only cite numbers actually in this PDF
- Be specific. Each section ≥ 3 sentences.
Real table statistics:
{stats}
PDF text: {st.session_state.full_text[:5000]}"""
    ai = ask_pdf(prompt)
    figs = []
    for i, (title, df) in enumerate(tables[:3]):
        nc = num_cols(df)
        cc = cat_cols(df)
        if cc and nc:
            fig = make_bar(df, cc[0], nc[0], f"'{title}' — {nc[0]} by {cc[0]}", C[i])
            figs.append(fig)
        elif nc:
            fig = make_hist(df, nc[0], f"'{title}' — {nc[0]}", C[i])
            if fig:
                figs.append(fig)
    return ai, figs

def run_charts():
    tables = st.session_state.tables
    figs = []
    num_info = [(t, num_cols(df)) for t, df in tables]
    cat_info = [(t, cat_cols(df)) for t, df in tables]
    prompt = f"""You are a data visualisation expert. Recommend the 5 best charts for this PDF.
For each: chart type · exact column names · insight revealed · why it fits.
Tables with numeric columns: {[(t, nc) for t, nc in num_info if nc]}
Tables with categorical columns: {[(t, cc) for t, cc in cat_info if cc]}
Text sample: {st.session_state.full_text[:2000]}"""
    ai = ask_pdf(prompt)
    for i, (title, df) in enumerate(tables[:4]):
        nc = num_cols(df)
        cc = cat_cols(df)
        if cc and nc:
            fig = make_bar(df, cc[0], nc[0], f"'{title}' — {nc[0]} by {cc[0]}", C[i])
            figs.append(fig)
            fig2 = make_pie(df, cc[0], nc[0], f"'{title}' — composition")
            if fig2:
                figs.append(fig2)
        if len(nc) >= 2:
            fig = make_scatter(df, nc[0], nc[1], cc[0] if cc else None, f"'{title}' — {nc[0]} vs {nc[1]}")
            figs.append(fig)
    return ai, figs

def run_anomaly():
    tables = st.session_state.tables
    figs, flags = [], []
    for title, df in tables[:5]:
        nc = num_cols(df)
        for col in nc:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) < 4:
                continue
            z = np.abs(sp_stats.zscore(clean))
            out_z = int((z > 3).sum())
            np_ = round(df[col].isnull().mean()*100, 1)
            q1, q3 = clean.quantile(.25), clean.quantile(.75)
            iqr = q3 - q1
            out_i = int(((clean < q1-1.5*iqr)|(clean > q3+1.5*iqr)).sum())
            if out_z:
                flags.append(("HIGH" if out_z>3 else "MED", f"'{title}' · '{col}': {out_z} z-score outlier(s) max={clean.max():.2f}"))
            if np_ > 10:
                flags.append(("HIGH" if np_>25 else "MED", f"'{title}' · '{col}': {np_}% missing values"))
            if out_i:
                flags.append(("LOW", f"'{title}' · '{col}': {out_i} IQR outlier(s)"))
        dups = int(df.duplicated().sum())
        if dups:
            flags.append(("MED", f"'{title}': {dups} duplicate rows"))
        if nc:
            melt = (df[nc[:6]].copy().apply(pd.to_numeric, errors="coerce").melt(var_name="Column", value_name="Value").dropna())
            if len(melt) >= 2:
                fig = px.box(melt, x="Column", y="Value", title=f"'{title}' · Box plots",
                             color="Column", template="plotly_white", color_discrete_sequence=C)
                fig.update_layout(height=320, margin=dict(t=50,b=20), showlegend=False)
                figs.append(fig)
    flag_str = "\n".join(f"[{s}] {m}" for s,m in flags) or "No anomalies detected."
    prompt = f"""Anomalies found by statistical tests (z-score, IQR):
{flag_str}
For each issue:
1. Root cause
2. Analytical impact
3. Specific fix
4. Priority: Critical / High / Medium / Low
Also check this text for contradictions:
{st.session_state.full_text[:3000]}"""
    ai = ask_pdf(prompt)
    return ai, figs, flags

def run_text_insights():
    text = st.session_state.full_text
    figs = []
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    SW = {'that','this','with','from','they','have','been','will','were','their','what','when','which','also','into','more','than','then','about','some','there','would','could','should','these','those','very','just','like','each','such','does','where','other','most','both','only','said','make','made','year','years','used','using','data','page','table','chart','number','example','below','above','shown','given','find','note','frequency'}
    filtered = [w for w in words if w not in SW]
    top_kw = Counter(filtered).most_common(20)
    bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
    top_bg = Counter(bigrams).most_common(12)
    if top_kw:
        kdf = pd.DataFrame(top_kw, columns=["Keyword","Count"])
        fig = px.bar(kdf, x="Count", y="Keyword", orientation="h", title="Top 20 keywords by frequency",
                     color="Count", color_continuous_scale="Blues", template="plotly_white")
        fig.update_layout(height=460, margin=dict(t=50,b=20), yaxis=dict(autorange="reversed"))
        figs.append(fig)
    if top_bg:
        bgdf = pd.DataFrame(top_bg, columns=["Bigram","Count"])
        fig = px.bar(bgdf, x="Count", y="Bigram", orientation="h", title="Top bigrams from your PDF",
                     color="Count", color_continuous_scale="Greens", template="plotly_white")
        fig.update_layout(height=380, margin=dict(t=50,b=20), yaxis=dict(autorange="reversed"))
        figs.append(fig)
    prompt = f"""NLP analysis of this PDF.
Top keywords: {[w for w,_ in top_kw]}
Top bigrams: {[b for b,_ in top_bg]}
Provide:
1. Sentiment with 2 text examples
2. Main themes (top 5, one sentence each)
3. Named entities: people, orgs, locations, dates, numbers
4. Domain & sub-domain
5. Intended audience & expertise level
6. 5 key phrases and what they signal
7. 3-sentence plain-English summary for non-expert
Full text: {text[:6000]}"""
    ai = ask_pdf(prompt)
    return ai, figs

def run_correlation():
    figs = []
    target = None
    for _, df in st.session_state.tables:
        nc = num_cols(df)
        if len(nc) >= 2:
            target = df
            break
    if target is None:
        ai = ask_pdf("Identify and explain all relationships and correlations between variables in this PDF using exact values.")
        return ai, figs, None
    nc = num_cols(target)
    num_df = target[nc].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if num_df.shape[1] < 2 or len(num_df) < 3:
        ai = ask_pdf("Describe relationships between data columns in this PDF.")
        return ai, figs, None
    corr = num_df.corr().round(3)
    fig = px.imshow(corr, text_auto=True, title="Correlation matrix — computed from your PDF data",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white", aspect="auto")
    fig.update_layout(height=max(300, 70*len(nc)), margin=dict(t=55,b=20))
    figs.append(fig)
    if 2 <= len(nc) <= 6:
        fig2 = px.scatter_matrix(num_df, title="Scatter matrix — numeric column pairs",
                                  color_discrete_sequence=[C[0]], template="plotly_white")
        fig2.update_traces(marker=dict(size=3, opacity=0.5))
        fig2.update_layout(height=460, margin=dict(t=55,b=20))
        figs.append(fig2)
    prompt = f"""Interpret this real correlation matrix from the uploaded PDF:
{corr.to_string()}
Explain:
1. Strongest positive correlations and meaning in context
2. Strongest negative correlations and implications
3. Surprising or counterintuitive relationships
4. Domain insight from these correlations
5. Which pairs to investigate further"""
    ai = ask_pdf(prompt)
    return ai, figs, corr

def ask_question(question: str) -> str:
    prompt = f"""You are a precise document assistant. Answer ONLY from the PDF.
Cite exact numbers and quote text where helpful.
Say clearly if the answer is not in the document.
Question: {question}
Document text:
{st.session_state.full_text[:8000]}"""
    return ask_pdf(prompt)

def run_data_quality():
    tables = st.session_state.tables
    text = st.session_state.full_text
    figs = []
    ds = dict(words=len(text.split()), chars=len(text), tables=len(tables),
              missing_kw=sum(text.lower().count(w) for w in ["n/a","null","missing","none","not available"]))
    qr = []
    for title, df in tables[:8]:
        nc_ = int(df.isnull().sum().sum())
        dc = int(df.duplicated().sum())
        np_ = round(nc_/max(df.size,1)*100, 1)
        qr.append({"Table": title, "Rows": df.shape[0], "Cols": df.shape[1],
                   "Nulls": nc_, "Null %": np_, "Duplicates": dc, "Completeness %": round(100-np_,1)})
    if qr:
        qdf = pd.DataFrame(qr)
        fig = px.bar(qdf, x="Table", y="Completeness %", title="Data completeness per table",
                     color="Completeness %", color_continuous_scale=["#EF4444","#F59E0B","#22C55E"],
                     range_color=[0,100], template="plotly_white")
        fig.update_layout(height=300, margin=dict(t=50,b=20))
        figs.append(fig)
        fig2 = px.bar(qdf, x="Table", y=["Nulls","Duplicates"], title="Nulls and duplicates per table",
                      barmode="group", template="plotly_white",
                      color_discrete_map={"Nulls":"#EF4444","Duplicates":"#F59E0B"})
        fig2.update_layout(height=300, margin=dict(t=50,b=20))
        figs.append(fig2)
    per = "\n".join(f"'{r['Table']}': {r['Rows']}r×{r['Cols']}c | Nulls:{r['Nulls']}({r['Null %']}%) | Dups:{r['Duplicates']} | Complete:{r['Completeness %']}%" for r in qr) or "No tables."
    prompt = f"""Formal Data Quality Assessment for this PDF.
Doc: {ds['words']:,} words | {ds['tables']} tables | missing kw: {ds['missing_kw']}
Per-table:
{per}
Score each dimension (0-100%) with justification:
1. Completeness  2. Consistency  3. Accuracy
4. Timeliness    5. Validity     6. Uniqueness
Then:
- Overall quality score (0-10) with reasoning
- Critical issues to fix before analysis
- 3 specific improvement actions with priority"""
    ai = ask_pdf(prompt)
    return ai, figs, ds, qr

def run_comparative():
    tables = st.session_state.tables
    figs = []
    for title, df in tables[:4]:
        nc = num_cols(df)
        cc = cat_cols(df)
        if not (cc and nc and len(df) >= 3):
            continue
        fig = make_bar(df, cc[0], nc[0], f"'{title}' — {nc[0]} by {cc[0]}", C[0])
        figs.append(fig)
        fig2 = make_pie(df, cc[0], nc[0], f"'{title}' — composition")
        if fig2:
            figs.append(fig2)
        if len(nc) >= 2:
            fig3 = make_scatter(df, nc[0], nc[1], cc[0] if cc else None, f"'{title}' — {nc[0]} vs {nc[1]}")
            figs.append(fig3)
    stats = "\n\n".join(f"'{t}':\n{df_stats(df)}" for t, df in tables[:4]) or "No tables."
    prompt = f"""Comparative analysis of this PDF's data.
Table statistics:
{stats}
Text: {st.session_state.full_text[:4000]}
Identify and compare:
1. Groups compared (categories, regions, time periods)
2. Performance gaps — who leads, who lags, exact %
3. Trends — growing, declining, stable
4. [A] vs [B]: metric = X vs Y (±Z%) format
5. Statistical significance
6. Single headline finding"""
    ai = ask_pdf(prompt)
    return ai, figs

c1, c2, c3 = st.columns([1,2,1])
with c2:
    uploaded_file = st.file_uploader("📎 Upload your PDF", type=["pdf"])

if uploaded_file and gemini_key:
    fname = uploaded_file.name
    if st.session_state.processed_file != fname:
        for k, v in _D.items():
            st.session_state[k] = v
        st.session_state.processed_file = fname
    if not st.session_state.full_text:
        with st.spinner("🔄 Extracting text from PDF…"):
            try:
                text, raw_bytes = process_pdf(uploaded_file)
                if not text.strip():
                    st.error("❌ No extractable text. Try a non-scanned PDF.")
                    st.stop()
                st.session_state.full_text = text
                st.session_state.pdf_bytes = raw_bytes
                st.success(f"✅ Text extracted — {len(text.split()):,} words")
            except Exception as e:
                st.error(f"❌ Text extraction error: {e}")
                st.stop()
    if st.session_state.full_text and not st.session_state.tables:
        with st.spinner("🤖 Gemini is reading tables from your PDF…"):
            try:
                tables = load_tables()
                st.session_state.tables = tables
                if tables:
                    st.success(f"✅ **{len(tables)} table(s)** extracted by Gemini")
                    st.balloons()
                    with st.expander("🔍 Preview extracted tables"):
                        for title, df in tables:
                            st.markdown(f"**{title}** — {df.shape[0]} rows × {df.shape[1]} cols")
                            st.dataframe(df, use_container_width=True)
                else:
                    st.warning("⚠️ No structured tables found — text-only analysis available.")
            except Exception as e:
                st.error(f"❌ Table extraction error: {e}")

    OPTIONS = [
        {"id":"eda_summary",  "icon":"📊","label":"Auto EDA summary",    "desc":"Real stats + correct charts", "badge":"AI","bc":"b-ai"},
        {"id":"extraction",   "icon":"📄","label":"Structured extraction","desc":"Tables → DataFrames + CSV",   "badge":"Extract","bc":"b-ext"},
        {"id":"report",       "icon":"📝","label":"Generate EDA report",  "desc":"Full 6-section report",       "badge":"Report","bc":"b-rep"},
        {"id":"charts",       "icon":"📈","label":"Smart charts",         "desc":"Best charts from your data",  "badge":"Viz","bc":"b-viz"},
        {"id":"anomaly",      "icon":"🔍","label":"Anomaly detection",    "desc":"Real z-score & IQR",          "badge":"Detect","bc":"b-ai"},
        {"id":"text_insights","icon":"💬","label":"Text & NLP",           "desc":"Keywords, sentiment",         "badge":"NLP","bc":"b-nlp"},
        {"id":"correlation",  "icon":"🔗","label":"Correlation mapping",  "desc":"Heatmap + scatter matrix",    "badge":"Relate","bc":"b-viz"},
        {"id":"qa_chat",      "icon":"🤖","label":"AI Q&A chat",          "desc":"Ask anything — Gemini answers","badge":"Q&A","bc":"b-qa"},
        {"id":"data_quality", "icon":"🩺","label":"Data quality",         "desc":"Completeness, dups, scores",  "badge":"Quality","bc":"b-ext"},
        {"id":"comparative",  "icon":"⚖️","label":"Comparative analysis", "desc":"A vs B charts",              "badge":"Compare","bc":"b-viz"},
    ]

    st.markdown("---")
    st.markdown("### 🚀 Choose your analysis")

    for row in [OPTIONS[:5], OPTIONS[5:]]:
        cols = st.columns(5)
        for col, opt in zip(cols, row):
            with col:
                active = st.session_state.active_option == opt["id"]
                bdr = "2px solid #4285F4" if active else "1px solid #E5E7EB"
                bg  = "#EFF6FF" if active else "#ffffff"
                st.markdown(f"""
<div style="background:{bg};border:{bdr};border-radius:14px;
     padding:12px;text-align:center;min-height:114px;">
  <div style="font-size:20px;margin-bottom:4px;">{opt['icon']}</div>
  <span class="badge {opt['bc']}">{opt['badge']}</span>
  <p style="font-size:11px;font-weight:600;color:#111827;margin:0 0 2px;">{opt['label']}</p>
  <p style="font-size:10px;color:#9CA3AF;margin:0;line-height:1.3;">{opt['desc']}</p>
</div>""", unsafe_allow_html=True)
                if st.button("▶ Run", key=f"btn_{opt['id']}", use_container_width=True):
                    st.session_state.active_option  = opt["id"]
                    st.session_state.option_result  = None
                    if opt["id"] != "qa_chat":
                        st.rerun()

    st.markdown("---")
    active  = st.session_state.active_option
    opt_map = {o["id"]: o for o in OPTIONS}

    if active == "qa_chat":
        st.markdown("### 🤖 AI Q&A — Ask anything about your PDF")
        st.caption(f"Powered by {model_choice} — reads your entire PDF natively")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="ql">You</div><div class="qb">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="al">Gemini</div><div class="ab">{msg["content"]}</div>', unsafe_allow_html=True)
        if q := st.chat_input("Ask anything about your PDF…"):
            st.session_state.messages.append({"role":"user","content":q})
            st.markdown(f'<div class="ql">You</div><div class="qb">{q}</div>', unsafe_allow_html=True)
            with st.spinner("Gemini is reading your PDF…"):
                ans = ask_question(q)
            st.session_state.messages.append({"role":"assistant","content":ans})
            st.markdown(f'<div class="al">Gemini</div><div class="ab">{ans}</div>', unsafe_allow_html=True)
        if st.session_state.messages:
            if st.button("🗑️ Clear chat"):
                st.session_state.messages = []
                st.rerun()

    elif active and st.session_state.option_result is None:
        opt = opt_map[active]
        st.markdown(f"### {opt['icon']} {opt['label']}")
        with st.spinner("🤖 Gemini is analysing your PDF…"):
            try:
                if   active == "eda_summary":   r = ("eda_summary",  *run_eda_summary())
                elif active == "extraction":     r = ("extraction",   *run_extraction())
                elif active == "report":         r = ("report",       *run_report())
                elif active == "charts":         r = ("charts",       *run_charts())
                elif active == "anomaly":        r = ("anomaly",      *run_anomaly())
                elif active == "text_insights":  r = ("text_insights",*run_text_insights())
                elif active == "correlation":    r = ("correlation",  *run_correlation())
                elif active == "data_quality":   r = ("data_quality", *run_data_quality())
                elif active == "comparative":    r = ("comparative",  *run_comparative())
                else:                            r = None
                st.session_state.option_result = r
                st.rerun()
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")
                st.exception(e)

    elif active and active != "qa_chat" and st.session_state.option_result:
        res  = st.session_state.option_result
        kind = res[0]
        opt  = opt_map[active]
        st.markdown(f"### {opt['icon']} {opt['label']}")

        if kind == "eda_summary":
            _, ai, figs = res
            if figs:
                st.markdown("#### 📊 Real charts from your PDF data")
                show_figs(figs)
            else:
                st.info("ℹ️ No chartable data found. Gemini's text analysis is below.")
            show_ai(ai)
        elif kind == "extraction":
            _, ai, figs, tbls = res
            st.markdown(ai)
            show_figs(figs)
            if tbls:
                st.markdown("#### 📋 Extracted tables")
                for title, df in tbls[:8]:
                    with st.expander(f"{title} — {df.shape[0]}r × {df.shape[1]}c"):
                        st.dataframe(df, use_container_width=True)
                        st.download_button(f"⬇️ Download '{title}' as CSV", data=df.to_csv(index=False),
                                           file_name=f"{title.replace(' ','_')}.csv", mime="text/csv", key=f"dl_{title[:10]}")
        elif kind == "report":
            _, ai, figs = res
            show_figs(figs)
            show_ai(ai)
            st.download_button("⬇️ Download report as Markdown", data=ai, file_name="eda_report.md", mime="text/markdown")
        elif kind == "charts":
            _, ai, figs = res
            if figs:
                st.markdown("#### 📈 Charts from your PDF data")
                show_figs(figs)
            else:
                st.info("ℹ️ No chartable columns found.")
            show_ai(ai)
        elif kind == "anomaly":
            _, ai, figs, flags = res
            if flags:
                st.markdown("#### 🚨 Anomalies detected")
                for sev, msg in flags:
                    css = {"HIGH":"fh","MED":"fm","LOW":"fl"}.get(sev,"fl")
                    st.markdown(f'<div class="{css}"><strong>[{sev}]</strong> {msg}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No statistical anomalies detected.")
            show_figs(figs)
            show_ai(ai)
        elif kind == "text_insights":
            _, ai, figs = res
            show_figs(figs)
            show_ai(ai)
        elif kind == "correlation":
            _, ai, figs, corr = res
            show_figs(figs)
            if corr is not None:
                with st.expander("📊 Raw correlation matrix"):
                    st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)
            show_ai(ai)
        elif kind == "data_quality":
            _, ai, figs, ds, qr = res
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total words",  f"{ds['words']:,}")
            m2.metric("Tables found", ds['tables'])
            m3.metric("Missing kw",   ds['missing_kw'])
            m4.metric("Total chars",  f"{ds['chars']:,}")
            if qr:
                st.markdown("#### 📋 Per-table quality")
                st.dataframe(pd.DataFrame(qr), use_container_width=True)
            show_figs(figs)
            show_ai(ai)
        elif kind == "comparative":
            _, ai, figs = res
            show_figs(figs)
            show_ai(ai)

        st.markdown("---")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔄 Re-run analysis", use_container_width=True):
                st.session_state.option_result = None
                st.rerun()
        with b2:
            if st.button("🤖 Follow-up in Q&A ↗", use_container_width=True):
                st.session_state.active_option = "qa_chat"
                st.rerun()

elif not gemini_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar.")
    st.info("👉 Free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)")
else:
    st.info("👆 Upload a PDF to unlock all 10 AI-powered analyses.")
ENDOFFILE
echo "PDF page created"