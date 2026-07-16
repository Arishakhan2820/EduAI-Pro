"""
EduAI Pro  —  Streamlit Application  (v4 — Three.js 3D Animated UI)
Run:  streamlit run app/main.py
"""

import sys, os, json, io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.engine import (load_artefacts, predict_one,
                        predict_batch, ENG_FEATURES, FEATURES)

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EduAI Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ══════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;1,14..32,300&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Background ── */
.stApp {
    background: #04080f;
}

/* ── Hide default chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Header bar stays visible (sidebar hamburger lives here) ── */
[data-testid="stHeader"] {
    background: rgba(4,8,15,0.95) !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(12px) !important;
    height: 3rem !important;
}

/* ── Block container ── */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* ══════════════════════════════════════
   SIDEBAR
══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #060c17 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    min-width: 252px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.8rem;
}

/* Sidebar radio nav */
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    margin: 2px 8px !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: #4b5a6e !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
    border: 1px solid transparent !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.04) !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] { gap: 0 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { margin: 0 !important; }

/* ══════════════════════════════════════
   KPI CARDS
══════════════════════════════════════ */
.kpi-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 22px 18px 18px;
    text-align: center;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(8px);
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(99,155,255,0.25);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,155,255,0.1);
}
.kpi-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,155,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 8px 0 6px;
    color: #e2e8f0;
}
.kpi-value-accent { color: #639bff; }
.kpi-label {
    font-size: 0.67rem;
    color: rgba(255,255,255,0.22);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
}
.kpi-sub {
    font-size: 0.72rem;
    color: rgba(99,155,255,0.5);
    margin-top: 4px;
    font-weight: 500;
}

/* ══════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════ */
.page-header {
    padding: 0 0 22px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 26px;
}
.page-tag {
    display: inline-block;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    color: #639bff;
    background: rgba(99,155,255,0.08);
    border: 1px solid rgba(99,155,255,0.2);
    border-radius: 4px;
    padding: 3px 9px;
    margin-bottom: 10px;
}
.page-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.03em;
    margin: 0 0 5px;
    line-height: 1.2;
}
.page-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.28);
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

/* ══════════════════════════════════════
   SECTION DIVIDERS
══════════════════════════════════════ */
.sec-label {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255,255,255,0.22);
    padding: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    background: #639bff;
    border-radius: 2px;
    flex-shrink: 0;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ══════════════════════════════════════
   OUTCOME BADGES
══════════════════════════════════════ */
.badge {
    display: inline-block;
    padding: 7px 22px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-pass   { background: rgba(22,163,74,0.12);  border: 1px solid rgba(22,163,74,0.35);  color: #4ade80; }
.badge-atrisk { background: rgba(217,119,6,0.12);  border: 1px solid rgba(217,119,6,0.35);  color: #fbbf24; }
.badge-fail   { background: rgba(220,38,38,0.12);  border: 1px solid rgba(220,38,38,0.35);  color: #f87171; }

/* ══════════════════════════════════════
   RESULT CARD
══════════════════════════════════════ */
.result-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 30px 24px;
    text-align: center;
    backdrop-filter: blur(8px);
}
.result-label {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(255,255,255,0.22);
    font-weight: 700;
    margin-bottom: 14px;
}
.confidence-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: rgba(255,255,255,0.22);
    margin-top: 14px;
    letter-spacing: 0.08em;
}

/* ══════════════════════════════════════
   RECOMMENDATION CARDS
══════════════════════════════════════ */
.rec-card {
    border-radius: 10px;
    padding: 13px 16px;
    margin: 7px 0;
    border-left: 2px solid;
    transition: transform 0.15s ease;
}
.rec-card:hover { transform: translateX(4px); }
.rec-danger  { background: rgba(220,38,38,0.06);  border-color: rgba(220,38,38,0.5);  }
.rec-warning { background: rgba(217,119,6,0.06);  border-color: rgba(217,119,6,0.5);  }
.rec-info    { background: rgba(99,155,255,0.06); border-color: rgba(99,155,255,0.4); }
.rec-success { background: rgba(22,163,74,0.06);  border-color: rgba(22,163,74,0.4);  }
.rec-title   { font-weight: 600; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 4px; }
.rec-body    { font-size: 0.78rem; color: rgba(255,255,255,0.3); line-height: 1.6; }

/* ══════════════════════════════════════
   HISTORY ROWS
══════════════════════════════════════ */
.hist-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 14px;
    border-radius: 8px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    margin: 5px 0;
    font-size: 0.8rem;
}

/* ══════════════════════════════════════
   INPUT OVERRIDES
══════════════════════════════════════ */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}
[data-baseweb="input"] input,
[data-baseweb="select"] div[role="combobox"] {
    color: #cbd5e1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: rgba(99,155,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,155,255,0.1) !important;
}
label[data-testid="stWidgetLabel"] p { color: rgba(255,255,255,0.5) !important; font-size: 0.82rem !important; }

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #639bff !important;
    border-color: #639bff !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.18s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(99,155,255,0.3) !important;
    color: #e2e8f0 !important;
}
.stFormSubmitButton > button[kind="primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #2d6ef5, #1a4ed8) !important;
    border: 1px solid rgba(99,155,255,0.3) !important;
    color: #fff !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 20px rgba(45,110,245,0.25) !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3b7eff, #2563eb) !important;
    box-shadow: 0 6px 28px rgba(45,110,245,0.4) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stMetricLabel"] p { color: rgba(255,255,255,0.3) !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: .12em; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricDelta"] { display: none !important; }

/* Form */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.015) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    backdrop-filter: blur(8px) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: rgba(255,255,255,0.35) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(255,255,255,0.03) !important;
    color: rgba(255,255,255,0.3) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.05) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* Validation error */
.val-err {
    background: rgba(220,38,38,0.07);
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    color: #f87171;
    font-size: 0.8rem;
    margin: 6px 0;
    font-weight: 500;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load():
    try:
        return load_artefacts("saved_model"), True
    except Exception:
        return None, False

artefacts, ok = _load()
if ok:
    model, scaler, le, metrics = artefacts
else:
    model = scaler = le = metrics = None


# ══════════════════════════════════════════════════════════
# PLOTLY BASE LAYOUT
# ══════════════════════════════════════════════════════════
def plot_layout(h=380, **kw):
    base = dict(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(8,14,26,0.95)",
        font          = dict(family="Inter", color="rgba(255,255,255,0.35)", size=11),
        title_font    = dict(size=12, color="rgba(255,255,255,0.45)", family="Inter"),
        margin        = dict(t=44, b=20, l=10, r=10),
        height        = h,
        xaxis         = dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                             linecolor="rgba(255,255,255,0.06)",
                             tickcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(size=10)),
        yaxis         = dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                             linecolor="rgba(255,255,255,0.06)",
                             tickcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(size=10)),
    )
    base.update(kw)
    return base


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 14px 18px'>
      <div style='display:flex;align-items:center;gap:11px'>
        <div style='width:36px;height:36px;
              background:linear-gradient(135deg,#1a3a8f,#2d6ef5);
              border-radius:10px;display:flex;align-items:center;
              justify-content:center;font-size:1.1rem;
              box-shadow:0 4px 16px rgba(45,110,245,0.3);flex-shrink:0'>🎓</div>
        <div>
          <div style='font-size:0.95rem;font-weight:700;color:#e2e8f0;
                letter-spacing:-0.01em'>EduAI Pro</div>
          <div style='font-size:0.6rem;color:rgba(255,255,255,0.25);
                text-transform:uppercase;letter-spacing:.12em;font-weight:600;
                margin-top:1px'>Academic Intelligence</div>
        </div>
      </div>
    </div>
    <div style='height:1px;background:rgba(255,255,255,0.05);margin:0 14px 14px'></div>
    <div style='font-size:.6rem;text-transform:uppercase;letter-spacing:.14em;
          color:rgba(255,255,255,0.2);font-weight:700;padding:0 16px 8px'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        options=[
            "Overview",
            "Predict Student",
            "Batch Analysis",
            "Analytics Dashboard",
            "Model Insights",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.05);margin:12px 14px'></div>",
                unsafe_allow_html=True)

    if ok:
        st.markdown(f"""
        <div style='margin:0 10px;padding:14px;
              background:rgba(255,255,255,0.02);
              border:1px solid rgba(255,255,255,0.05);
              border-radius:12px'>
          <div style='font-size:.6rem;text-transform:uppercase;
                letter-spacing:.14em;color:rgba(255,255,255,0.2);
                font-weight:700;margin-bottom:12px'>Model Status</div>
          <div style='font-size:.72rem;color:rgba(255,255,255,0.3);line-height:2.1'>
            <div style='display:flex;justify-content:space-between'>
              <span>Status</span>
              <span style='color:#4ade80;font-weight:600'>● Online</span>
            </div>
            <div style='display:flex;justify-content:space-between'>
              <span>ROC-AUC</span>
              <span style='color:rgba(255,255,255,0.55);font-family:JetBrains Mono,monospace'>
                {metrics['roc_auc']*100:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between'>
              <span>CV Acc.</span>
              <span style='color:rgba(255,255,255,0.55);font-family:JetBrains Mono,monospace'>
                {metrics['cv_mean']*100:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between'>
              <span>F1 Macro</span>
              <span style='color:rgba(255,255,255,0.55);font-family:JetBrains Mono,monospace'>
                {metrics['f1_macro']*100:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between'>
              <span>Train Set</span>
              <span style='color:rgba(255,255,255,0.55);font-family:JetBrains Mono,monospace'>
                {metrics['n_train']:,}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        n_pred = len(st.session_state.prediction_history)
        if n_pred > 0:
            st.markdown(f"""
            <div style='margin:8px 10px 0;padding:10px 14px;
                  background:rgba(99,155,255,0.06);
                  border:1px solid rgba(99,155,255,0.15);
                  border-radius:10px;
                  display:flex;justify-content:space-between;align-items:center'>
              <span style='font-size:.72rem;color:rgba(255,255,255,0.3)'>Session predictions</span>
              <span style='font-family:JetBrains Mono,monospace;font-size:.9rem;
                    font-weight:700;color:#639bff'>{n_pred}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='margin:0 10px;padding:12px;background:rgba(220,38,38,0.08);
              border:1px solid rgba(220,38,38,0.25);border-radius:10px;
              font-size:.78rem;color:#f87171;text-align:center'>
          Model not loaded<br>
          <span style='font-size:.68rem;color:rgba(255,255,255,0.25)'>Run: python train.py</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 14px;font-size:.58rem;color:rgba(255,255,255,0.12);
          text-align:center;text-transform:uppercase;letter-spacing:.12em;
          font-family:JetBrains Mono,monospace'>
      RF · GBM · SVC · LR Ensemble
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def page_header(label, title, sub=""):
    st.markdown(f"""
    <div class='page-header'>
      <span class='page-tag'>{label}</span>
      <div class='page-title'>{title}</div>
      {'<div class="page-sub">' + sub + '</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)

def section(text):
    st.markdown(f"<div class='sec-label'>{text}</div>", unsafe_allow_html=True)

def val_error(msg):
    st.markdown(f"<div class='val-err'>{msg}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# THREE.JS HERO  (injected on Overview page)
# ══════════════════════════════════════════════════════════
def render_threejs_hero(roc_auc, f1, cv_acc, n_train):
    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#04080f; overflow:hidden; font-family:'Inter',system-ui,sans-serif; }}
  canvas {{ display:block; position:absolute; top:0; left:0; }}

  .overlay {{
    position:absolute; inset:0;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    pointer-events:none; z-index:10;
  }}

  .logo-ring {{
    width:72px; height:72px; border-radius:50%;
    background:radial-gradient(circle at 40% 35%, #2d6ef5, #0f2566);
    border:1px solid rgba(99,155,255,0.35);
    display:flex; align-items:center; justify-content:center;
    font-size:1.8rem;
    box-shadow:0 0 40px rgba(45,110,245,0.45),
               0 0 80px rgba(45,110,245,0.2),
               inset 0 1px 0 rgba(255,255,255,0.15);
    margin-bottom:20px;
    animation: float 4s ease-in-out infinite;
  }}
  @keyframes float {{
    0%,100% {{ transform:translateY(0); }}
    50%       {{ transform:translateY(-8px); }}
  }}

  .hero-badge {{
    font-size:0.58rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.18em; color:rgba(99,155,255,0.7);
    background:rgba(99,155,255,0.07);
    border:1px solid rgba(99,155,255,0.2);
    border-radius:50px; padding:5px 14px; margin-bottom:16px;
  }}

  .hero-title {{
    font-size:2.4rem; font-weight:800; color:#e2e8f0;
    letter-spacing:-0.04em; text-align:center;
    line-height:1.15; margin-bottom:10px;
  }}
  .hero-title span {{ color:#639bff; }}

  .hero-sub {{
    font-size:0.9rem; color:rgba(255,255,255,0.35);
    text-align:center; max-width:480px; line-height:1.7;
    margin-bottom:36px; font-weight:400;
  }}

  .stats-row {{
    display:flex; gap:1px; align-items:stretch;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; overflow:hidden;
    backdrop-filter:blur(16px);
  }}
  .stat {{
    padding:18px 28px; text-align:center;
    background:rgba(4,8,15,0.6);
    position:relative;
  }}
  .stat:not(:last-child)::after {{
    content:''; position:absolute; right:0; top:20%; bottom:20%;
    width:1px; background:rgba(255,255,255,0.07);
  }}
  .stat-val {{
    font-family:'JetBrains Mono',monospace;
    font-size:1.5rem; font-weight:700; color:#e2e8f0;
    letter-spacing:-0.03em; margin-bottom:4px;
  }}
  .stat-lbl {{
    font-size:0.6rem; text-transform:uppercase;
    letter-spacing:0.14em; color:rgba(255,255,255,0.25); font-weight:600;
  }}
  .stat-val-blue {{ color:#639bff; }}
  .stat-val-grn  {{ color:#4ade80; }}
  .stat-val-amb  {{ color:#fbbf24; }}

  .scroll-hint {{
    position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
    font-size:0.62rem; letter-spacing:0.14em; text-transform:uppercase;
    color:rgba(255,255,255,0.18); animation:pulse 2.5s ease-in-out infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:.18}} 50%{{opacity:.5}} }}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div class="overlay">
  <div class="logo-ring">🎓</div>
  <div class="hero-badge">Ensemble ML · 4-Learner Soft-Voting</div>
  <div class="hero-title">Academic<br><span>Intelligence</span> Platform</div>
  <div class="hero-sub">Predict student outcomes with precision — Pass, At-Risk, or Fail —<br>powered by a trained Random Forest &amp; Gradient Boosting ensemble.</div>
  <div class="stats-row">
    <div class="stat">
      <div class="stat-val stat-val-blue">{roc_auc:.1f}%</div>
      <div class="stat-lbl">ROC-AUC</div>
    </div>
    <div class="stat">
      <div class="stat-val stat-val-grn">{f1:.1f}%</div>
      <div class="stat-lbl">F1 Macro</div>
    </div>
    <div class="stat">
      <div class="stat-val stat-val-amb">{cv_acc:.1f}%</div>
      <div class="stat-lbl">CV Accuracy</div>
    </div>
    <div class="stat">
      <div class="stat-val">{n_train:,}</div>
      <div class="stat-lbl">Trained On</div>
    </div>
  </div>
</div>

<div class="scroll-hint">↓ &nbsp; scroll to explore &nbsp; ↓</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script>
(function() {{
  const W = window.innerWidth, H = window.innerHeight;
  const renderer = new THREE.WebGLRenderer({{ canvas: document.getElementById('c'), antialias:true, alpha:true }});
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x04080f, 1);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, W/H, 0.1, 1000);
  camera.position.set(0, 0, 28);

  /* ── Fog ── */
  scene.fog = new THREE.FogExp2(0x04080f, 0.018);

  /* ── Node particles ── */
  const N_NODES = 120;
  const positions = [];
  const nodeData  = [];

  for (let i = 0; i < N_NODES; i++) {{
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = 8 + Math.random() * 10;
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta) * 0.55;
    const z = r * Math.cos(phi);
    positions.push(x, y, z);
    nodeData.push({{ x, y, z,
      vx: (Math.random() - 0.5) * 0.004,
      vy: (Math.random() - 0.5) * 0.004,
      vz: (Math.random() - 0.5) * 0.004,
      phase: Math.random() * Math.PI * 2
    }});
  }}

  /* ── Point geometry ── */
  const ptGeo = new THREE.BufferGeometry();
  const ptPos = new Float32Array(positions);
  ptGeo.setAttribute('position', new THREE.BufferAttribute(ptPos, 3));

  const ptMat = new THREE.PointsMaterial({{
    color: 0x639bff,
    size: 0.22,
    transparent: true,
    opacity: 0.85,
    sizeAttenuation: true
  }});
  const points = new THREE.Points(ptGeo, ptMat);
  scene.add(points);

  /* ── Edge lines ── */
  const EDGE_DIST = 6.5;
  const linePosArr = [];
  const edgePairs  = [];

  for (let i = 0; i < N_NODES; i++) {{
    for (let j = i+1; j < N_NODES; j++) {{
      const dx = nodeData[i].x - nodeData[j].x;
      const dy = nodeData[i].y - nodeData[j].y;
      const dz = nodeData[i].z - nodeData[j].z;
      const d  = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (d < EDGE_DIST) {{
        edgePairs.push([i, j, d]);
        linePosArr.push(
          nodeData[i].x, nodeData[i].y, nodeData[i].z,
          nodeData[j].x, nodeData[j].y, nodeData[j].z
        );
      }}
    }}
  }}

  const lineGeo = new THREE.BufferGeometry();
  const lineBuf = new Float32Array(linePosArr);
  lineGeo.setAttribute('position', new THREE.BufferAttribute(lineBuf, 3));
  const lineMat = new THREE.LineBasicMaterial({{
    color: 0x1a3a8f,
    transparent: true,
    opacity: 0.4
  }});
  const lines = new THREE.LineSegments(lineGeo, lineMat);
  scene.add(lines);

  /* ── Central glowing orb ── */
  const orbGeo = new THREE.SphereGeometry(1.6, 32, 32);
  const orbMat = new THREE.MeshBasicMaterial({{
    color: 0x1a3a8f,
    transparent: true,
    opacity: 0.35,
    wireframe: false
  }});
  const orb = new THREE.Mesh(orbGeo, orbMat);
  scene.add(orb);

  /* Wireframe ring around orb */
  const wireGeo = new THREE.SphereGeometry(1.7, 16, 16);
  const wireMat = new THREE.MeshBasicMaterial({{ color:0x2d6ef5, wireframe:true, transparent:true, opacity:0.12 }});
  const wire = new THREE.Mesh(wireGeo, wireMat);
  scene.add(wire);

  /* Outer ring plane */
  const ringGeo = new THREE.TorusGeometry(3.2, 0.06, 8, 60);
  const ringMat = new THREE.MeshBasicMaterial({{ color:0x2d6ef5, transparent:true, opacity:0.25 }});
  const ring1 = new THREE.Mesh(ringGeo, ringMat);
  ring1.rotation.x = Math.PI / 2.5;
  scene.add(ring1);

  const ring2Geo = new THREE.TorusGeometry(4.8, 0.04, 8, 80);
  const ring2Mat = new THREE.MeshBasicMaterial({{ color:0x1a3a8f, transparent:true, opacity:0.18 }});
  const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
  ring2.rotation.x = Math.PI / 3;
  ring2.rotation.z = Math.PI / 6;
  scene.add(ring2);

  /* ── Ambient floating particles ── */
  const dustGeo = new THREE.BufferGeometry();
  const dustN = 350;
  const dustPos = new Float32Array(dustN * 3);
  for (let i = 0; i < dustN * 3; i++) dustPos[i] = (Math.random() - 0.5) * 50;
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  const dustMat = new THREE.PointsMaterial({{ color:0x243a6e, size:0.1, transparent:true, opacity:0.45 }});
  const dust = new THREE.Points(dustGeo, dustMat);
  scene.add(dust);

  /* ── Animation loop ── */
  let t = 0;
  let mx = 0, my = 0;
  document.addEventListener('mousemove', e => {{
    mx = (e.clientX / W - 0.5) * 2;
    my = (e.clientY / H - 0.5) * 2;
  }});

  function animate() {{
    requestAnimationFrame(animate);
    t += 0.008;

    /* gentle camera sway */
    camera.position.x += (mx * 2 - camera.position.x) * 0.02;
    camera.position.y += (-my * 1.2 - camera.position.y) * 0.02;
    camera.lookAt(0, 0, 0);

    /* rotate main node cloud */
    points.rotation.y = t * 0.06;
    points.rotation.x = Math.sin(t * 0.04) * 0.08;

    /* rotate lines with nodes */
    lines.rotation.y = t * 0.06;
    lines.rotation.x = Math.sin(t * 0.04) * 0.08;

    /* orb pulse */
    const pulse = 1 + 0.06 * Math.sin(t * 1.2);
    orb.scale.setScalar(pulse);
    wire.scale.setScalar(pulse * 1.02);
    orbMat.opacity = 0.28 + 0.12 * Math.sin(t * 1.2);

    /* rings */
    ring1.rotation.z = t * 0.22;
    ring2.rotation.z = -t * 0.14;
    ring2.rotation.y = t * 0.08;

    /* dust drift */
    dust.rotation.y = t * 0.012;
    dust.rotation.x = t * 0.007;

    renderer.render(scene, camera);
  }}
  animate();
}})();
</script>
</body>
</html>
"""
    components.html(html_code, height=480, scrolling=False)


# ══════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":

    if not ok:
        st.error("Model not loaded — run `python train.py` first.")
        st.stop()

    k = metrics

    # Three.js hero
    render_threejs_hero(
        roc_auc  = k['roc_auc']  * 100,
        f1       = k['f1_macro'] * 100,
        cv_acc   = k['cv_mean']  * 100,
        n_train  = k['n_train'],
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # KPI cards
    section("Key Performance Metrics")
    kpi_data = [
        ("ROC-AUC",       f"{k['roc_auc']*100:.1f}%",   "Ensemble OvR"),
        ("F1 Macro",      f"{k['f1_macro']*100:.1f}%",   "3-class avg"),
        ("CV Accuracy",   f"{k['cv_mean']*100:.1f}%",    "5-fold stratified"),
        ("Training Size", f"{k['n_train']:,}",            "samples"),
        ("Features",      str(k['n_features']),           "engineered inputs"),
        ("Learners",      "4",                            "in ensemble"),
    ]
    cols = st.columns(6, gap="small")
    for col, (label, val, sub) in zip(cols, kpi_data):
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value kpi-value-accent'>{val}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Charts
    fi = k["feature_importances"]
    fi_df = (pd.DataFrame.from_dict(fi, orient="index", columns=["importance"])
               .reset_index().rename(columns={"index": "feature"})
               .sort_values("importance", ascending=True).tail(15))

    fig_fi = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale=["rgba(13,28,64,1)","rgba(29,78,216,1)","rgba(99,155,255,1)"],
        title="Feature Importances — Random Forest"
    )
    fig_fi.update_layout(**plot_layout(420), coloraxis_showscale=False)
    fig_fi.update_traces(marker_line_width=0)

    classes = k["classes"]
    cm_arr  = np.array(k["confusion_matrix"])
    totals  = cm_arr.sum(axis=1)

    fig_pie = go.Figure(go.Pie(
        labels=classes, values=totals, hole=0.6,
        marker=dict(
            colors=["rgba(120,53,15,0.85)","rgba(127,29,29,0.85)","rgba(5,46,22,0.85)"],
            line=dict(color="#04080f", width=3)
        ),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        textinfo="percent+label",
        textfont=dict(color="rgba(255,255,255,0.5)", size=11),
    ))
    fig_pie.update_layout(
        **plot_layout(420,
            title="Test Set — Class Distribution",
            showlegend=False,
        )
    )

    c1, c2 = st.columns([3, 2], gap="medium")
    c1.plotly_chart(fig_fi,  use_container_width=True)
    c2.plotly_chart(fig_pie, use_container_width=True)

    section("Confusion Matrix")
    fig_cm = px.imshow(
        cm_arr, x=classes, y=classes,
        labels=dict(x="Predicted", y="Actual"),
        text_auto=True,
        color_continuous_scale=["#04080f","#0c1f3d","#1e40af","#639bff"],
        title="Test-Set Confusion Matrix"
    )
    fig_cm.update_layout(**plot_layout(300))
    st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 1 — PREDICT STUDENT
# ══════════════════════════════════════════════════════════
elif page == "Predict Student":
    page_header(
        "Prediction",
        "Student Outcome Predictor",
        "Enter the student profile — the model returns outcome probability, estimated GPA and personalised interventions."
    )

    if not ok:
        st.error("Run `python train.py` first.")
        st.stop()

    with st.form("predict_form", clear_on_submit=False):

        section("Academic Performance")
        c1, c2, c3 = st.columns(3, gap="medium")
        study    = c1.number_input("Study Hours / Day",   0.0, 16.0,  5.0, 0.5,
                                   help="0 – 16 hours")
        attend   = c2.number_input("Attendance %",        0.0,100.0, 80.0, 1.0,
                                   help="0 – 100%")
        prev_gpa = c3.number_input("Previous GPA",        0.0,  4.0,  2.8, 0.1,
                                   help="0.0 – 4.0")

        c4, c5, c6 = st.columns(3, gap="medium")
        assign   = c4.number_input("Assignment Score %",  0.0,100.0, 75.0, 1.0)
        quiz     = c5.number_input("Quiz Average %",      0.0,100.0, 70.0, 1.0)
        lab      = c6.number_input("Lab Score %",         0.0,100.0, 75.0, 1.0)

        section("Lifestyle & Wellbeing")
        c7, c8, c9 = st.columns(3, gap="medium")
        sleep    = c7.number_input("Sleep Hours / Night", 3.0, 12.0,  7.0, 0.5,
                                   help="3 – 12 hours")
        stress   = c8.slider("Stress Level (1–10)",   1, 10, 4,
                             help="1 = very low  ·  10 = extremely high")
        health   = c9.slider("Health Score (1–10)",   1, 10, 7,
                             help="1 = poor  ·  10 = excellent")

        c10, c11, c12 = st.columns(3, gap="medium")
        internet = c10.number_input("Recreational Internet Hrs/Day",
                                    0.0, 16.0, 3.0, 0.5)
        commute  = c11.number_input("Commute Hours / Day",
                                    0.0,  6.0, 1.0, 0.5)
        particip = c12.slider("Class Participation (1–10)", 1, 10, 6)

        section("Background")
        c13, c14, c15 = st.columns(3, gap="medium")
        job      = c13.selectbox("Part-Time Job?",     ["No", "Yes"])
        extra    = c14.selectbox("Extracurricular?",   ["No", "Yes"])
        fam      = c15.slider("Family Support (1–10)", 1, 10, 7)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Run Prediction",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        errors = []
        if not (0 <= study <= 16):
            errors.append("Study hours must be between 0 and 16.")
        if not (0 <= attend <= 100):
            errors.append("Attendance must be between 0% and 100%.")
        if not (0 <= prev_gpa <= 4):
            errors.append("Previous GPA must be between 0.0 and 4.0.")
        if not (0 <= assign <= 100):
            errors.append("Assignment score must be between 0% and 100%.")
        if not (0 <= quiz <= 100):
            errors.append("Quiz average must be between 0% and 100%.")
        if not (0 <= lab <= 100):
            errors.append("Lab score must be between 0% and 100%.")
        if not (3 <= sleep <= 12):
            errors.append("Sleep hours must be between 3 and 12.")
        if not (0 <= internet <= 16):
            errors.append("Internet hours must be between 0 and 16.")
        if not (0 <= commute <= 6):
            errors.append("Commute hours must be between 0 and 6.")

        if errors:
            for e in errors:
                val_error(e)
            st.stop()

        raw = {
            "study_hours_per_day": study,
            "sleep_hours":         sleep,
            "attendance_pct":      attend,
            "prev_gpa":            prev_gpa,
            "assignment_score":    assign,
            "quiz_avg":            quiz,
            "lab_score":           lab,
            "participation_score": float(particip),
            "stress_level":        float(stress),
            "internet_hours":      internet,
            "part_time_job":       1 if job == "Yes" else 0,
            "family_support":      float(fam),
            "commute_hours":       commute,
            "health_score":        float(health),
            "extracurricular":     1 if extra == "Yes" else 0,
        }

        with st.spinner("Running ensemble inference …"):
            res = predict_one(raw, model, scaler, le)

        st.session_state.prediction_history.append({
            "outcome":    res["outcome"],
            "risk_score": res["risk_score"],
            "est_gpa":    res["est_gpa"],
            "confidence": res["confidence"],
        })

        st.markdown("---")
        section("Result")

        oc = res["outcome"]
        badge_cls = {"Pass":"badge-pass","At-Risk":"badge-atrisk","Fail":"badge-fail"}[oc]

        col_badge, col_kpi = st.columns([1, 3], gap="medium")

        with col_badge:
            st.markdown(f"""
            <div class='result-card'>
              <div class='result-label'>Outcome</div>
              <div class='{badge_cls} badge' style='margin-bottom:12px'>{oc}</div>
              <div class='confidence-mono'>CONFIDENCE&nbsp;&nbsp;{res['confidence']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        with col_kpi:
            k1, k2 = st.columns(2, gap="small")
            k3, k4 = st.columns(2, gap="small")
            k1.metric("Risk Score",  f"{res['risk_score']:.1f} / 100")
            k2.metric("Est. Score",  f"{res['est_score']:.1f}%")
            k3.metric("Est. Grade",  res['est_grade'])
            k4.metric("Est. GPA",    str(res['est_gpa']))

        probs  = res["probabilities"]
        p_vals = [probs.get(c, 0) for c in le.classes_]

        fig_prob = go.Figure(go.Bar(
            x=list(le.classes_), y=p_vals,
            marker_color=["rgba(120,53,15,0.85)","rgba(127,29,29,0.85)","rgba(5,46,22,0.85)"],
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in p_vals],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.5)", size=12),
        ))
        fig_prob.update_layout(
            **plot_layout(260,
                title="Outcome Probability Distribution",
                yaxis=dict(range=[0,120], gridcolor="rgba(255,255,255,0.04)"),
                showlegend=False
            )
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        risk = res["risk_score"]
        gauge_color = "#4ade80" if risk < 30 else "#fbbf24" if risk < 65 else "#f87171"
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk,
            title = {"text": "Risk Score", "font": {"color":"rgba(255,255,255,0.35)","size":12}},
            gauge = {
                "axis": {"range":[0,100],
                         "tickcolor":"rgba(255,255,255,0.1)",
                         "tickfont":{"color":"rgba(255,255,255,0.25)","size":10}},
                "bar":  {"color": gauge_color, "thickness": 0.2},
                "bgcolor": "rgba(8,14,26,0.6)",
                "bordercolor": "rgba(255,255,255,0.06)",
                "steps": [
                    {"range":[0,30],  "color":"rgba(5,46,22,0.6)"},
                    {"range":[30,65], "color":"rgba(67,20,7,0.6)"},
                    {"range":[65,100],"color":"rgba(69,10,10,0.6)"},
                ],
                "threshold": {"line":{"color":"rgba(255,255,255,0.6)","width":2},"value":risk},
            },
            number={"font":{"color":"#e2e8f0","size":28,"family":"JetBrains Mono"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(8,14,26,0.95)", height=200,
            margin=dict(t=30, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        section("AI Recommendations")
        for r in res["recommendations"]:
            st.markdown(f"""
            <div class='rec-card rec-{r["level"]}'>
              <div class='rec-title'>{r["icon"]}&nbsp; {r["title"]}</div>
              <div class='rec-body'>{r["body"]}</div>
            </div>""", unsafe_allow_html=True)

        if len(st.session_state.prediction_history) > 1:
            section("Session History")
            for h in reversed(st.session_state.prediction_history[-5:]):
                oc_col = {"Pass":"#4ade80","At-Risk":"#fbbf24","Fail":"#f87171"}.get(h["outcome"],"#94a3b8")
                st.markdown(f"""
                <div class='hist-row'>
                  <span style='color:{oc_col};font-weight:600;font-size:.82rem;min-width:72px'>{h["outcome"]}</span>
                  <span style='color:rgba(255,255,255,0.25);font-family:JetBrains Mono,monospace;font-size:.74rem'>
                    Risk {h["risk_score"]:.0f} &nbsp;|&nbsp;
                    GPA {h["est_gpa"]} &nbsp;|&nbsp;
                    Conf {h["confidence"]:.0f}%
                  </span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "Batch Analysis":
    page_header(
        "Batch",
        "Class-Wide Batch Analysis",
        "Upload a student CSV — the model predicts outcomes for every row and generates a downloadable report."
    )

    if not ok:
        st.error("Run `python train.py` first.")
        st.stop()

    sample = pd.DataFrame([{
        "name":"Ali Khan","roll_no":"STU-0001",
        "study_hours_per_day":6,"sleep_hours":7,
        "attendance_pct":82,"prev_gpa":3.1,
        "assignment_score":78,"quiz_avg":72,
        "lab_score":80,"participation_score":7,
        "stress_level":4,"internet_hours":3,
        "part_time_job":0,"family_support":8,
        "commute_hours":1,"health_score":8,
        "extracurricular":1,
    }])

    col_dl, col_up = st.columns([1, 2], gap="large")
    with col_dl:
        section("Step 1 — Template")
        st.download_button(
            "Download CSV Template",
            data=sample.to_csv(index=False),
            file_name="eduai_template.csv",
            mime="text/csv",
            use_container_width=True)
        st.markdown("""
        <div style='font-size:.72rem;color:rgba(255,255,255,0.25);margin-top:8px;line-height:1.8'>
          Fill one row per student.<br>All 15 feature columns are required.
        </div>""", unsafe_allow_html=True)

    with col_up:
        section("Step 2 — Upload")
        uploaded = st.file_uploader("Student CSV", type="csv",
                                    label_visibility="collapsed")

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
        except Exception as e:
            val_error(f"Cannot parse CSV: {e}")
            st.stop()

        missing = [c for c in FEATURES if c not in df_up.columns]
        if missing:
            val_error(f"Missing columns: {', '.join(missing)}")
            st.stop()
        if len(df_up) == 0:
            val_error("The uploaded file is empty.")
            st.stop()
        if len(df_up) > 10000:
            st.warning(f"Large batch: {len(df_up):,} rows — may take a moment.")

        st.info(f"Loaded **{len(df_up):,}** students from {uploaded.name}")

        section("Step 3 — Run")
        if st.button("Run Batch Prediction", type="primary",
                     use_container_width=True):
            with st.spinner("Running predictions …"):
                out = predict_batch(df_up, model, scaler, le)

            total  = len(out)
            n_pass = (out.outcome == "Pass").sum()
            n_risk = (out.outcome == "At-Risk").sum()
            n_fail = (out.outcome == "Fail").sum()

            st.markdown("---")
            section("Summary")

            c1, c2, c3, c4 = st.columns(4, gap="small")
            for col, (label, val, sub, ac) in zip(
                [c1,c2,c3,c4],
                [
                    ("Total",   f"{total:,}",  "students",                    "#639bff"),
                    ("Passing", f"{n_pass:,}", f"{n_pass/total*100:.0f}%",    "#4ade80"),
                    ("At-Risk", f"{n_risk:,}", f"{n_risk/total*100:.0f}%",    "#fbbf24"),
                    ("Failing", f"{n_fail:,}", f"{n_fail/total*100:.0f}%",    "#f87171"),
                ]):
                col.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{label}</div>
                  <div class='kpi-value' style='color:{ac}'>{val}</div>
                  <div class='kpi-sub'>{sub}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            fig_donut = go.Figure(go.Pie(
                labels=["Pass","At-Risk","Fail"],
                values=[n_pass, n_risk, n_fail],
                hole=0.6,
                marker=dict(
                    colors=["rgba(5,46,22,0.85)","rgba(120,53,15,0.85)","rgba(127,29,29,0.85)"],
                    line=dict(color="#04080f", width=3)
                ),
                textinfo="percent+label",
                textfont=dict(color="rgba(255,255,255,0.5)", size=11),
            ))
            fig_donut.update_layout(
                **plot_layout(280, title="Outcome Distribution", showlegend=False)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            section("Results Table")
            disp_cols = (
                ["name","roll_no","outcome","prob_pass","prob_at_risk","prob_fail","risk_score"]
                if "name" in out.columns
                else ["outcome","prob_pass","prob_at_risk","prob_fail","risk_score"]
            )
            st.dataframe(
                out[disp_cols].style.background_gradient(
                    subset=["risk_score"], cmap="RdYlGn_r"),
                use_container_width=True, height=380, hide_index=True
            )

            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download Enriched Results",
                data=buf.getvalue(),
                file_name="eduai_predictions.csv",
                mime="text/csv",
                use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════
elif page == "Analytics Dashboard":
    page_header(
        "Analytics",
        "Analytics Dashboard",
        "Filterable EDA — department, city, semester"
    )

    try:
        df = pd.read_csv("data/students.csv")
    except FileNotFoundError:
        st.error("Run `python train.py` to generate the dataset first.")
        st.stop()

    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3, gap="medium")
        sel_dept = fc1.multiselect("Department", sorted(df.department.unique()),
                                   default=list(df.department.unique()))
        sel_city = fc2.multiselect("City",       sorted(df.city.unique()),
                                   default=list(df.city.unique()))
        sel_sem  = fc3.multiselect("Semester",   sorted(df.semester.unique()),
                                   default=list(df.semester.unique()))

    mask = (df.department.isin(sel_dept) &
            df.city.isin(sel_city) &
            df.semester.isin(sel_sem))
    d = df[mask]

    if d.empty:
        val_error("No data matches the selected filters. Try broadening your selection.")
        st.stop()

    section("Filtered Metrics")
    kpis = [
        ("Students",    f"{len(d):,}",                             "filtered",  "#639bff"),
        ("Pass Rate",   f"{(d.outcome=='Pass').mean()*100:.1f}%",  "",          "#4ade80"),
        ("At-Risk",     f"{(d.outcome=='At-Risk').mean()*100:.1f}%","",         "#fbbf24"),
        ("Fail Rate",   f"{(d.outcome=='Fail').mean()*100:.1f}%",  "",          "#f87171"),
        ("Avg GPA",     f"{d.final_gpa.mean():.2f}",               "",          "#639bff"),
        ("Avg Attend.", f"{d.attendance_pct.mean():.1f}%",         "",          "#639bff"),
    ]
    cols = st.columns(6, gap="small")
    for col, (label, val, sub, ac) in zip(cols, kpis):
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value' style='color:{ac}'>{val}</div>
          <div class='kpi-sub'>{sub if sub else "&nbsp;"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        dept_out = d.groupby(["department","outcome"]).size().reset_index(name="n")
        fig1 = px.bar(dept_out, x="department", y="n",
                      color="outcome", barmode="stack",
                      color_discrete_map={
                          "Pass":"rgba(5,46,22,0.9)","At-Risk":"rgba(120,53,15,0.9)","Fail":"rgba(127,29,29,0.9)"},
                      title="Outcome by Department")
        fig1.update_layout(**plot_layout(340,
            legend=dict(title="", orientation="h", y=1.08, x=0,
                        font=dict(color="rgba(255,255,255,0.4)", size=10))
        ))
        r1c1.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        fig2 = px.histogram(d, x="final_gpa", nbins=20,
                            color="outcome", barmode="overlay",
                            opacity=0.75,
                            color_discrete_map={
                                "Pass":"#4ade80","At-Risk":"#fbbf24","Fail":"#f87171"},
                            title="GPA Distribution by Outcome")
        fig2.update_layout(**plot_layout(340,
            legend=dict(title="", orientation="h", y=1.08, x=0,
                        font=dict(color="rgba(255,255,255,0.4)", size=10))
        ))
        r1c2.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        fig3 = px.scatter(d.sample(min(800, len(d))),
                          x="study_hours_per_day", y="final_score",
                          color="outcome", size="attendance_pct",
                          hover_data=["attendance_pct","prev_gpa"],
                          color_discrete_map={
                              "Pass":"#4ade80","At-Risk":"#fbbf24","Fail":"#f87171"},
                          title="Study Hours vs Final Score",
                          opacity=0.65)
        fig3.update_layout(**plot_layout(340,
            legend=dict(title="", orientation="h", y=1.08, x=0,
                        font=dict(color="rgba(255,255,255,0.4)", size=10))
        ))
        r2c1.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        city_pass = (d[d.outcome=="Pass"].groupby("city").size() /
                     d.groupby("city").size() * 100
                    ).reset_index(name="pass_rate")
        fig4 = px.bar(city_pass.sort_values("pass_rate"),
                      x="pass_rate", y="city", orientation="h",
                      color="pass_rate",
                      color_continuous_scale=["rgba(13,28,64,1)","rgba(29,78,216,1)","rgba(99,155,255,1)"],
                      title="Pass Rate by City (%)")
        fig4.update_layout(**plot_layout(340, coloraxis_showscale=False))
        r2c2.plotly_chart(fig4, use_container_width=True)

    section("Feature Correlation Matrix")
    num_cols = ["study_hours_per_day","sleep_hours","attendance_pct",
                "prev_gpa","assignment_score","quiz_avg",
                "lab_score","stress_level","internet_hours","final_score"]
    corr = d[num_cols].corr().round(2)
    fig5 = px.imshow(corr, text_auto=True,
                     color_continuous_scale="RdBu",
                     zmin=-1, zmax=1,
                     title="Pearson Correlation — Key Features")
    fig5.update_layout(**plot_layout(460))
    st.plotly_chart(fig5, use_container_width=True)

    section("Score Distributions by Outcome")
    box_feat = st.selectbox(
        "Feature to analyse",
        ["study_hours_per_day","attendance_pct","prev_gpa",
         "stress_level","sleep_hours","assignment_score"],
        label_visibility="collapsed"
    )
    fig6 = px.box(d, x="outcome", y=box_feat, color="outcome",
                  color_discrete_map={
                      "Pass":"#4ade80","At-Risk":"#fbbf24","Fail":"#f87171"},
                  points="outliers",
                  title=f"{box_feat.replace('_',' ').title()} by Outcome")
    fig6.update_layout(**plot_layout(360, showlegend=False))
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "Model Insights":
    page_header(
        "Model",
        "Model Insights & Explainability",
        "Per-class performance, feature importances, model card and raw diagnostics."
    )

    if not ok:
        st.error("Run `python train.py` first.")
        st.stop()

    section("Per-Class Performance")
    report = metrics["class_report"]
    rows = []
    for cls in le.classes_:
        r = report[cls]
        rows.append({
            "Class":     cls,
            "Precision": f"{r['precision']*100:.1f}%",
            "Recall":    f"{r['recall']*100:.1f}%",
            "F1-Score":  f"{r['f1-score']*100:.1f}%",
            "Support":   int(r['support']),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    section("Feature Importance — Random Forest")
    fi  = metrics["feature_importances"]
    fi_df = (pd.DataFrame.from_dict(fi, orient="index", columns=["importance"])
               .reset_index().rename(columns={"index":"feature"})
               .sort_values("importance", ascending=False))
    fig_fi = px.bar(fi_df, x="feature", y="importance",
                    color="importance",
                    color_continuous_scale=["rgba(13,28,64,1)","rgba(29,78,216,1)","rgba(99,155,255,1)"],
                    title="All Feature Importances (sorted)")
    fig_fi.update_layout(**plot_layout(400, coloraxis_showscale=False,
                         xaxis=dict(tickangle=-40,
                                    gridcolor="rgba(255,255,255,0.04)")))
    st.plotly_chart(fig_fi, use_container_width=True)

    section("Model Card")
    cl, cr = st.columns(2, gap="medium")
    left_items  = [
        ("Architecture",  "Soft-Voting Ensemble"),
        ("Learners",      "RF · GBM · SVC · LR"),
        ("Problem Type",  "3-class: Pass / At-Risk / Fail"),
        ("Features",      f"{metrics['n_features']} engineered from {len(FEATURES)} raw"),
    ]
    right_items = [
        ("Training Size", f"{metrics['n_train']:,} students"),
        ("Test Size",     f"{metrics['n_test']:,} students"),
        ("ROC-AUC",       f"{metrics['roc_auc']*100:.2f}%"),
        ("CV Accuracy",   f"{metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%"),
    ]
    for side, items in [(cl, left_items), (cr, right_items)]:
        rows_html = "".join(f"""
        <div style='display:flex;justify-content:space-between;
              padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.04)'>
          <span style='color:rgba(255,255,255,0.3);font-size:.78rem;font-weight:600'>{k}</span>
          <span style='color:rgba(255,255,255,0.6);font-size:.78rem;
                font-family:JetBrains Mono,monospace'>{v}</span>
        </div>""" for k, v in items)
        side.markdown(f"""
        <div style='background:rgba(255,255,255,0.02);
              border:1px solid rgba(255,255,255,0.05);
              border-radius:12px;padding:18px'>{rows_html}</div>
        """, unsafe_allow_html=True)

    with st.expander("Raw Metrics JSON"):
        display = {k: v for k, v in metrics.items()
                   if k not in ("confusion_matrix","class_report","feature_importances")}
        st.json(display)
