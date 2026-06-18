"""Frontend Streamlit — F1 Pit Stop Prediction Dashboard."""
from __future__ import annotations

import os
import httpx
import pandas as pd
import streamlit as st

API_URL         = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SAMPLE_CSV      = os.getenv("TRAIN_DATA_PATH", "data/train_sample.csv")
EXPERIMENT_NAME = "F1-PitStop-Prediction"
PUBLIC_API_URL      = os.getenv("PUBLIC_API_URL", API_URL)
PUBLIC_MLFLOW_URL   = os.getenv("PUBLIC_MLFLOW_URL", MLFLOW_URI)
PUBLIC_AIRFLOW_URL  = os.getenv("PUBLIC_AIRFLOW_URL", "http://88.96.50.33:8080")

st.set_page_config(
    page_title="F1 Pit Stop AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;900&family=Rajdhani:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

/* Fond principal — bleu nuit profond, pas noir pur */
.stApp { background: #0b0c1a !important; color: #f0f0f0 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11122a 0%, #0e0f22 100%) !important;
    border-right: 2px solid #E8002D !important;
}
section[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #f0f0f0 !important; }

/* Zone de contenu principal */
.block-container { padding-top: 1.5rem !important; }

/* ── HEADER ─────────────────────────────────────────────────── */
.f1-header {
    background: linear-gradient(135deg, #1a0a0a 0%, #200d0d 40%, #16102a 100%);
    border: 1px solid #E8002D55;
    border-radius: 14px;
    padding: 30px 40px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 40px #E8002D22;
}
.f1-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 7px; height: 100%;
    background: linear-gradient(180deg, #ff0028, #E8002D, #ff6b35);
    border-radius: 14px 0 0 14px;
}
.f1-header::after {
    content: '';
    position: absolute;
    right: -5%;
    top: 50%;
    transform: translateY(-50%);
    width: 280px; height: 280px;
    background: radial-gradient(circle, #E8002D18 0%, transparent 70%);
    pointer-events: none;
}
.f1-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
    text-shadow: 0 0 30px #E8002D44;
}
.f1-title span { color: #E8002D; }
.f1-subtitle {
    font-size: 0.9rem;
    color: #a0a0c0 !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 8px;
}
.f1-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 3px 10px;
    border-radius: 3px;
    text-transform: uppercase;
    margin-right: 6px;
    margin-bottom: 10px;
}
.badge-red   { background: #E8002D; color: #fff !important; }
.badge-white { background: transparent; border: 1px solid #ffffff55; color: #ccc !important; }

/* ── BANDES ─────────────────────────────────────────────────── */
.f1-stripe {
    height: 3px;
    background: linear-gradient(90deg, #E8002D 0%, #ff6b35 40%, #E8002D 70%, transparent 100%);
    border-radius: 2px;
    margin: 20px 0;
}
.f1-stripe-sm {
    height: 2px;
    background: linear-gradient(90deg, #E8002D55 0%, #ff6b3522 60%, transparent 100%);
    margin: 12px 0;
}

/* ── CARTES MÉTRIQUES ────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #161730 0%, #1e2040 100%);
    border: 1px solid #2e3060;
    border-top: 3px solid #E8002D;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 16px #00000044;
}
.metric-card .val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #E8002D !important;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.7rem;
    color: #9090b0 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}
.metric-card .dlt {
    font-size: 0.78rem;
    color: #c0c0d8 !important;
    margin-top: 3px;
}

/* ── CARTES DÉVELOPPEURS ─────────────────────────────────────── */
.dev-card {
    background: linear-gradient(135deg, #161730, #1e2040);
    border: 1px solid #2e3060;
    border-left: 3px solid #E8002D;
    border-radius: 8px;
    padding: 11px 15px;
    margin: 6px 0;
}
.dev-name { font-weight: 700; font-size: 0.92rem; color: #ffffff !important; }
.dev-role { font-size: 0.73rem; color: #9090b0 !important; margin-top: 3px; }

/* ── LIENS SIDEBAR ───────────────────────────────────────────── */
.ext-link {
    display: block;
    background: #161730;
    border: 1px solid #2e3060;
    border-radius: 7px;
    padding: 9px 14px;
    margin: 5px 0;
    text-decoration: none;
    font-size: 0.83rem;
    color: #d0d0f0 !important;
    transition: all 0.2s;
}
.ext-link:hover {
    border-color: #E8002D;
    background: #1e0a12;
    color: #ffffff !important;
}

/* ── STATUT ──────────────────────────────────────────────────── */
.status-row { display: flex; align-items: center; padding: 5px 0; font-size: 0.85rem; color: #d0d0f0 !important; }
.dot-ok  { width:9px; height:9px; border-radius:50%; background:#00e676; display:inline-block; margin-right:8px; box-shadow: 0 0 8px #00e676aa; flex-shrink:0; }
.dot-err { width:9px; height:9px; border-radius:50%; background:#E8002D; display:inline-block; margin-right:8px; box-shadow: 0 0 8px #E8002Daa; flex-shrink:0; }

/* ── SECTION HEADER ──────────────────────────────────────────── */
.sh {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding-left: 14px;
    border-left: 4px solid #E8002D;
    margin: 24px 0 12px 0;
    line-height: 1.3;
}

/* ── PRÉDICTION ──────────────────────────────────────────────── */
.pred-pit {
    background: linear-gradient(135deg, #2a0a0a, #3a0c0c);
    border: 2px solid #E8002D;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px #E8002D33;
}
.pred-nopit {
    background: linear-gradient(135deg, #062015, #0a2e1c);
    border: 2px solid #00e676;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px #00e67633;
}
.pred-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #ffffff !important;
}
.pred-pit   .pred-label { text-shadow: 0 0 20px #E8002D; }
.pred-nopit .pred-label { text-shadow: 0 0 20px #00e676; color: #00e676 !important; }
.pred-sub { font-size: 0.85rem; color: #b0b0c8 !important; margin-top: 8px; }

/* ── STREAMLIT OVERRIDES ─────────────────────────────────────── */
/* Métriques */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161730, #1e2040) !important;
    border: 1px solid #2e3060 !important;
    border-top: 2px solid #E8002D !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricLabel"] {
    color: #9090b0 !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
div[data-testid="stMetricDelta"] { color: #a0a0c0 !important; }

/* Boutons */
div[data-testid="stButton"] button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #E8002D 0%, #c0001e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 5px !important;
    box-shadow: 0 4px 15px #E8002D44 !important;
}

/* Onglets */
div[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    color: #8080a0 !important;
    font-size: 0.9rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #E8002D !important;
}
div[data-testid="stTabs"] { border-bottom: 1px solid #2e3060 !important; }

/* Barre de progression */
div[data-testid="stProgressBar"] > div > div { background: #E8002D !important; }

/* Inputs */
div[data-testid="stNumberInput"] input { background: #161730 !important; border-color: #2e3060 !important; color: #f0f0f0 !important; }
div[data-baseweb="select"] > div { background: #161730 !important; border-color: #2e3060 !important; color: #f0f0f0 !important; }

/* Texte général */
p, span, label, div { color: #f0f0f0; }
.stMarkdown p { color: #d0d0e8 !important; }
h1, h2, h3 { color: #ffffff !important; }

/* Caption / small */
.stCaption { color: #8080a0 !important; }

/* Sidebar logo */
.sb-logo {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.sb-logo span { color: #E8002D; }

/* Download button */
div[data-testid="stDownloadButton"] button {
    background: #161730 !important;
    border: 1px solid #2e3060 !important;
    color: #d0d0f0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px !important;
}

/* JSON viewer */
div[data-testid="stJson"] {
    background: #161730 !important;
    border: 1px solid #2e3060 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
div[data-testid="stJson"] * { color: #d0d0f0 !important; }
div[data-testid="stJson"] span[style*="color: var(--text-color)"] { color: #d0d0f0 !important; }
.stJson { background: #161730 !important; }

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #161730 !important;
    border: 2px dashed #2e3060 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
div[data-testid="stFileUploader"] * { color: #d0d0f0 !important; }
div[data-testid="stFileUploader"] small { color: #9090b0 !important; }
div[data-testid="stFileUploaderDropzone"] {
    background: #161730 !important;
    border: 2px dashed #E8002D55 !important;
    border-radius: 8px !important;
}
div[data-testid="stFileUploaderDropzone"] * { color: #d0d0f0 !important; }
div[data-testid="stFileUploaderDropzone"] button {
    background: #E8002D !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <div>
        <span class="f1-badge badge-red">MLOps</span>
        <span class="f1-badge badge-white">F1 2022–2025</span>
        <span class="f1-badge badge-white">ESGI Majeure Data</span>
    </div>
    <div class="f1-title">F1 PIT STOP <span>PREDICTION</span></div>
    <div class="f1-subtitle">Artificial Intelligence · Real-time Tyre Strategy · Machine Learning</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">🏎️ PIT<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem; color:#E8002D; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">État des services</div>', unsafe_allow_html=True)

    try:
        httpx.get(f"{API_URL}/health", timeout=3)
        st.markdown('<div class="status-row"><div class="dot-ok"></div><span style="color:#d0d0f0;">API opérationnelle</span></div>', unsafe_allow_html=True)
        api_ok = True
    except Exception:
        st.markdown('<div class="status-row"><div class="dot-err"></div><span style="color:#d0d0f0;">API hors ligne</span></div>', unsafe_allow_html=True)
        api_ok = False

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        client.search_experiments()
        st.markdown('<div class="status-row"><div class="dot-ok"></div><span style="color:#d0d0f0;">MLflow opérationnel</span></div>', unsafe_allow_html=True)
        mlflow_ok = True
    except Exception:
        st.markdown('<div class="status-row"><div class="dot-err"></div><span style="color:#d0d0f0;">MLflow hors ligne</span></div>', unsafe_allow_html=True)
        mlflow_ok = False

    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem; color:#E8002D; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">Accès direct</div>', unsafe_allow_html=True)
    st.markdown(f"""
<a class="ext-link" href="{PUBLIC_API_URL}/docs" target="_blank">⚡&nbsp; API FastAPI — Docs interactives</a>
<a class="ext-link" href="{PUBLIC_MLFLOW_URL}" target="_blank">📊&nbsp; MLflow — Tracking UI</a>
<a class="ext-link" href="{PUBLIC_AIRFLOW_URL}" target="_blank">🔁&nbsp; Airflow — Orchestration</a>
""", unsafe_allow_html=True)

    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem; color:#E8002D; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">Équipe</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="dev-card">
    <div class="dev-name">Fanda Tongnia Dilane Chatelain</div>
    <div class="dev-role">MLOps Engineer · ESGI Majeure Data</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem; color:#505070; text-align:center; letter-spacing:1px;">ESGI · Majeure Data · 2025–2026</div>', unsafe_allow_html=True)


# ── Onglets ────────────────────────────────────────────────────────────────────
tab_home, tab_pred, tab_suivi, tab_eval, tab_table = st.tabs([
    "🏠  ACCUEIL",
    "🏎️  PRÉDICTION",
    "📊  SUIVI MLFLOW",
    "📈  ÉVALUATION",
    "📋  BATCH",
])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 0 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="sh">Le Projet</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#c8c8e0; line-height:1.9; font-size:0.95rem;">Projet développé dans le cadre du <strong style="color:#ffffff;">fil rouge MLOps</strong> à l\'ESGI — Majeure Data Engineering.<br><br>L\'objectif est de prédire si un pilote de Formule 1 va <strong style="color:#E8002D;">rentrer aux stands au tour suivant</strong>, à partir des données télémétriques : état des pneus, temps au tour, position, avancement de la course.</p>', unsafe_allow_html=True)

        st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">Stack Technique</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-card"><div class="val">RF · XGB · LGBM</div><div class="lbl">Modèles ML</div><div class="dlt">GridSearchCV + Optuna</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="val">FastAPI</div><div class="lbl">API REST</div><div class="dlt">Prédiction temps réel</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card"><div class="val">Airflow</div><div class="lbl">Orchestration</div><div class="dlt">Re-entraînement auto</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="background:#161730; border:1px solid #2e3060; border-left:3px solid #E8002D;
            border-radius:8px; padding:20px 24px; font-family:monospace; font-size:0.88rem;
            color:#d0d0f0; line-height:2;">
  <span style="color:#E8002D;">data/train.csv</span>
  &nbsp;→&nbsp; <span style="color:#ffffff;">feature.py</span>
  &nbsp;→&nbsp; <span style="color:#ffffff;">train.py</span>
  &nbsp;→&nbsp; <span style="color:#E8002D;">model.joblib</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <span style="color:#9090b0;">MLflow</span>
  <span style="color:#E8002D;">:5000</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <span style="color:#ffffff;">api.py</span>
  <span style="color:#E8002D;">:8000</span>
  <span style="color:#9090b0;">(FastAPI)</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <span style="color:#ffffff;">app.py</span>
  <span style="color:#E8002D;">:8501</span>
  <span style="color:#9090b0;">(Streamlit)</span>
  <span style="color:#E8002D; font-weight:700;">← vous êtes ici</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <span style="color:#ffffff;">Airflow</span>
  <span style="color:#E8002D;">:8080</span>
  <span style="color:#9090b0;">(re-train hebdomadaire)</span>
</div>
""", unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="sh">Statistiques</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="metric-card" style="margin-bottom:10px;">
    <div class="val">440 000</div><div class="lbl">Tours de course</div><div class="dlt">F1 saisons 2022–2025</div>
</div>
<div class="metric-card" style="margin-bottom:10px;">
    <div class="val">3</div><div class="lbl">Modèles comparés</div><div class="dlt">RandomForest · XGBoost · LightGBM</div>
</div>
<div class="metric-card" style="margin-bottom:10px;">
    <div class="val">PitNextLap</div><div class="lbl">Variable cible</div><div class="dlt">Classification binaire 0 / 1</div>
</div>
<div class="metric-card" style="margin-bottom:10px;">
    <div class="val">12</div><div class="lbl">Features</div><div class="dlt">Tyre · Lap · Position · Race</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="f1-stripe-sm" style="margin-top:14px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sh" style="font-size:1rem; margin-top:8px;">Équipe</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="dev-card">
    <div class="dev-name">Fanda Tongnia Dilane Chatelain</div>
    <div class="dev-role">MLOps Engineer · ESGI Majeure Data</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="sh">Prédiction d\'un Tour</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9090b0; font-size:0.88rem; margin-bottom:16px;">Renseignez les données télémétriques pour obtenir une prédiction instantanée via l\'API.</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div style="color:#E8002D; font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:8px;">Données de course</div>', unsafe_allow_html=True)
            year            = st.number_input("Année",                 min_value=2018, max_value=2030, value=2024)
            lap_number      = st.number_input("Numéro de tour",        min_value=1, value=35)
            stint           = st.number_input("Stint",                 min_value=1, value=2)
            tyre_life       = st.number_input("Âge des pneus (tours)", min_value=0, value=20)
            position        = st.number_input("Position en course",    min_value=1, max_value=20, value=4)
            compound        = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=1)
        with col2:
            st.markdown('<div style="color:#E8002D; font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:8px;">Données de performance</div>', unsafe_allow_html=True)
            lap_time_s      = st.number_input("Temps au tour (s)",       min_value=0.1, value=91.8)
            lap_time_delta  = st.number_input("Delta temps (s)",          value=0.4)
            cumul_deg       = st.number_input("Dégradation cumulée",      min_value=0.0, value=1.5)
            race_progress   = st.slider("Avancement course (0.0 → 1.0)", 0.0, 1.0, 0.57)
            position_change = st.number_input("Changement de position",   value=0)
            pit_stop        = st.selectbox("Pit stop ce tour ?", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

        submitted = st.form_submit_button("⚡  LANCER LA PRÉDICTION", use_container_width=True)

    if submitted:
        payload = {
            "Year": int(year), "LapNumber": int(lap_number), "Stint": int(stint),
            "TyreLife": int(tyre_life), "Position": int(position), "Compound": compound,
            "LapTime_s": float(lap_time_s), "LapTime_Delta": float(lap_time_delta),
            "Cumulative_Degradation": float(cumul_deg), "RaceProgress": float(race_progress),
            "Position_Change": int(position_change), "PitStop": int(pit_stop),
        }
        try:
            with httpx.Client(base_url=API_URL, timeout=10.0) as c:
                resp = c.post("/predict", json=payload)
            resp.raise_for_status()
            pred  = resp.json()["prediction"]
            proba = resp.json()["probability"]

            st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
            col_res, col_g = st.columns([2, 1])
            with col_res:
                if pred == 1:
                    st.markdown(f'<div class="pred-pit"><div class="pred-label">🛞 PIT STOP PRÉVU</div><div class="pred-sub">Le modèle prédit un arrêt aux stands au prochain tour</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="pred-nopit"><div class="pred-label">✅ PAS DE PIT STOP</div><div class="pred-sub">Le pilote devrait continuer en piste</div></div>', unsafe_allow_html=True)
            with col_g:
                st.metric("Probabilité pit stop", f"{proba:.1%}")
                st.progress(proba)

        except httpx.HTTPStatusError as e:
            st.error(f"Erreur API {e.response.status_code} : {e.response.text}")
        except Exception as e:
            st.error(f"Impossible de joindre l'API : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — SUIVI MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
with tab_suivi:
    st.markdown('<div class="sh">Suivi des Runs MLflow</div>', unsafe_allow_html=True)

    if not mlflow_ok:
        st.warning("MLflow n'est pas joignable.")
        st.stop()

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()

    experiments = client.search_experiments()
    exp_names   = [e.name for e in experiments]
    exp_choice  = st.selectbox("Expérience", exp_names,
                               index=exp_names.index(EXPERIMENT_NAME) if EXPERIMENT_NAME in exp_names else 0)
    exp         = client.get_experiment_by_name(exp_choice)

    if exp is None:
        st.info("Aucune expérience trouvée.")
    else:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=50)
        if not runs:
            st.info("Aucun run enregistré.")
        else:
            rows = []
            for r in runs:
                rows.append({
                    "Run"     : r.info.run_name or r.info.run_id[:8],
                    "Statut"  : r.info.status,
                    "F1"      : round(r.data.metrics.get("f1", float("nan")), 4),
                    "AUC-ROC" : round(r.data.metrics.get("roc_auc", float("nan")), 4),
                    "Accuracy": round(r.data.metrics.get("accuracy", float("nan")), 4),
                    "Recall"  : round(r.data.metrics.get("recall", float("nan")), 4),
                    "Modèle"  : r.data.params.get("model", "—"),
                    "Date"    : pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                })
            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)
            st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
            col_f1, col_auc = st.columns(2)
            with col_f1:
                st.markdown('<div style="color:#E8002D; font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">F1 par run</div>', unsafe_allow_html=True)
                st.bar_chart(df_runs.set_index("Run")["F1"])
            with col_auc:
                st.markdown('<div style="color:#E8002D; font-size:0.72rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">AUC-ROC par run</div>', unsafe_allow_html=True)
                st.bar_chart(df_runs.set_index("Run")["AUC-ROC"])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="sh">Évaluation du Dernier Modèle</div>', unsafe_allow_html=True)

    if not mlflow_ok:
        st.warning("MLflow n'est pas joignable.")
    else:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)

        if exp is None:
            st.info("Aucune expérience trouvée.")
        else:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=1)
            if not runs:
                st.info("Aucun run trouvé.")
            else:
                run     = runs[0]
                metrics = run.data.metrics
                params  = run.data.params

                st.markdown(f'<p style="color:#9090b0; font-size:0.82rem; letter-spacing:1px;">Run : <strong style="color:#ffffff;">{run.info.run_name}</strong> &nbsp;·&nbsp; ID : <strong style="color:#E8002D;">{run.info.run_id[:8]}</strong> &nbsp;·&nbsp; Modèle : <strong style="color:#ffffff;">{params.get("model","—")}</strong> &nbsp;·&nbsp; <span style="color:#9090b0;">{pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M")}</span></p>', unsafe_allow_html=True)
                st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("F1",        f"{metrics.get('f1', 0):.4f}")
                c2.metric("AUC-ROC",   f"{metrics.get('roc_auc', 0):.4f}")
                c3.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
                c4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                c5.metric("Recall",    f"{metrics.get('recall', 0):.4f}")

                st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sh" style="font-size:1rem;">Validation des seuils</div>', unsafe_allow_html=True)
                f1_ok  = metrics.get("f1", 0) >= 0.50
                auc_ok = metrics.get("roc_auc", 0) >= 0.80
                col_v1, col_v2 = st.columns(2)
                col_v1.metric("F1 ≥ 0.50",     "✅ VALIDÉ" if f1_ok else "❌ INSUFFISANT", delta=f"{metrics.get('f1',0)-0.50:+.4f}")
                col_v2.metric("AUC-ROC ≥ 0.80", "✅ VALIDÉ" if auc_ok else "❌ INSUFFISANT", delta=f"{metrics.get('roc_auc',0)-0.80:+.4f}")

                st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sh" style="font-size:1rem;">Hyperparamètres</div>', unsafe_allow_html=True)
                hp = {k: v for k, v in params.items() if k != "model"}
                rows_hp = "".join(
                    f'<div style="padding:6px 0; border-bottom:1px solid #2e3060; display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="color:#9090b0; font-size:0.85rem; letter-spacing:1px;">{k}</span>'
                    f'<span style="color:#E8002D; font-family:\'Rajdhani\',sans-serif; font-size:1rem; font-weight:700;">{v}</span>'
                    f'</div>'
                    for k, v in hp.items()
                )
                st.markdown(
                    f'<div style="background:#161730; border:1px solid #2e3060; border-radius:8px; padding:12px 18px;">{rows_hp}</div>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    st.markdown('<div class="sh">Prévisions Batch</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#9090b0; font-size:0.88rem; margin-bottom:16px;">Importez un CSV ou utilisez l\'échantillon pour prédire plusieurs tours en masse.</p>', unsafe_allow_html=True)

    COL_MAP = {
        "LapTime (s)": "LapTime_s",
        "Year": "Year", "LapNumber": "LapNumber", "Stint": "Stint",
        "TyreLife": "TyreLife", "Position": "Position", "Compound": "Compound",
        "LapTime_Delta": "LapTime_Delta", "Cumulative_Degradation": "Cumulative_Degradation",
        "RaceProgress": "RaceProgress", "Position_Change": "Position_Change", "PitStop": "PitStop",
    }

    uploaded = st.file_uploader("Importer un CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
    else:
        st.caption(f"Aucun fichier — utilisation de `{SAMPLE_CSV}`")
        try:
            df_raw = pd.read_csv(SAMPLE_CSV)
        except FileNotFoundError:
            st.error(f"Fichier introuvable : {SAMPLE_CSV}")
            st.stop()

    n_rows   = st.slider("Nombre de lignes", min_value=5, max_value=min(200, len(df_raw)), value=20)
    df_sample = df_raw.sample(n=n_rows, random_state=42).reset_index(drop=True)

    if st.button("⚡  LANCER LES PRÉVISIONS BATCH", use_container_width=True):
        results = []
        errors  = 0
        bar = st.progress(0, text="Prédictions en cours…")

        with httpx.Client(base_url=API_URL, timeout=10.0) as c:
            for i, row in df_sample.iterrows():
                try:
                    payload = {api_col: row[csv_col] for csv_col, api_col in COL_MAP.items() if csv_col in row}
                    payload["Compound"] = str(payload.get("Compound", "MEDIUM")).upper()
                    for k in ["Year","LapNumber","Stint","TyreLife","Position","Position_Change","PitStop"]:
                        if k in payload: payload[k] = int(payload[k])
                    for k in ["LapTime_s","LapTime_Delta","Cumulative_Degradation","RaceProgress"]:
                        if k in payload: payload[k] = float(payload[k])
                    resp = c.post("/predict", json=payload)
                    resp.raise_for_status()
                    results.append({
                        "Tour"      : int(row.get("LapNumber", i)),
                        "Compound"  : row.get("Compound", "—"),
                        "Âge pneu"  : int(row.get("TyreLife", 0)),
                        "Avancement": f"{float(row.get('RaceProgress', 0)):.0%}",
                        "Réel"      : int(row["PitNextLap"]) if "PitNextLap" in row else "—",
                        "Prédiction": "🛞 PIT" if resp.json()["prediction"] == 1 else "✅ NO PIT",
                        "Proba"     : f"{resp.json()['probability']:.1%}",
                    })
                except Exception:
                    errors += 1
                bar.progress((i + 1) / n_rows)

        bar.empty()
        if results:
            df_out = pd.DataFrame(results)
            n_pit  = sum(1 for r in results if "PIT" in r["Prédiction"])
            st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tours analysés",   len(results))
            c2.metric("Pit stops prévus", n_pit, delta=f"{n_pit/len(results):.0%} des tours")
            c3.metric("Erreurs",          errors)
            st.dataframe(df_out, use_container_width=True, hide_index=True)
            st.download_button("⬇️  Télécharger CSV", df_out.to_csv(index=False).encode(), "predictions_f1.csv", "text/csv")
        else:
            st.error("Aucune prédiction — l'API est-elle démarrée ?")
