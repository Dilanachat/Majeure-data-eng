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

# URL publiques pour les liens (déduites depuis l'URL courante ou env)
PUBLIC_API_URL    = os.getenv("PUBLIC_API_URL", API_URL)
PUBLIC_MLFLOW_URL = os.getenv("PUBLIC_MLFLOW_URL", MLFLOW_URI)

st.set_page_config(
    page_title="F1 Pit Stop AI — Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;900&family=Rajdhani:wght@500;600;700&display=swap');

/* Reset & base */
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8e8; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12121f 60%, #0a0a0f 100%);
    border-right: 1px solid #2a2a3e;
}
section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

/* Header principal */
.f1-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0505 50%, #0f0f1a 100%);
    border: 1px solid #E8002D33;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.f1-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 100%;
    background: linear-gradient(180deg, #E8002D, #ff6b35, #E8002D);
}
.f1-header::after {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, #E8002D11 0%, transparent 70%);
    pointer-events: none;
}
.f1-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}
.f1-title span { color: #E8002D; }
.f1-subtitle {
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}
.f1-badge {
    display: inline-block;
    background: #E8002D;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    margin-right: 8px;
}

/* Ligne de séparation */
.f1-stripe {
    height: 3px;
    background: linear-gradient(90deg, #E8002D 0%, #ff6b35 30%, #E8002D 60%, transparent 100%);
    border-radius: 2px;
    margin: 16px 0;
}
.f1-stripe-sm {
    height: 2px;
    background: linear-gradient(90deg, #E8002D 0%, #ff6b35 50%, transparent 100%);
    margin: 10px 0;
}

/* Cartes métriques */
.metric-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a3e;
    border-top: 3px solid #E8002D;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
}
.metric-card .value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #E8002D;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #888;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}
.metric-card .delta {
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 2px;
}

/* Cartes développeurs */
.dev-card {
    background: linear-gradient(135deg, #12121f, #1a1a2e);
    border: 1px solid #2a2a3e;
    border-left: 3px solid #E8002D;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
}
.dev-name { font-weight: 700; font-size: 0.9rem; color: #fff; }
.dev-role { font-size: 0.73rem; color: #888; margin-top: 2px; }

/* Liens externes */
.ext-link {
    display: block;
    background: #12121f;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 8px 14px;
    margin: 5px 0;
    text-decoration: none;
    font-size: 0.82rem;
    color: #ccc !important;
    transition: border-color 0.2s;
}
.ext-link:hover { border-color: #E8002D; color: #fff !important; }
.ext-link .icon { margin-right: 6px; }

/* Statut */
.status-dot-ok  { width:8px; height:8px; border-radius:50%; background:#00e676; display:inline-block; margin-right:6px; box-shadow: 0 0 6px #00e676; }
.status-dot-err { width:8px; height:8px; border-radius:50%; background:#E8002D; display:inline-block; margin-right:6px; box-shadow: 0 0 6px #E8002D; }
.status-row { display:flex; align-items:center; padding: 5px 0; font-size:0.85rem; }

/* Prédiction résultat */
.pred-box {
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    margin: 12px 0;
}
.pred-pit {
    background: linear-gradient(135deg, #1a0505, #2a0808);
    border: 2px solid #E8002D;
}
.pred-nopit {
    background: linear-gradient(135deg, #051a0d, #082a12);
    border: 2px solid #00e676;
}
.pred-title { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; letter-spacing: 2px; }
.pred-pit .pred-title   { color: #E8002D; }
.pred-nopit .pred-title { color: #00e676; }
.pred-sub { font-size: 0.85rem; color: #aaa; margin-top: 6px; }

/* Section headers */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding-left: 12px;
    border-left: 3px solid #E8002D;
    margin: 20px 0 12px 0;
}

/* Table */
.dataframe { background: #12121f !important; }

/* Override Streamlit defaults */
div[data-testid="stMetric"] {
    background: #12121f;
    border: 1px solid #2a2a3e;
    border-top: 2px solid #E8002D;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="stMetricValue"] { color: #E8002D !important; font-family: 'Rajdhani', sans-serif; font-size: 1.8rem !important; }
div[data-testid="stMetricLabel"] { color: #aaa !important; font-size: 0.75rem !important; letter-spacing: 1px; text-transform: uppercase; }
div[data-testid="stMetricDelta"] svg { display: none; }

/* Boutons */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #E8002D, #c0001e) !important;
    color: white !important;
    border: none !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 8px 20px !important;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #E8002D, #c0001e) !important;
    color: white !important;
    border: none !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    color: #888 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #E8002D !important;
    border-bottom: 2px solid #E8002D !important;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div { background: #E8002D !important; }

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div { background: #12121f !important; border-color: #2a2a3e !important; color: #e8e8e8 !important; }

/* Sidebar logo */
.sidebar-logo {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.sidebar-logo span { color: #E8002D; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="f1-header">
    <div>
        <span class="f1-badge">MLOps</span>
        <span class="f1-badge" style="background:#1a1a2e; border:1px solid #E8002D; color:#E8002D;">F1 2022–2025</span>
    </div>
    <div class="f1-title" style="margin-top:10px;">F1 PIT STOP <span>PREDICTION</span></div>
    <div class="f1-subtitle">Artificial Intelligence · Real-time Tyre Strategy · ESGI Majeure Data</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏎️ PIT<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)

    # Statut des services
    st.markdown('<div class="section-header" style="font-size:0.8rem; margin:8px 0;">Services</div>', unsafe_allow_html=True)

    try:
        httpx.get(f"{API_URL}/health", timeout=3)
        st.markdown('<div class="status-row"><div class="status-dot-ok"></div>API opérationnelle</div>', unsafe_allow_html=True)
        api_ok = True
    except Exception:
        st.markdown('<div class="status-row"><div class="status-dot-err"></div>API hors ligne</div>', unsafe_allow_html=True)
        api_ok = False

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        client.search_experiments()
        st.markdown('<div class="status-row"><div class="status-dot-ok"></div>MLflow opérationnel</div>', unsafe_allow_html=True)
        mlflow_ok = True
    except Exception:
        st.markdown('<div class="status-row"><div class="status-dot-err"></div>MLflow hors ligne</div>', unsafe_allow_html=True)
        mlflow_ok = False

    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)

    # Liens externes
    st.markdown('<div class="section-header" style="font-size:0.8rem; margin:8px 0;">Liens</div>', unsafe_allow_html=True)
    st.markdown(f"""
<a class="ext-link" href="{PUBLIC_API_URL}/docs" target="_blank">
    <span class="icon">⚡</span> API FastAPI — Documentation
</a>
<a class="ext-link" href="{PUBLIC_MLFLOW_URL}" target="_blank">
    <span class="icon">📊</span> MLflow — Tracking UI
</a>
""", unsafe_allow_html=True)

    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)

    # Équipe
    st.markdown('<div class="section-header" style="font-size:0.8rem; margin:8px 0;">Développeurs</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="dev-card">
    <div class="dev-name">Fanda Tongnia</div>
    <div class="dev-role">MLOps Engineer · ESGI Majeure Data</div>
</div>
<div class="dev-card">
    <div class="dev-name">Dilane Chatelain</div>
    <div class="dev-role">MLOps Engineer · ESGI Majeure Data</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#444; text-align:center; letter-spacing:1px;">ESGI · Majeure Data · 2025–2026</div>', unsafe_allow_html=True)


# ── Onglets ────────────────────────────────────────────────────────────────────
tab_home, tab_pred, tab_suivi, tab_eval, tab_table = st.tabs([
    "🏠  ACCUEIL",
    "🏎️  PRÉDICTION",
    "📊  SUIVI MLflow",
    "📈  ÉVALUATION",
    "📋  BATCH",
])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 0 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="section-header">Le Projet</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="color:#ccc; line-height:1.8; font-size:0.95rem;">
Projet développé dans le cadre du <strong style="color:white;">fil rouge MLOps</strong>
à l'ESGI — Majeure Data Engineering.<br><br>
L'objectif est de prédire si un pilote de Formule 1 va
<strong style="color:#E8002D;">rentrer aux stands au tour suivant</strong>,
à partir des données télémétriques disponibles en temps réel : état des pneus,
temps au tour, position, avancement de la course.
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Stack Technique</div>', unsafe_allow_html=True)

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("""
<div class="metric-card">
    <div class="value">RF · XGB · LGBM</div>
    <div class="label">Modèles ML</div>
    <div class="delta">GridSearchCV + Optuna</div>
</div>""", unsafe_allow_html=True)
        with col_t2:
            st.markdown("""
<div class="metric-card">
    <div class="value">FastAPI</div>
    <div class="label">API REST</div>
    <div class="delta">Prédiction temps réel</div>
</div>""", unsafe_allow_html=True)
        with col_t3:
            st.markdown("""
<div class="metric-card">
    <div class="value">Airflow</div>
    <div class="label">Orchestration</div>
    <div class="delta">Re-entraînement auto</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
        st.code("""
data/train.csv  →  feature.py  →  train.py  →  model.joblib
                                      │
                                 MLflow :5000
                                      │
                              api.py  :8000  (FastAPI)
                                      │
                           app.py  :8501  (Streamlit) ← ici
                                      │
                            Airflow :8080  (re-train hebdo)
""", language=None)

    with col_side:
        st.markdown('<div class="section-header">Statistiques</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="metric-card" style="margin-bottom:10px;">
    <div class="value">440 000</div>
    <div class="label">Tours de course</div>
    <div class="delta">F1 saisons 2022 – 2025</div>
</div>
<div class="metric-card" style="margin-bottom:10px;">
    <div class="value">3</div>
    <div class="label">Modèles comparés</div>
    <div class="delta">RandomForest · XGBoost · LightGBM</div>
</div>
<div class="metric-card" style="margin-bottom:10px;">
    <div class="value">PitNextLap</div>
    <div class="label">Variable cible</div>
    <div class="delta">Classification binaire 0 / 1</div>
</div>
<div class="metric-card" style="margin-bottom:10px;">
    <div class="value">12</div>
    <div class="label">Features</div>
    <div class="delta">Tyre · Lap · Position · Race</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="f1-stripe-sm"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Équipe</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="dev-card">
    <div class="dev-name">Fanda Tongnia</div>
    <div class="dev-role">MLOps Engineer · ESGI Majeure Data</div>
</div>
<div class="dev-card">
    <div class="dev-name">Dilane Chatelain</div>
    <div class="dev-role">MLOps Engineer · ESGI Majeure Data</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION UNITAIRE
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="section-header">Prédiction d\'un Tour</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#888; font-size:0.88rem; margin-bottom:16px;">Renseignez les données télémétriques d\'un tour pour obtenir une prédiction instantanée via l\'API.</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div style="color:#E8002D; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:8px;">Données de course</div>', unsafe_allow_html=True)
            year            = st.number_input("Année",                  min_value=2018, max_value=2030, value=2024)
            lap_number      = st.number_input("Numéro de tour",         min_value=1, value=35)
            stint           = st.number_input("Stint",                  min_value=1, value=2)
            tyre_life       = st.number_input("Âge des pneus (tours)",  min_value=0, value=20)
            position        = st.number_input("Position en course",     min_value=1, max_value=20, value=4)
            compound        = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=1)

        with col2:
            st.markdown('<div style="color:#E8002D; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:8px;">Données de performance</div>', unsafe_allow_html=True)
            lap_time_s      = st.number_input("Temps au tour (s)",          min_value=0.1, value=91.8)
            lap_time_delta  = st.number_input("Delta temps (s)",             value=0.4)
            cumul_deg       = st.number_input("Dégradation cumulée",         min_value=0.0, value=1.5)
            race_progress   = st.slider("Avancement course", 0.0, 1.0, 0.57, format="%.0%%")
            position_change = st.number_input("Changement de position",      value=0)
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
            col_res, col_gauge = st.columns([2, 1])

            with col_res:
                if pred == 1:
                    st.markdown(f"""
<div class="pred-box pred-pit">
    <div class="pred-title">🛞 PIT STOP PRÉVU</div>
    <div class="pred-sub">Le modèle prédit un arrêt aux stands au prochain tour</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="pred-box pred-nopit">
    <div class="pred-title">✅ PAS DE PIT STOP</div>
    <div class="pred-sub">Le pilote devrait continuer en piste</div>
</div>""", unsafe_allow_html=True)

            with col_gauge:
                st.metric("Probabilité pit stop", f"{proba:.1%}")
                st.markdown(f'<div style="color:#888; font-size:0.78rem; margin-top:-8px; letter-spacing:1px;">CONFIANCE DU MODÈLE</div>', unsafe_allow_html=True)
                st.progress(proba)

        except httpx.HTTPStatusError as e:
            st.error(f"Erreur API {e.response.status_code} : {e.response.text}")
        except Exception as e:
            st.error(f"Impossible de joindre l'API : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — SUIVI MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
with tab_suivi:
    st.markdown('<div class="section-header">Suivi des Runs MLflow</div>', unsafe_allow_html=True)

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
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )
        if not runs:
            st.info("Aucun run enregistré.")
        else:
            rows = []
            for r in runs:
                rows.append({
                    "Run"      : r.info.run_name or r.info.run_id[:8],
                    "Statut"   : r.info.status,
                    "F1"       : round(r.data.metrics.get("f1", float("nan")), 4),
                    "AUC-ROC"  : round(r.data.metrics.get("roc_auc", float("nan")), 4),
                    "Accuracy" : round(r.data.metrics.get("accuracy", float("nan")), 4),
                    "Recall"   : round(r.data.metrics.get("recall", float("nan")), 4),
                    "Modèle"   : r.data.params.get("model", "—"),
                    "Date"     : pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                })
            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)

            st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
            col_f1, col_auc = st.columns(2)
            with col_f1:
                st.markdown('<div style="color:#E8002D; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">F1 par run</div>', unsafe_allow_html=True)
                st.bar_chart(df_runs.set_index("Run")["F1"])
            with col_auc:
                st.markdown('<div style="color:#E8002D; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; font-weight:700; margin-bottom:6px;">AUC-ROC par run</div>', unsafe_allow_html=True)
                st.bar_chart(df_runs.set_index("Run")["AUC-ROC"])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<div class="section-header">Évaluation du Dernier Modèle</div>', unsafe_allow_html=True)

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
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs:
                st.info("Aucun run trouvé.")
            else:
                run     = runs[0]
                metrics = run.data.metrics
                params  = run.data.params

                st.markdown(f'<div style="color:#888; font-size:0.82rem; letter-spacing:1px;">RUN : <span style="color:#fff;">{run.info.run_name}</span> &nbsp;·&nbsp; ID : <span style="color:#E8002D;">{run.info.run_id[:8]}</span> &nbsp;·&nbsp; Modèle : <span style="color:#fff;">{params.get("model","—")}</span> &nbsp;·&nbsp; {pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
                st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("F1",        f"{metrics.get('f1', 0):.4f}")
                c2.metric("AUC-ROC",   f"{metrics.get('roc_auc', 0):.4f}")
                c3.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
                c4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                c5.metric("Recall",    f"{metrics.get('recall', 0):.4f}")

                st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Validation des Seuils</div>', unsafe_allow_html=True)
                f1_ok  = metrics.get("f1", 0) >= 0.50
                auc_ok = metrics.get("roc_auc", 0) >= 0.80
                col_v1, col_v2 = st.columns(2)
                col_v1.metric("F1 ≥ 0.50",
                              "✅ VALIDÉ" if f1_ok else "❌ INSUFFISANT",
                              delta=f"{metrics.get('f1', 0) - 0.50:+.4f}")
                col_v2.metric("AUC-ROC ≥ 0.80",
                              "✅ VALIDÉ" if auc_ok else "❌ INSUFFISANT",
                              delta=f"{metrics.get('roc_auc', 0) - 0.80:+.4f}")

                st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Hyperparamètres</div>', unsafe_allow_html=True)
                st.json({k: v for k, v in params.items() if k != "model"})


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    st.markdown('<div class="section-header">Prévisions Batch</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#888; font-size:0.88rem; margin-bottom:16px;">Importez un CSV ou utilisez l\'échantillon pour prédire plusieurs tours en masse.</div>', unsafe_allow_html=True)

    COL_MAP = {
        "LapTime (s)": "LapTime_s",
        "Year": "Year", "LapNumber": "LapNumber", "Stint": "Stint",
        "TyreLife": "TyreLife", "Position": "Position", "Compound": "Compound",
        "LapTime_Delta": "LapTime_Delta", "Cumulative_Degradation": "Cumulative_Degradation",
        "RaceProgress": "RaceProgress", "Position_Change": "Position_Change", "PitStop": "PitStop",
    }

    uploaded = st.file_uploader("Importer un CSV (colonnes F1 brutes)", type="csv")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
    else:
        st.caption(f"Aucun fichier importé — utilisation de `{SAMPLE_CSV}`")
        try:
            df_raw = pd.read_csv(SAMPLE_CSV)
        except FileNotFoundError:
            st.error(f"Fichier introuvable : {SAMPLE_CSV}")
            st.stop()

    n_rows = st.slider("Nombre de lignes à prédire", min_value=5, max_value=min(200, len(df_raw)), value=20)
    df_sample = df_raw.sample(n=n_rows, random_state=42).reset_index(drop=True)

    if st.button("⚡  LANCER LES PRÉVISIONS BATCH", use_container_width=True):
        results = []
        errors  = 0
        progress_bar = st.progress(0, text="Prédictions en cours…")

        with httpx.Client(base_url=API_URL, timeout=10.0) as c:
            for i, row in df_sample.iterrows():
                try:
                    payload = {api_col: row[csv_col] for csv_col, api_col in COL_MAP.items() if csv_col in row}
                    payload["Compound"] = str(payload.get("Compound", "MEDIUM")).upper()
                    for k in ["Year", "LapNumber", "Stint", "TyreLife", "Position", "Position_Change", "PitStop"]:
                        if k in payload: payload[k] = int(payload[k])
                    for k in ["LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"]:
                        if k in payload: payload[k] = float(payload[k])

                    resp = c.post("/predict", json=payload)
                    resp.raise_for_status()
                    results.append({
                        "Tour"              : int(row.get("LapNumber", i)),
                        "Compound"          : row.get("Compound", "—"),
                        "Âge pneu"          : int(row.get("TyreLife", 0)),
                        "Avancement"        : f"{float(row.get('RaceProgress', 0)):.0%}",
                        "Réel"              : int(row["PitNextLap"]) if "PitNextLap" in row else "—",
                        "Prédiction"        : "🛞 PIT" if resp.json()["prediction"] == 1 else "✅ NO PIT",
                        "Proba"             : f"{resp.json()['probability']:.1%}",
                    })
                except Exception:
                    errors += 1
                progress_bar.progress((i + 1) / n_rows)

        progress_bar.empty()

        if results:
            df_out = pd.DataFrame(results)
            n_pit  = sum(1 for r in results if "PIT" in r["Prédiction"])
            st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tours analysés",   len(results))
            c2.metric("Pit stops prévus", n_pit, delta=f"{n_pit/len(results):.0%} des tours")
            c3.metric("Erreurs",          errors)
            st.dataframe(df_out, use_container_width=True, hide_index=True)
            csv_dl = df_out.to_csv(index=False).encode()
            st.download_button("⬇️  Télécharger les résultats CSV", csv_dl, "predictions_f1.csv", "text/csv")
        else:
            st.error("Aucune prédiction obtenue — vérifiez que l'API est démarrée.")
