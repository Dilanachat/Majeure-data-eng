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

st.set_page_config(
    page_title="F1 Pit Stop Dashboard",
    page_icon="🏎️",
    layout="wide",
)

# ── Style CSS personnalisé ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Titre principal */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #E8002D;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    /* Carte développeurs */
    .dev-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #E8002D;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: white;
    }
    .dev-name { font-weight: 700; font-size: 0.95rem; color: #fff; }
    .dev-role { font-size: 0.78rem; color: #aaa; }
    /* Badge statut */
    .status-ok  { color: #00c851; font-weight: 600; }
    .status-err { color: #E8002D; font-weight: 600; }
    /* Séparateur rouge F1 */
    .f1-divider {
        height: 3px;
        background: linear-gradient(90deg, #E8002D, #ff6b35, #E8002D);
        border-radius: 2px;
        margin: 1rem 0;
    }
    /* Résultat prédiction */
    .pred-pit {
        background: #fff0f0;
        border: 2px solid #E8002D;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #E8002D;
    }
    .pred-nopit {
        background: #f0fff4;
        border: 2px solid #00c851;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00c851;
    }
</style>
""", unsafe_allow_html=True)

# ── Titre principal ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏎️ F1 Pit Stop Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Prédiction en temps réel des arrêts aux stands — F1 2022–2025</div>', unsafe_allow_html=True)
st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏎️ F1 Pit Stop Dashboard")
    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

    # Statut des services
    st.markdown("**État des services**")
    try:
        httpx.get(f"{API_URL}/health", timeout=3)
        st.markdown('<span class="status-ok">● API opérationnelle</span>', unsafe_allow_html=True)
        api_ok = True
    except Exception:
        st.markdown('<span class="status-err">● API hors ligne</span>', unsafe_allow_html=True)
        api_ok = False

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.MlflowClient()
        client.search_experiments()
        st.markdown('<span class="status-ok">● MLflow opérationnel</span>', unsafe_allow_html=True)
        mlflow_ok = True
    except Exception:
        st.markdown('<span class="status-err">● MLflow hors ligne</span>', unsafe_allow_html=True)
        mlflow_ok = False

    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

    # Équipe
    st.markdown("**Développé par**")
    st.markdown("""
<div class="dev-card">
    <div class="dev-name">Fanda Tongnia</div>
    <div class="dev-role">MLOps Engineer — ESGI Majeure Data</div>
</div>
<div class="dev-card">
    <div class="dev-name">Dilane Chatelain</div>
    <div class="dev-role">MLOps Engineer — ESGI Majeure Data</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
    st.caption("ESGI — Majeure Data · Fil rouge MLOps · 2025–2026")


# ── Onglets ────────────────────────────────────────────────────────────────────
tab_home, tab_pred, tab_suivi, tab_eval, tab_table = st.tabs([
    "🏠 Accueil",
    "🏎️ Prédiction",
    "📊 Suivi du modèle",
    "📈 Évaluation",
    "📋 Table de prévisions",
])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 0 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
## Bienvenue sur le Dashboard F1 Pit Stop Prediction

Ce projet est développé dans le cadre du **fil rouge MLOps** (ESGI — Majeure Data).
Il prédit si un pilote de Formule 1 va **rentrer aux stands au tour suivant**, à partir
des données télémétriques disponibles en temps réel pendant la course.

---

### Objectif

> **Prédire `PitNextLap`** — variable binaire :
> - `1` → le pilote rentre aux stands au prochain tour
> - `0` → il continue en piste

Le modèle est entraîné sur des données F1 **2022–2025** (~440 000 tours de course)
avec trois algorithmes comparés : RandomForest, XGBoost et LightGBM.

---

### Contenu des onglets

| Onglet | Description |
|---|---|
| 🏎️ **Prédiction** | Saisir les données d'un tour et obtenir une prédiction via l'API |
| 📊 **Suivi du modèle** | Historique des runs MLflow avec métriques et graphiques |
| 📈 **Évaluation** | Métriques détaillées du dernier modèle (F1, AUC-ROC, Precision, Recall) |
| 📋 **Table de prévisions** | Prédictions batch sur un fichier CSV |
""")

    with col_right:
        st.markdown("### Statistiques du projet")
        st.metric("Dataset", "~440 000 tours", "F1 2022–2025")
        st.metric("Modèles comparés", "3", "RF · XGBoost · LightGBM")
        st.metric("Cible", "PitNextLap", "Classification binaire")

        st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Équipe")
        st.markdown("""
<div class="dev-card">
    <div class="dev-name">Fanda Tongnia</div>
    <div class="dev-role">MLOps Engineer</div>
</div>
<div class="dev-card">
    <div class="dev-name">Dilane Chatelain</div>
    <div class="dev-role">MLOps Engineer</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Architecture")
    st.code("""
data/train.csv  →  src/feature.py  →  src/train.py  →  models/model.joblib
                                            │
                                       MLflow tracking
                                            │
                                   src/api.py (FastAPI :8000)
                                            │
                               frontend/app.py (Streamlit :8501)  ← vous êtes ici
""")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — PRÉDICTION UNITAIRE
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("Prédiction d'un tour de course")
    st.caption("Renseignez les données télémétriques d'un tour pour obtenir une prédiction instantanée.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Données de course**")
            year            = st.number_input("Année",          min_value=2018, max_value=2030, value=2024)
            lap_number      = st.number_input("Numéro de tour", min_value=1,    value=35)
            stint           = st.number_input("Stint",          min_value=1,    value=2)
            tyre_life       = st.number_input("Âge des pneus (TyreLife)", min_value=0, value=20)
            position        = st.number_input("Position",       min_value=1, max_value=20, value=4)
            compound        = st.selectbox("Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"], index=1)

        with col2:
            st.markdown("**Données de performance**")
            lap_time_s      = st.number_input("Temps au tour (s)",         min_value=0.1, value=91.8)
            lap_time_delta  = st.number_input("Delta temps (LapTime_Delta)", value=0.4)
            cumul_deg       = st.number_input("Dégradation cumulée",       min_value=0.0, value=1.5)
            race_progress   = st.slider("Avancement course (RaceProgress)", 0.0, 1.0, 0.57, format="%.0%%")
            position_change = st.number_input("Changement de position",    value=0)
            pit_stop        = st.selectbox("Pit stop ce tour ?", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")

        submitted = st.form_submit_button("🔮 Lancer la prédiction", use_container_width=True, type="primary")

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

            st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
            col_res, col_proba, col_bar = st.columns([1, 1, 2])

            with col_res:
                if pred == 1:
                    st.markdown('<div class="pred-pit">🛞 PIT STOP prévu</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="pred-nopit">✅ Pas de pit stop</div>', unsafe_allow_html=True)

            with col_proba:
                st.metric("Probabilité pit stop", f"{proba:.1%}")

            with col_bar:
                st.markdown(f"**Confiance du modèle**")
                st.progress(proba, text=f"{proba:.1%}")

        except httpx.HTTPStatusError as e:
            st.error(f"Erreur API {e.response.status_code} : {e.response.text}")
        except Exception as e:
            st.error(f"Impossible de joindre l'API : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — SUIVI DU MODÈLE (MLflow)
# ══════════════════════════════════════════════════════════════════════════════
with tab_suivi:
    st.subheader("Suivi des runs MLflow")

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

            st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
            col_f1, col_auc = st.columns(2)
            with col_f1:
                st.markdown("**F1 par run**")
                st.bar_chart(df_runs.set_index("Run")["F1"])
            with col_auc:
                st.markdown("**AUC-ROC par run**")
                st.bar_chart(df_runs.set_index("Run")["AUC-ROC"])


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — ÉVALUATION DU DERNIER MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Évaluation — dernier run")

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

                st.markdown(f"**Run :** `{run.info.run_name}` — `{run.info.run_id[:8]}`")
                st.markdown(f"**Modèle :** `{params.get('model','—')}` | **Date :** {pd.to_datetime(run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M')}")
                st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("F1",        f"{metrics.get('f1', 0):.4f}")
                c2.metric("AUC-ROC",   f"{metrics.get('roc_auc', 0):.4f}")
                c3.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
                c4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                c5.metric("Recall",    f"{metrics.get('recall', 0):.4f}")

                st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
                f1_ok  = metrics.get("f1", 0) >= 0.50
                auc_ok = metrics.get("roc_auc", 0) >= 0.80
                st.markdown("**Validation des seuils**")
                col_v1, col_v2 = st.columns(2)
                col_v1.metric("F1 ≥ 0.50", "✅ Validé" if f1_ok else "❌ Insuffisant",
                              delta=f"{metrics.get('f1', 0) - 0.50:+.4f}")
                col_v2.metric("AUC-ROC ≥ 0.80", "✅ Validé" if auc_ok else "❌ Insuffisant",
                              delta=f"{metrics.get('roc_auc', 0) - 0.80:+.4f}")

                st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
                st.markdown("**Hyperparamètres**")
                st.json({k: v for k, v in params.items() if k != "model"})


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — TABLE DE PRÉVISIONS (batch)
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    st.subheader("Table de prévisions batch")
    st.caption("Importez un CSV ou utilisez l'échantillon pour prédire en masse.")

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

    if st.button("🔮 Lancer les prévisions", use_container_width=True, type="primary"):
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
                        "Avancement course" : f"{float(row.get('RaceProgress', 0)):.0%}",
                        "Réel (PitNextLap)" : int(row["PitNextLap"]) if "PitNextLap" in row else "—",
                        "Prédiction"        : "🛞 PIT" if resp.json()["prediction"] == 1 else "✅ NO PIT",
                        "Proba pit"         : f"{resp.json()['probability']:.1%}",
                    })
                except Exception:
                    errors += 1
                progress_bar.progress((i + 1) / n_rows)

        progress_bar.empty()

        if results:
            df_out = pd.DataFrame(results)
            n_pit  = sum(1 for r in results if "PIT" in r["Prédiction"])
            st.markdown('<div class="f1-divider"></div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tours analysés",   len(results))
            c2.metric("Pit stops prévus", n_pit, delta=f"{n_pit/len(results):.0%} des tours")
            c3.metric("Erreurs",          errors)
            st.dataframe(df_out, use_container_width=True, hide_index=True)
            csv_dl = df_out.to_csv(index=False).encode()
            st.download_button("⬇️ Télécharger les résultats", csv_dl, "predictions.csv", "text/csv")
        else:
            st.error("Aucune prédiction — l'API est-elle démarrée ?")
