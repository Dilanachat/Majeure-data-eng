"""
Entraînement des 3 modèles avec GridSearchCV + tracking MLflow.

Usage
-----
# Lancer le serveur MLflow d'abord :
#   mlflow server --host 127.0.0.1 --port 5000 \
#       --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Entraîner un modèle :
python -m src.train --model rf
python -m src.train --model xgb
python -m src.train --model lgbm

# Entraîner les 3 d'un coup :
python -m src.train --model all
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")   # pas d'affichage graphique en script
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    TARGET,
    MODEL_FEATURES,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
)
from src.data import load_data, sample_stratified
from src.feature import build_features
from src.model import split_data


# ── Grilles de recherche ──────────────────────────────────────────────────────

GRIDS = {
    "rf": {
        "label": "RandomForest",
        "estimator": RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth":    [10, 15],
            "min_samples_leaf": [3, 5],
        },
        "log_fn": mlflow.sklearn.log_model,
    },
    "xgb": {
        "label": "XGBoost",
        "estimator": XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=4,   # ~80/20 → poids inverse de la classe positive
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
        "param_grid": {
            "n_estimators":  [100, 200],
            "max_depth":     [4, 6],
            "learning_rate": [0.05, 0.1],
        },
        "log_fn": mlflow.xgboost.log_model,
    },
    "lgbm": {
        "label": "LightGBM",
        "estimator": LGBMClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        "param_grid": {
            "n_estimators":  [100, 200],
            "max_depth":     [6, 8],
            "learning_rate": [0.05, 0.1],
            "num_leaves":    [31, 63],
        },
        "log_fn": mlflow.lightgbm.log_model,
    },
}


# ── Pipeline de données ───────────────────────────────────────────────────────

def prepare_data():
    df_full = load_data()
    df      = sample_stratified(df_full)
    df_feat = build_features(df)
    X_train, X_test, y_train, y_test = split_data(df_feat)
    return X_train, X_test, y_train, y_test


# ── Métriques ─────────────────────────────────────────────────────────────────

def compute_metrics(y_test, y_pred, y_proba) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


# ── Artefact : matrice de confusion ──────────────────────────────────────────

def save_confusion_matrix(y_test, y_pred, path: str = "confusion.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Pit (0)", "Pit (1)"],
        ax=ax, colorbar=False, cmap="Blues",
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# ── Entraînement + MLflow ─────────────────────────────────────────────────────

def train(model_key: str, X_train, X_test, y_train, y_test) -> dict:
    cfg   = GRIDS[model_key]
    label = cfg["label"]

    print(f"\n{'='*55}")
    print(f"  {label}  —  GridSearchCV")
    print(f"{'='*55}")

    # TODO (S5-1, S5-2) — connexion au serveur et nom de l'expérience
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # GridSearchCV : scoring sur f1 (classe minoritaire = pit)
    gs = GridSearchCV(
        estimator=cfg["estimator"],
        param_grid=cfg["param_grid"],
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    best_model  = gs.best_estimator_
    best_params = gs.best_params_
    print(f"\nMeilleurs paramètres : {best_params}")

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(classification_report(y_test, y_pred, target_names=["No Pit (0)", "Pit (1)"]))

    # TODO (S5-3) — encadrer dans un run MLflow
    with mlflow.start_run(run_name=label) as run:

        # TODO (S5-4) — logger les paramètres
        mlflow.log_params({"model": label, **best_params})

        # TODO (S5-5) — logger les métriques
        mlflow.log_metrics(metrics)

        # TODO (S5-6) — logger le modèle
        cfg["log_fn"](best_model, artifact_path="model")

        # TODO (S5-7 bonus) — logger la matrice de confusion
        cm_path = save_confusion_matrix(y_test, y_pred, f"confusion_{model_key}.png")
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        run_id = run.info.run_id

    print(f"  Run loggé → {run_id}")
    print(f"  F1={metrics['f1']} | AUC={metrics['roc_auc']}")

    return {"model": best_model, "params": best_params, "metrics": metrics, "run_id": run_id}


# ── Comparaison finale ────────────────────────────────────────────────────────

def print_comparison(results: dict) -> str:
    print(f"\n{'='*55}")
    print("  COMPARAISON DES MODÈLES")
    print(f"{'='*55}")
    print(f"{'Modèle':<22} {'F1':>6} {'AUC':>7} {'Accuracy':>10} {'Recall':>8}")
    print("-" * 55)

    best_name, best_auc = None, 0.0
    for name, res in results.items():
        m = res["metrics"]
        print(
            f"{name:<22} {m['f1']:>6.3f} {m['roc_auc']:>7.3f}"
            f" {m['accuracy']:>10.3f} {m['recall']:>8.3f}"
        )
        if m["roc_auc"] > best_auc:
            best_auc, best_name = m["roc_auc"], name

    print("-" * 55)
    print(f"\n  Meilleur modèle : {best_name}  (AUC-ROC = {best_auc:.4f})")
    print(f"  Voir l'UI MLflow → {MLFLOW_TRACKING_URI}")
    return best_name


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraîne et log un modèle dans MLflow")
    parser.add_argument(
        "--model",
        choices=["rf", "xgb", "lgbm", "all"],
        default="all",
        help="Modèle à entraîner : rf | xgb | lgbm | all",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data()

    keys = list(GRIDS.keys()) if args.model == "all" else [args.model]
    results = {}
    for key in keys:
        results[GRIDS[key]["label"]] = train(key, X_train, X_test, y_train, y_test)

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
