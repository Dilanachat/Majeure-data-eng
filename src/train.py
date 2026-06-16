"""
Entraînement des 3 modèles avec GridSearchCV ou Optuna + tracking MLflow.

Usage
-----
# Lancer le serveur MLflow d'abord :
#   mlflow server --host 127.0.0.1 --port 5000 \
#       --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Entraîner un modèle (GridSearchCV par défaut) :
python -m src.train --model rf
python -m src.train --model xgb
python -m src.train --model lgbm

# Entraîner avec Optuna :
python -m src.train --model rf --optimizer optuna --n-trials 50

# Entraîner les 3 d'un coup :
python -m src.train --model all
python -m src.train --model all --optimizer optuna --n-trials 30
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
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
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

from src.config import TARGET, MODEL_FEATURES, MLFLOW_TRACKING_URI, MODEL_NAME
from src.data import load_data, sample_stratified
from src.feature import build_features
from src.model import split_data
from src.tracking import setup_experiment, log_dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# ── Espaces de recherche Optuna ───────────────────────────────────────────────

OPTUNA_SPACES = {
    "rf": {
        "label": "RandomForest (Optuna)",
        "estimator_cls": RandomForestClassifier,
        "base_params": {"class_weight": "balanced", "n_jobs": -1, "random_state": 42},
        "suggest": lambda trial: {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 400),
            "max_depth":       trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        },
        "log_fn": mlflow.sklearn.log_model,
    },
    "xgb": {
        "label": "XGBoost (Optuna)",
        "estimator_cls": XGBClassifier,
        "base_params": {
            "eval_metric": "logloss",
            "scale_pos_weight": 4,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        },
        "suggest": lambda trial: {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        },
        "log_fn": mlflow.xgboost.log_model,
    },
    "lgbm": {
        "label": "LightGBM (Optuna)",
        "estimator_cls": LGBMClassifier,
        "base_params": {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
        "suggest": lambda trial: {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
            "max_depth":     trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
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
    return X_train, X_test, y_train, y_test, df_feat


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

def train(model_key: str, X_train, X_test, y_train, y_test, df_feat=None) -> dict:
    cfg   = GRIDS[model_key]
    label = cfg["label"]

    print(f"\n{'='*55}")
    print(f"  {label}  —  GridSearchCV")
    print(f"{'='*55}")

    # S5-1, S5-2 — setup centralisé via tracking.py
    setup_experiment()

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

        # S5-4 — paramètres
        mlflow.log_params({"model": label, **best_params})

        # S5-5 — métriques
        mlflow.log_metrics(metrics)

        # S5-6 — modèle
        cfg["log_fn"](best_model, artifact_path="model")

        # S5-7 bonus — matrice de confusion
        cm_path = save_confusion_matrix(y_test, y_pred, f"confusion_{model_key}.png")
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # S5-9 — traçabilité des données (via tracking.py)
        if df_feat is not None:
            features = [f for f in MODEL_FEATURES if f in df_feat.columns]
            log_dataset(df_feat[features + [TARGET]].iloc[X_train.index], context="training", name="train")
            log_dataset(df_feat[features + [TARGET]].iloc[X_test.index],  context="evaluation", name="test")

        run_id = run.info.run_id

    mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    print(f"  Run loggé → {run_id}")
    print(f"  F1={metrics['f1']} | AUC={metrics['roc_auc']}")

    return {"model": best_model, "params": best_params, "metrics": metrics, "run_id": run_id}


# ── Entraînement + Optuna + MLflow ───────────────────────────────────────────

def train_optuna(
    model_key: str,
    X_train,
    X_test,
    y_train,
    y_test,
    df_feat=None,
    n_trials: int = 30,
) -> dict:
    cfg   = OPTUNA_SPACES[model_key]
    label = cfg["label"]

    print(f"\n{'='*55}")
    print(f"  {label}  —  Optuna ({n_trials} trials)")
    print(f"{'='*55}")

    setup_experiment()

    def objective(trial):
        params = cfg["suggest"](trial)
        model  = cfg["estimator_cls"](**cfg["base_params"], **params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name=label)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nMeilleurs paramètres : {best_params}")
    print(f"Meilleur F1 CV       : {study.best_value:.4f}")

    # Réentraînement final sur tout le jeu d'entraînement
    best_model = cfg["estimator_cls"](**cfg["base_params"], **best_params)
    best_model.fit(X_train, y_train)

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(classification_report(y_test, y_pred, target_names=["No Pit (0)", "Pit (1)"]))

    with mlflow.start_run(run_name=label) as run:
        mlflow.log_params({
            "model":    label,
            "n_trials": n_trials,
            **best_params,
        })
        mlflow.log_metric("optuna_best_cv_f1", study.best_value)
        mlflow.log_metrics(metrics)

        cfg["log_fn"](best_model, artifact_path="model")

        cm_path = save_confusion_matrix(
            y_test, y_pred, f"confusion_{model_key}_optuna.png"
        )
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        if df_feat is not None:
            features = [f for f in MODEL_FEATURES if f in df_feat.columns]
            log_dataset(
                df_feat[features + [TARGET]].iloc[X_train.index],
                context="training",
                name="train",
            )
            log_dataset(
                df_feat[features + [TARGET]].iloc[X_test.index],
                context="evaluation",
                name="test",
            )

        run_id = run.info.run_id

    mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
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
    parser.add_argument(
        "--optimizer",
        choices=["grid", "optuna"],
        default="grid",
        help="Optimiseur : grid (GridSearchCV) | optuna",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Nombre de trials Optuna (ignoré si --optimizer grid)",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, df_feat = prepare_data()

    keys = list(GRIDS.keys()) if args.model == "all" else [args.model]
    results = {}

    for key in keys:
        if args.optimizer == "optuna":
            res   = train_optuna(key, X_train, X_test, y_train, y_test, df_feat, n_trials=args.n_trials)
            label = OPTUNA_SPACES[key]["label"]
        else:
            res   = train(key, X_train, X_test, y_train, y_test, df_feat)
            label = GRIDS[key]["label"]
        results[label] = res

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
