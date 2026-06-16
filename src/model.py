import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from lightgbm import LGBMClassifier

from src.config import TARGET, MODEL_FEATURES, MLFLOW_EXPERIMENT


# ── Split ────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train : {X_train.shape[0]:,} lignes | Test : {X_test.shape[0]:,} lignes")
    print(f"Features utilisées : {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


# ── Modèles ──────────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Logistic Regression entraîné.")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    print("Random Forest entraîné.")
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    random_state: int = 42,
) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    print("LightGBM entraîné.")
    return model


# ── Métriques ────────────────────────────────────────────────────────────────

def compute_metrics(y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["No Pit (0)", "Pit (1)"]))
    print(f"AUC-ROC : {metrics['auc_roc']:.4f}")

    return {"y_pred": y_pred, "y_proba": y_proba, **metrics}


# ── Visualisations ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, title: str = "") -> plt.Figure:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Pit (0)", "Pit (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Matrice de confusion {title}")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test: pd.Series, y_proba: np.ndarray, auc: float, title: str = "") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"{title} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title(f"Courbe ROC — {title}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(model, X_train: pd.DataFrame, top_n: int = 20, title: str = "") -> plt.Figure:
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=X_train.columns)
    else:
        importance = pd.Series(
            np.abs(model.coef_[0]), index=X_train.columns
        )
    importance = importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["coral" if i < 5 else "steelblue" for i in range(len(importance))]
    importance.plot(kind="barh", ax=ax, color=colors[::-1], alpha=0.85)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} features — {title}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def plot_roc_comparison(results_dict: dict) -> plt.Figure:
    """Superpose les courbes ROC des 3 modèles sur un même graphique."""
    colors = {"Logistic Regression": "tomato", "Random Forest": "steelblue", "LightGBM": "mediumseagreen"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, "gray"),
                label=f"{name} (AUC = {res['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Comparaison ROC — 3 modèles")
    ax.legend()
    plt.tight_layout()
    return fig


# ── MLflow ───────────────────────────────────────────────────────────────────

def log_model_to_mlflow(
    model,
    model_name: str,
    params: dict,
    metrics: dict,
    X_train: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> str:
    """Enregistre un modèle et ses résultats dans MLflow. Retourne le run_id."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=model_name) as run:
        # Paramètres
        mlflow.log_params(params)

        # Métriques
        mlflow.log_metrics(metrics)

        # Figures
        fig_cm = plot_confusion_matrix(y_test, y_pred, title=model_name)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        fig_roc = plot_roc_curve(y_test, y_proba, metrics["auc_roc"], title=model_name)
        mlflow.log_figure(fig_roc, "roc_curve.png")
        plt.close(fig_roc)

        fig_fi = plot_feature_importance(model, X_train, top_n=20, title=model_name)
        mlflow.log_figure(fig_fi, "feature_importance.png")
        plt.close(fig_fi)

        # Modèle
        if isinstance(model, LGBMClassifier):
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        print(f"  Run ID : {run_id}")

    return run_id


def run_all_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Entraîne les 3 modèles, les évalue et les log dans MLflow."""

    experiments = {
        "Logistic Regression": {
            "train_fn": train_logistic_regression,
            "params": {"C": 1.0, "max_iter": 1000},
        },
        "Random Forest": {
            "train_fn": train_random_forest,
            "params": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5},
        },
        "LightGBM": {
            "train_fn": train_lightgbm,
            "params": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "num_leaves": 63},
        },
    }

    results = {}
    for name, cfg in experiments.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        model  = cfg["train_fn"](X_train, y_train, **cfg["params"])  # type: ignore[operator, arg-type]
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        print(classification_report(y_test, y_pred, target_names=["No Pit (0)", "Pit (1)"]))
        print(f"AUC-ROC : {metrics['auc_roc']:.4f}")

        run_id = log_model_to_mlflow(
            model=model,
            model_name=name,
            params=cfg["params"],  # type: ignore[arg-type]
            metrics=metrics,
            X_train=X_train,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        results[name] = {
            **metrics,
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_test": y_test,
            "run_id": run_id,
        }

    return results
