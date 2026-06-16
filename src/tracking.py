"""Suivi MLflow centralisé — setup expérience + traçabilité des données.

Toute la configuration MLflow passe par ce module pour éviter de dupliquer
`set_tracking_uri` + `set_experiment` dans chaque script d'entraînement.
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from src.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configure le tracking MLflow et les métadonnées de l'expérience.

    - Pointe le client vers MLFLOW_TRACKING_URI
    - Crée ou sélectionne l'expérience MLFLOW_EXPERIMENT
    - Applique la description et les tags définis dans config.py

    Idempotente : peut être appelée plusieurs fois sans erreur.
    """
    # TODO (S5-8) implémenté
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = mlflow.MlflowClient()

    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )

    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)

    logger.info("MLflow experiment '%s' prête (id=%s)", MLFLOW_EXPERIMENT, experiment.experiment_id)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Log un DataFrame dans le run courant pour la traçabilité des données.

    Visible dans l'onglet "Datasets" de l'UI MLflow.

    Parameters
    ----------
    df      : DataFrame à référencer (features + cible)
    context : rôle du dataset — "training" ou "evaluation"
    name    : nom logique affiché dans l'UI
    """
    # TODO (S5-9) implémenté
    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH),
        targets=TARGET,
        name=name,
    )
    mlflow.log_input(dataset, context=context)
    logger.info("Dataset '%s' loggé (context=%s, lignes=%d)", name, context, len(df))
