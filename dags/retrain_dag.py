"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline : preparation des donnees -> entrainement -> controle qualite.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    # S17-1 : regenerer les features et sauvegarder dans data/train_features.csv
    from src.data import load_data, sample_stratified
    from src.feature import build_features

    df = build_features(sample_stratified(load_data()))
    df.to_csv("data/train_features.csv", index=False)
    logger.info("Donnees preparees : %d lignes x %d colonnes", *df.shape)


def task_train(**context) -> None:
    # S17-2 : entrainer le modele RF et pousser le f1 dans XCom
    from src.train import prepare_data, train

    X_train, X_test, y_train, y_test, df_feat = prepare_data()
    metrics = train("rf", X_train, X_test, y_train, y_test, df_feat)
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    logger.info("Entrainement termine — F1 : %.4f", metrics["f1"])


def task_check_quality(**context) -> None:
    # S17-3 : verifier que le F1 depasse le seuil minimum
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualite insuffisante : F1={f1:.4f} < seuil={QUALITY_THRESHOLD}"
        )
    logger.info("Qualite validee — F1 : %.4f >= %.2f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    # S17-4 : tous les lundis a 3h du matin
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    # S17-5 : ordre d'execution
    prepare >> train_task >> check
