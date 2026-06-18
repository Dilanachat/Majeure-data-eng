"""DAG Airflow - trafic de previsions quotidien.

Seance 17 - TP Airflow (suite)
    Chaque jour a 10h, echantillonne 20 lignes et les envoie en POST /predict.
    Simule un flux de previsions en production.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """Echantillonner N_PREDICTIONS lignes et les envoyer a l'API /predict."""
    import httpx

    from src.config import TARGET
    from src.data import load_data

    API_URL = os.getenv("API_URL", "http://api:8000")

    features = load_data().drop(columns=[TARGET])

    # S17-6 : echantillonner N_PREDICTIONS lignes
    sample = features.sample(n=N_PREDICTIONS, random_state=42)

    # S17-7 : envoyer chaque ligne en POST /predict
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()

    logger.info("%d previsions envoyees a %s", N_PREDICTIONS, API_URL)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    # S17-8 : tous les jours a 10h
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
