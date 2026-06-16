"""Client de test pour l'API FastAPI du modele F1 Pit Stop.

Envoie quelques payloads de test a une instance locale de l'API
(`make api`) et affiche les reponses de `/health`, `/predict` et
`/model-info`.

Lancement (depuis la racine du projet) :
    PYTHONPATH=. python script/predict_client.py
    PYTHONPATH=. python script/predict_client.py --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import logging

import httpx

from src.config import TARGET, RAW_DATA
from src.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000"
N_SAMPLES = 3

# Colonnes brutes attendues par l'API (noms du CSV → noms JSON du payload)
PAYLOAD_COLS = {
    "Year": "Year",
    "LapNumber": "LapNumber",
    "Stint": "Stint",
    "TyreLife": "TyreLife",
    "Position": "Position",
    "LapTime (s)": "LapTime_s",
    "LapTime_Delta": "LapTime_Delta",
    "Cumulative_Degradation": "Cumulative_Degradation",
    "RaceProgress": "RaceProgress",
    "Position_Change": "Position_Change",
    "PitStop": "PitStop",
    "Compound": "Compound",
}


def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Construire n payloads de test a partir du jeu de donnees.

    Retire la cible, renomme les colonnes selon le schema de l'API,
    et convertit chaque ligne en dict JSON natif.
    """
    df = load_data(RAW_DATA).drop(columns=[TARGET])
    sample = df[list(PAYLOAD_COLS.keys())].sample(n=n)
    payloads = []
    for _, row in sample.iterrows():
        payload = {api_name: row[csv_name] for csv_name, api_name in PAYLOAD_COLS.items()}
        payloads.append(json.loads(json.dumps(payload, default=lambda x: x.item() if hasattr(x, "item") else x)))
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=API_URL, help="URL de base de l'API (defaut: %(default)s)"
    )
    args = parser.parse_args()

    payloads = build_payloads()

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        # GET /health
        health = client.get("/health")
        logger.info("GET /health -> %s %s", health.status_code, health.json())

        # POST /predict pour chaque payload
        for i, payload in enumerate(payloads):
            logger.info("Payload #%d : %s", i, payload)
            response = client.post("/predict", json=payload)
            logger.info("POST /predict (#%d) -> %s %s", i, response.status_code, response.json())

        # GET /model-info
        info = client.get("/model-info")
        logger.info("GET /model-info -> %s %s", info.status_code, info.json())


if __name__ == "__main__":
    main()
