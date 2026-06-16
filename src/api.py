"""API d'inference pour la prediction de pit stop F1 (FastAPI).

Seance 12 - TP FastAPI
    Reçoit les features brutes d'un tour de course et retourne la probabilite
    que le pilote rentre aux stands au tour suivant (PitNextLap).
    Le feature engineering est applique cote serveur (etapes deterministes).
    Les encodages dependants du dataset (target enc, freq enc, LapTime_relative)
    sont substitues par des valeurs neutres.

Lancement :
    uvicorn src.api:app --reload
    ou : make api
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR, MODEL_FEATURES
from src.feature import (
    add_interaction_features,
    add_position_features,
    add_race_progress_features,
    add_stint_features,
    add_tyre_features,
    encode_compound,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


# ── Chargement du modele (une seule fois au demarrage) ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists():
        ml["model"] = joblib.load(model_path)
        logger.info("Modele charge : %s", model_path)
    else:
        logger.warning("Aucun modele trouve dans %s — lancez d'abord make train", MODEL_DIR)
    yield
    ml.clear()


app = FastAPI(
    title="F1 Pit Stop Prediction API",
    description="Predit si un pilote va rentrer aux stands au tour suivant.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── S12-1 : Schema d'entree ───────────────────────────────────────────────────

class Features(BaseModel):
    # Colonnes numeriques brutes (disponibles en temps reel pendant la course)
    Year: int               = Field(..., ge=2018, le=2030,  description="Annee de la course")
    LapNumber: int          = Field(..., ge=1,              description="Numero du tour")
    Stint: int              = Field(..., ge=1,              description="Numero du relais")
    TyreLife: int           = Field(..., ge=0,              description="Age des pneus (tours)")
    Position: int           = Field(..., ge=1, le=20,       description="Position en course")
    LapTime_s: float        = Field(..., gt=0,              description="Temps au tour (secondes)")
    LapTime_Delta: float    = Field(0.0,                    description="Variation vs tour precedent")
    Cumulative_Degradation: float = Field(0.0, ge=0,       description="Degradation cumulee")
    RaceProgress: float     = Field(..., ge=0.0, le=1.0,   description="Avancement de la course (0-1)")
    Position_Change: int    = Field(0,                      description="Positions gagnees/perdues ce tour")
    PitStop: int            = Field(0, ge=0, le=1,          description="Pit stop ce tour (0=non, 1=oui)")
    # Compound (encodage deterministe, ne necessite pas le dataset)
    Compound: str           = Field(...,                    description="SOFT | MEDIUM | HARD | INTERMEDIATE | WET")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Year": 2024,
                "LapNumber": 35,
                "Stint": 2,
                "TyreLife": 20,
                "Position": 4,
                "LapTime_s": 91.8,
                "LapTime_Delta": 0.4,
                "Cumulative_Degradation": 1.5,
                "RaceProgress": 0.57,
                "Position_Change": 0,
                "PitStop": 0,
                "Compound": "MEDIUM",
            }]
        }
    }


# ── S12-2 : Schema de sortie ──────────────────────────────────────────────────

class PredictionOut(BaseModel):
    prediction: int     = Field(..., description="Classe predite : 0 (no pit) ou 1 (pit)")
    probability: float  = Field(..., description="Probabilite d'entrer aux stands au tour suivant")


# ── Feature engineering cote serveur ─────────────────────────────────────────

def build_row_for_model(features: Features) -> pd.DataFrame:
    """Construit le vecteur de features attendu par le modele a partir des entrees brutes."""
    row = pd.DataFrame([{
        "Year":                   features.Year,
        "PitStop":                features.PitStop,
        "LapNumber":              features.LapNumber,
        "Stint":                  features.Stint,
        "TyreLife":               features.TyreLife,
        "Position":               features.Position,
        "LapTime (s)":            features.LapTime_s,
        "LapTime_Delta":          features.LapTime_Delta,
        "Cumulative_Degradation": features.Cumulative_Degradation,
        "RaceProgress":           features.RaceProgress,
        "Position_Change":        features.Position_Change,
        "Compound":               features.Compound.upper(),
    }])

    # Feature engineering deterministe (pas besoin du dataset complet)
    row = encode_compound(row)
    row = add_tyre_features(row)
    row = add_race_progress_features(row)
    row = add_stint_features(row)
    row = add_position_features(row)
    row = add_interaction_features(row)

    # Encodages dataset-dependants → valeurs neutres
    row["Driver_target_enc"] = 0.0
    row["Race_target_enc"]   = 0.0
    row["Driver_freq_enc"]   = 0.0
    row["Race_freq_enc"]     = 0.0
    row["LapTime_relative"]  = 1.0  # temps relatif neutre (= moyenne du circuit)

    # Selectionner uniquement les features attendues par le modele, dans le bon ordre
    cols = [c for c in MODEL_FEATURES if c in row.columns]
    return row[cols]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# S12-4 : prediction
@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge — lancez make train d'abord")
    row   = build_row_for_model(features)
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


# S12-5 bonus : infos sur le modele servi
@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
