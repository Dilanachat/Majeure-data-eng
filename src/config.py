from pathlib import Path

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DATA   = DATA_DIR / "train.csv"
FEAT_DATA  = DATA_DIR / "train_features.csv"

# ── Cible ────────────────────────────────────────────────────────────────────
# Problème : classification binaire
# 1 = le pilote rentre aux stands le tour suivant
# 0 = il ne rentre pas
TARGET = "PitNextLap"

# ── Colonnes brutes ──────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Driver", "Compound", "Race"]
NUMERICAL_COLS   = [
    "Year",
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]
DROP_COLS = ["id"]   # identifiant sans valeur prédictive

# ── Connaissance métier F1 ────────────────────────────────────────────────────
# Durée de vie max estimée par type de pneu (tours)
MAX_TYRE_LIFE: dict[str, int] = {
    "SOFT":         25,
    "MEDIUM":       40,
    "HARD":         55,
    "INTERMEDIATE": 30,
    "WET":          30,
}

# Encoding ordinal du compound (du plus dur au plus mou / pluie)
COMPOUND_ORDER: dict[str, int] = {
    "WET":          0,
    "INTERMEDIATE": 1,
    "HARD":         2,
    "MEDIUM":       3,
    "SOFT":         4,
}

# Fenêtre typique de pit stop en F1 (% de la course)
PIT_WINDOW_START = 0.30
PIT_WINDOW_END   = 0.75

# ── Échantillonnage ───────────────────────────────────────────────────────────
SAMPLE_N_TARGET    = 20_000
SAMPLE_STRATA_COLS = ["Compound", TARGET]
SAMPLE_MIN_PER_GROUP = 30
SAMPLE_RANDOM_STATE  = 42

# ── Features construites ─────────────────────────────────────────────────────
FEATURES_COMPOUND = [
    "Compound_ordinal",
    "Is_Soft",
    "Is_Hard",
    "Is_Medium",
    "Is_Wet_Compound",
]

FEATURES_TYRE = [
    "TyreLife_sq",
    "TyreLife_log",
    "DegradationRate",
    "DegradationPerLap",
    "Max_TyreLife",
    "TyreLife_Remaining",
    "TyreLife_Pct",
]

FEATURES_RACE_PROGRESS = [
    "RemainingRace",
    "In_Pit_Window",
    "Is_Early_Race",
    "Is_Mid_Race",
    "Is_Late_Race",
]

FEATURES_STINT = [
    "Is_First_Stint",
    "Is_Last_Stint",
    "TyreLife_Per_Stint",
]

FEATURES_POSITION = [
    "Is_Leading",
    "Is_On_Podium",
    "Is_In_Points",
    "Is_Losing_Pos",
    "Is_Gaining_Pos",
    "Position_Change_abs",
]

FEATURES_INTERACTION = [
    "Compound_x_TyreLife",
    "TyreLife_x_Position",
    "Degradation_x_Progress",
    "Tyre_Stress",
]

FEATURES_ENCODING = [
    "Driver_target_enc",
    "Race_target_enc",
    "Driver_freq_enc",
    "Race_freq_enc",
    "LapTime_relative",
]

# Toutes les nouvelles features créées par le feature engineering
NEW_FEATURES: list[str] = (
    FEATURES_COMPOUND
    + FEATURES_TYRE
    + FEATURES_RACE_PROGRESS
    + FEATURES_STINT
    + FEATURES_POSITION
    + FEATURES_INTERACTION
    + FEATURES_ENCODING
)

# Features finales à donner au modèle (numériques brutes + features construites)
MODEL_FEATURES: list[str] = NUMERICAL_COLS + NEW_FEATURES

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT   = "F1-PitStop-Prediction"

MLFLOW_EXPERIMENT_DESCRIPTION = (
    "Prédiction de l'arrêt aux stands au tour suivant (PitNextLap). "
    "Dataset F1 2022-2025, 20 000 lignes échantillonnées de manière stratifiée. "
    "Modèles : RandomForest, XGBoost, LightGBM — sélection par GridSearchCV."
)

MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project":  "F1-PitStop",
    "task":     "binary-classification",
    "target":   TARGET,
    "team":     "Majeure Data",
}

# Alias utilisé par tracking.py pour la traçabilité des données
DATA_PATH = RAW_DATA

# ── Évaluation et Model Registry ─────────────────────────────────────────────
MODEL_NAME       = "F1-PitStop-Classifier"
EVAL_F1_MIN      = 0.50   # seuil minimum F1 (classe pit stop)
EVAL_ROC_AUC_MIN = 0.80   # seuil minimum AUC-ROC
