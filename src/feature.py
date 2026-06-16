import pandas as pd
import numpy as np

from src.config import (
    MAX_TYRE_LIFE,
    COMPOUND_ORDER,
    PIT_WINDOW_START,
    PIT_WINDOW_END,
    TARGET,
)


def encode_compound(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Compound_ordinal"] = df["Compound"].map(COMPOUND_ORDER)
    df["Is_Soft"]          = (df["Compound"] == "SOFT").astype(int)
    df["Is_Hard"]          = (df["Compound"] == "HARD").astype(int)
    df["Is_Medium"]        = (df["Compound"] == "MEDIUM").astype(int)
    df["Is_Wet_Compound"]  = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    return df


def add_tyre_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TyreLife_sq"]  = df["TyreLife"] ** 2
    df["TyreLife_log"] = np.log1p(df["TyreLife"])

    df["DegradationRate"] = np.where(
        df["TyreLife"] > 0,
        df["Cumulative_Degradation"] / df["TyreLife"],
        0,
    )
    df["DegradationPerLap"] = np.where(
        df["TyreLife"] > 1,
        df["LapTime_Delta"] / df["TyreLife"],
        df["LapTime_Delta"],
    )

    df["Max_TyreLife"]       = df["Compound"].map(MAX_TYRE_LIFE)
    df["TyreLife_Remaining"] = df["Max_TyreLife"] - df["TyreLife"]
    df["TyreLife_Pct"]       = (df["TyreLife"] / df["Max_TyreLife"]).clip(0, 1)

    return df


def add_race_progress_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RemainingRace"]  = 1 - df["RaceProgress"]
    df["In_Pit_Window"]  = (
        (df["RaceProgress"] >= PIT_WINDOW_START) & (df["RaceProgress"] <= PIT_WINDOW_END)
    ).astype(int)
    df["Is_Early_Race"]  = (df["RaceProgress"] < 0.20).astype(int)
    df["Is_Mid_Race"]    = (
        (df["RaceProgress"] >= 0.20) & (df["RaceProgress"] < 0.70)
    ).astype(int)
    df["Is_Late_Race"]   = (df["RaceProgress"] >= 0.70).astype(int)
    return df


def add_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Is_First_Stint"]     = (df["Stint"] == 1).astype(int)
    df["Is_Last_Stint"]      = (df["Stint"] >= 3).astype(int)
    df["TyreLife_Per_Stint"] = np.where(
        df["Stint"] > 0,
        df["TyreLife"] / df["Stint"],
        df["TyreLife"],
    )
    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Is_Leading"]          = (df["Position"] == 1).astype(int)
    df["Is_On_Podium"]        = (df["Position"] <= 3).astype(int)
    df["Is_In_Points"]        = (df["Position"] <= 10).astype(int)
    df["Is_Losing_Pos"]       = (df["Position_Change"] < 0).astype(int)
    df["Is_Gaining_Pos"]      = (df["Position_Change"] > 0).astype(int)
    df["Position_Change_abs"] = df["Position_Change"].abs()
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Compound_x_TyreLife"]      = df["Compound_ordinal"] * df["TyreLife"]
    df["TyreLife_x_Position"]      = df["TyreLife"] * df["Position"]
    df["Degradation_x_Progress"]   = df["Cumulative_Degradation"] * df["RaceProgress"]
    df["Tyre_Stress"]              = df["TyreLife_Pct"] * df["DegradationRate"].abs()
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Driver", "Race"]:
        df[f"{col}_target_enc"] = df.groupby(col)[TARGET].transform("mean")
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq_enc"] = df[col].map(freq)
    return df


def normalize_laptime_by_circuit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    circuit_mean = df.groupby("Race")["LapTime (s)"].transform("mean")
    df["LapTime_relative"] = df["LapTime (s)"] / circuit_mean
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_compound(df)
    df = add_tyre_features(df)
    df = add_race_progress_features(df)
    df = add_stint_features(df)
    df = add_position_features(df)
    df = add_interaction_features(df)
    df = encode_categoricals(df)
    df = normalize_laptime_by_circuit(df)
    return df


