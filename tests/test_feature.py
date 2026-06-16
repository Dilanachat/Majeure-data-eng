import pandas as pd
import numpy as np
import pytest

from src.feature import (
    encode_compound,
    add_tyre_features,
    add_race_progress_features,
    add_stint_features,
    add_position_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Compound": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
            "TyreLife": [5, 15, 30, 10, 8],
            "Cumulative_Degradation": [0.5, 1.2, 2.0, 0.8, 0.6],
            "LapTime_Delta": [0.1, 0.2, 0.3, 0.15, 0.12],
            "RaceProgress": [0.1, 0.4, 0.7, 0.85, 0.05],
            "Stint": [1, 2, 3, 1, 2],
            "Position": [1, 3, 10, 5, 15],
            "Position_Change": [-1, 2, 0, -3, 1],
            "PitNextLap": [0, 1, 0, 1, 0],
            "Driver": ["VER", "HAM", "LEC", "NOR", "ALO"],
            "Race": ["Bahrain", "Monaco", "Bahrain", "Monaco", "Bahrain"],
            "LapTime (s)": [90.0, 72.0, 91.0, 73.0, 92.0],
        }
    )


class TestEncodeCompound:
    def test_ordinal_values(self, sample_df: pd.DataFrame) -> None:
        result = encode_compound(sample_df)
        assert result.loc[result["Compound"] == "SOFT", "Compound_ordinal"].iloc[0] == 4
        assert result.loc[result["Compound"] == "HARD", "Compound_ordinal"].iloc[0] == 2

    def test_binary_flags(self, sample_df: pd.DataFrame) -> None:
        result = encode_compound(sample_df)
        assert result.loc[result["Compound"] == "SOFT", "Is_Soft"].iloc[0] == 1
        assert result.loc[result["Compound"] == "MEDIUM", "Is_Soft"].iloc[0] == 0
        assert result.loc[result["Compound"] == "INTERMEDIATE", "Is_Wet_Compound"].iloc[0] == 1
        assert result.loc[result["Compound"] == "WET", "Is_Wet_Compound"].iloc[0] == 1

    def test_no_mutation(self, sample_df: pd.DataFrame) -> None:
        original_cols = list(sample_df.columns)
        encode_compound(sample_df)
        assert list(sample_df.columns) == original_cols


class TestAddTyreFeatures:
    def test_columns_created(self, sample_df: pd.DataFrame) -> None:
        df = encode_compound(sample_df)
        result = add_tyre_features(df)
        for col in ["TyreLife_sq", "TyreLife_log", "DegradationRate", "Max_TyreLife", "TyreLife_Pct"]:
            assert col in result.columns

    def test_tyre_pct_clipped(self, sample_df: pd.DataFrame) -> None:
        df = encode_compound(sample_df)
        result = add_tyre_features(df)
        assert (result["TyreLife_Pct"] >= 0).all()
        assert (result["TyreLife_Pct"] <= 1).all()

    def test_tyrlife_sq(self, sample_df: pd.DataFrame) -> None:
        df = encode_compound(sample_df)
        result = add_tyre_features(df)
        assert (result["TyreLife_sq"] == result["TyreLife"] ** 2).all()


class TestAddRaceProgressFeatures:
    def test_pit_window(self, sample_df: pd.DataFrame) -> None:
        result = add_race_progress_features(sample_df)
        # RaceProgress=0.4 → in window [0.30, 0.75]
        assert result.loc[result["RaceProgress"] == 0.4, "In_Pit_Window"].iloc[0] == 1
        # RaceProgress=0.1 → before window
        assert result.loc[result["RaceProgress"] == 0.1, "In_Pit_Window"].iloc[0] == 0

    def test_remaining_race(self, sample_df: pd.DataFrame) -> None:
        result = add_race_progress_features(sample_df)
        expected = 1 - sample_df["RaceProgress"]
        pd.testing.assert_series_equal(result["RemainingRace"], expected, check_names=False)

    def test_race_phases_mutually_exclusive(self, sample_df: pd.DataFrame) -> None:
        result = add_race_progress_features(sample_df)
        phase_sum = result["Is_Early_Race"] + result["Is_Mid_Race"] + result["Is_Late_Race"]
        assert (phase_sum == 1).all()


class TestAddStintFeatures:
    def test_first_stint_flag(self, sample_df: pd.DataFrame) -> None:
        result = add_stint_features(sample_df)
        assert result.loc[result["Stint"] == 1, "Is_First_Stint"].iloc[0] == 1
        assert result.loc[result["Stint"] == 2, "Is_First_Stint"].iloc[0] == 0

    def test_last_stint_flag(self, sample_df: pd.DataFrame) -> None:
        result = add_stint_features(sample_df)
        assert result.loc[result["Stint"] == 3, "Is_Last_Stint"].iloc[0] == 1
        assert result.loc[result["Stint"] == 1, "Is_Last_Stint"].iloc[0] == 0


class TestAddPositionFeatures:
    def test_leading_flag(self, sample_df: pd.DataFrame) -> None:
        result = add_position_features(sample_df)
        assert result.loc[result["Position"] == 1, "Is_Leading"].iloc[0] == 1
        assert result.loc[result["Position"] == 3, "Is_Leading"].iloc[0] == 0

    def test_podium_flag(self, sample_df: pd.DataFrame) -> None:
        result = add_position_features(sample_df)
        assert result.loc[result["Position"] == 3, "Is_On_Podium"].iloc[0] == 1
        assert result.loc[result["Position"] == 10, "Is_On_Podium"].iloc[0] == 0

    def test_position_change_abs(self, sample_df: pd.DataFrame) -> None:
        result = add_position_features(sample_df)
        assert (result["Position_Change_abs"] >= 0).all()
