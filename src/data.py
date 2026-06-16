import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RAW_DATA,
    TARGET,
    SAMPLE_N_TARGET,
    SAMPLE_STRATA_COLS,
    SAMPLE_MIN_PER_GROUP,
    SAMPLE_RANDOM_STATE,
)


def load_data(path=RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"Shape : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print("=" * 60)

    print("\n── Types & valeurs manquantes ──")
    info = pd.DataFrame({
        "dtype": df.dtypes,
        "null": df.isnull().sum(),
        "null_%": (df.isnull().mean() * 100).round(2),
        "nunique": df.nunique(),
    })
    print(info)

    print("\n── Statistiques numériques ──")
    print(df.describe().round(3))

    print(f"\n── Distribution de la cible ({TARGET}) ──")
    counts = df[TARGET].value_counts()
    pct = df[TARGET].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "%": pct.round(2)}))

    print("\n── Compounds ──")
    print(df["Compound"].value_counts())

    print("\n── Années ──")
    print(df["Year"].value_counts().sort_index())


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Distribution des variables numériques clés", fontsize=14)

    num_cols = ["TyreLife", "LapTime (s)", "LapTime_Delta",
                "Cumulative_Degradation", "RaceProgress", "Position"]

    for ax, col in zip(axes.flat, num_cols):
        df[col].hist(bins=50, ax=ax, edgecolor="none", color="steelblue", alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.show()


def plot_target_by_compound(df: pd.DataFrame) -> None:
    pit_rate = (
        df.groupby("Compound")["PitNextLap"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=pit_rate, x="Compound", y="PitNextLap", ax=ax, palette="viridis")
    ax.set_title("Taux de pit stop par compound")
    ax.set_ylabel("P(PitNextLap = 1)")
    plt.tight_layout()
    plt.show()


def plot_target_by_tyre_life(df: pd.DataFrame) -> None:
    pivot = df.groupby(["Compound", "TyreLife"])["PitNextLap"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for compound, grp in pivot.groupby("Compound"):
        ax.plot(grp["TyreLife"], grp["PitNextLap"], label=compound, alpha=0.8)
    ax.set_title("Probabilité de pitter selon l'âge du pneu")
    ax.set_xlabel("TyreLife (tours)")
    ax.set_ylabel("P(PitNextLap = 1)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=np.number).drop(columns=["id"])
    corr = num_df.corr()[[TARGET]].sort_values(TARGET, ascending=False)
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Corrélation avec PitNextLap")
    plt.tight_layout()
    plt.show()


def sample_stratified(
    df: pd.DataFrame,
    n_target: int = SAMPLE_N_TARGET,
    strata_cols: list[str] | None = None,
    min_per_group: int = SAMPLE_MIN_PER_GROUP,
    random_state: int = SAMPLE_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Échantillonnage stratifié proportionnel.

    Chaque groupe (défini par strata_cols) est représenté proportionnellement
    à sa taille dans le dataset original. Les groupes trop petits reçoivent
    au minimum min_per_group lignes (ou toutes leurs lignes si < min_per_group).

    Parameters
    ----------
    df           : DataFrame source
    n_target     : nombre de lignes cibles
    strata_cols  : colonnes qui définissent les strates
                   (défaut : ["Compound", "PitNextLap"])
    min_per_group: nombre minimum de lignes garanties par groupe
    random_state : graine aléatoire pour la reproductibilité
    """
    if strata_cols is None:
        strata_cols = SAMPLE_STRATA_COLS

    total = len(df)
    fraction = n_target / total

    samples = []
    for _, group in df.groupby(strata_cols, observed=True):
        n_prop = int(len(group) * fraction)
        n_sample = max(min_per_group, n_prop)
        n_sample = min(n_sample, len(group))   # ne pas dépasser le groupe
        samples.append(group.sample(n=n_sample, random_state=random_state))

    result = pd.concat(samples).sample(frac=1, random_state=random_state)
    result = result.reset_index(drop=True)

    # Ajustement final : si on dépasse n_target, on tronque aléatoirement
    if len(result) > n_target:
        result = result.sample(n=n_target, random_state=random_state).reset_index(drop=True)

    return result


def compare_distributions(df_orig: pd.DataFrame, df_sample: pd.DataFrame) -> None:
    """Vérifie que l'échantillon préserve les distributions clés."""
    print("=" * 60)
    print(f"Original : {len(df_orig):>7,} lignes")
    print(f"Échantillon : {len(df_sample):>4,} lignes")
    print("=" * 60)

    print("\n── PitNextLap ──")
    orig_pct = df_orig["PitNextLap"].value_counts(normalize=True) * 100
    samp_pct = df_sample["PitNextLap"].value_counts(normalize=True) * 100
    comp = pd.DataFrame({"Original %": orig_pct, "Échantillon %": samp_pct}).round(2)
    print(comp)

    print("\n── Compound ──")
    orig_c = df_orig["Compound"].value_counts(normalize=True) * 100
    samp_c = df_sample["Compound"].value_counts(normalize=True) * 100
    comp_c = pd.DataFrame({"Original %": orig_c, "Échantillon %": samp_c}).round(2)
    print(comp_c)

    print("\n── Year ──")
    orig_y = df_orig["Year"].value_counts(normalize=True).sort_index() * 100
    samp_y = df_sample["Year"].value_counts(normalize=True).sort_index() * 100
    comp_y = pd.DataFrame({"Original %": orig_y, "Échantillon %": samp_y}).round(2)
    print(comp_y)

    print("\n── Statistiques numériques clés ──")
    cols = ["TyreLife", "RaceProgress", "Position", "LapTime (s)"]
    orig_stats = df_orig[cols].mean().round(3)
    samp_stats = df_sample[cols].mean().round(3)
    print(pd.DataFrame({"Moyenne originale": orig_stats, "Moyenne échantillon": samp_stats}))
