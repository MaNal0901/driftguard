"""
Préparation des données de consommation énergétique pour DriftGuard.

Pipeline :
1. Nettoyage : dédoublonnage + réindexation horaire complète + interpolation
2. Feature engineering : extraction de features calendaires depuis Datetime
3. Split : train (entraînement du modèle) / reference (baseline pour le drift)
           / current (données "production" simulées, avec drift réel observé)
"""

import os

import pandas as pd
import numpy as np
import holidays

US_HOLIDAYS = holidays.US(years=range(2002, 2019))

RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw/PJME_hourly.csv")
PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", "data/processed")

TARGET_COLUMN = "PJME_MW"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Charge le CSV brut et convertit Datetime en vrai type datetime.

    Objectif dans le pipeline : point d'entrée unique des données brutes.
    Toute la suite du pipeline suppose que Datetime est bien typé.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset introuvable à {path}. "
            "Télécharge PJME_hourly.csv depuis Kaggle et place-le dans data/raw/."
        )
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les deux artefacts identifiés dans l'EDA, tous deux liés
    au changement d'heure US (DST) :
    - 4 timestamps dupliqués (passage à l'heure d'hiver) → on garde
      la première occurrence
    - 30 heures manquantes (passage à l'heure d'été) → on réindexe sur
      une plage horaire complète et on interpole les trous

    Objectif dans le pipeline : garantir une série temporelle continue
    et sans doublon. C'est indispensable pour le resampling et la
    décomposition dans les étapes suivantes (train.py utilise les
    features dérivées du temps, qui supposent une grille horaire propre).
    """
    df = df.drop_duplicates(subset="Datetime", keep="first")
    df = df.set_index("Datetime")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].interpolate(method="linear")

    df.index.name = "Datetime"
    return df.reset_index()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features calendaires + encodage cyclique + jours fériés US.

    Objectif dans le pipeline : encoder hour/month/dayofweek en sin/cos
    préserve leur continuité circulaire (23h est proche de 0h, décembre
    est proche de janvier) — un encodage brut ne le capture pas.
    is_holiday ajoute un signal réel (baisse d'activité) non couvert
    par is_weekend. Toutes ces features restent dérivées UNIQUEMENT du
    calendrier — aucune ne dépend de l'historique de la target, donc
    le monitoring de drift reste interprétable sans ambiguïté.
    """
    df = df.copy()
    df["year"] = df["Datetime"].dt.year
    df["month"] = df["Datetime"].dt.month
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["Datetime"].dt.normalize().isin(US_HOLIDAYS).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def split_train_reference_current(
    df: pd.DataFrame,
    train_years: tuple = (2002, 2015),
    reference_year: int = 2016,
    current_year: int = 2017,
):
    """
    Découpe la série en 3 sous-ensembles disjoints et chronologiquement
    ordonnés — aucun chevauchement entre train, reference et current :

    - train      : 2002-2015 → entraînement du modèle
    - reference  : 2016 → "photographie" de la normalité juste avant le
                   déploiement, sert de baseline pour Evidently
    - current    : 2017 → données de production simulées, comparées à
                   reference pour détecter le drift

    Objectif dans le pipeline : cette séparation stricte évite toute
    ambiguïté sur ce que "reference" représente — ce n'est pas un
    échantillon du train, c'est un point de comparaison indépendant,
    chronologiquement postérieur à l'entraînement.
    """
    train = df[
        (df["year"] >= train_years[0]) & (df["year"] <= train_years[1])
    ].reset_index(drop=True)

    reference = df[df["year"] == reference_year].reset_index(drop=True)
    current = df[df["year"] == current_year].reset_index(drop=True)

    return train, reference, current

def save_processed(train, reference, current, out_dir: str = PROCESSED_DIR):
    """
    Objectif dans le pipeline : matérialiser les 3 datasets sur disque
    pour que train.py et drift_check.py puissent les lire indépendamment,
    sans dépendre l'un de l'autre ni de preprocessing.py à l'exécution.
    C'est ce découplage qui permet à DVC de traiter chaque étape comme
    un stage séparé et rejouable indépendamment.
    """
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    reference.to_csv(os.path.join(out_dir, "reference.csv"), index=False)
    current.to_csv(os.path.join(out_dir, "current.csv"), index=False)

    print(f"[preprocessing] Données sauvegardées dans {out_dir}/")
    print(f"  train.csv     : {len(train)} lignes ({train['year'].min()}-{train['year'].max()})")
    print(f"  reference.csv : {len(reference)} lignes (année {reference['year'].iloc[0]})")
    print(f"  current.csv   : {len(current)} lignes (année {current['year'].iloc[0]})")


def run():
    """
    Objectif dans le pipeline : c'est le point d'entrée exécuté par
    `python -m src.data.preprocessing` ou par le stage `preprocessing`
    du dvc.yaml. Enchaîne toutes les étapes dans l'ordre.
    """
    df = load_raw_data()
    df = clean_data(df)
    df = add_calendar_features(df)
    train, reference, current = split_train_reference_current(df)
    save_processed(train, reference, current)


if __name__ == "__main__":
    run()