"""
Chargement du modèle entraîné et inférence sur de nouvelles dates.

L'API n'attend qu'une date/heure (format ISO) — tout le feature
engineering (encodage cyclique, jours fériés) est calculé en interne,
en réutilisant EXACTEMENT la même logique que preprocessing.py.
"""

import os

import joblib
import pandas as pd

from src.data.preprocessing import add_calendar_features
from src.models.train import FEATURE_COLUMNS

MODEL_DIR = os.getenv("MODEL_REGISTRY_PATH", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "energy_forecaster")

_model = None


def get_model():
    """
    Objectif dans le pipeline : charger le modèle une seule fois en
    mémoire (singleton), pour éviter de relire le fichier .joblib à
    chaque appel — important pour la latence de l'API.
    """
    global _model
    if _model is None:
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} introuvable. Lance d'abord `python -m src.models.train`."
            )
        _model = joblib.load(model_path)
    return _model


def build_features_from_datetime(datetime_str: str) -> pd.DataFrame:
    """
    Objectif dans le pipeline : transformer une simple date/heure en le
    même jeu de features que celui utilisé à l'entraînement.

    Réutilise add_calendar_features() de preprocessing.py — AUCUNE
    formule n'est dupliquée ici. C'est la seule façon de garantir que
    predict.py voit exactement le même feature engineering que train.py.
    """
    dt = pd.to_datetime(datetime_str)
    df = pd.DataFrame({"Datetime": [dt]})
    df = add_calendar_features(df)
    return df[FEATURE_COLUMNS]


def predict_one(datetime_str: str) -> dict:
    """
    Objectif dans le pipeline : point d'entrée principal pour une
    prédiction unique, utilisé par l'API FastAPI.
    """
    model = get_model()
    X = build_features_from_datetime(datetime_str)
    prediction = model.predict(X)[0]

    return {
        "datetime": datetime_str,
        "predicted_consumption_mw": round(float(prediction), 1),
    }


def predict_batch(datetimes: list[str]) -> pd.DataFrame:
    """
    Objectif dans le pipeline : prédire sur plusieurs dates d'un coup
    (utile pour visualiser une courbe de prévision sur le dashboard,
    par exemple les 24 prochaines heures).
    """
    model = get_model()
    df = pd.DataFrame({"Datetime": pd.to_datetime(datetimes)})
    df = add_calendar_features(df)
    X = df[FEATURE_COLUMNS]

    df["predicted_consumption_mw"] = model.predict(X)
    return df[["Datetime", "predicted_consumption_mw"]]


if __name__ == "__main__":
    # Exemple manuel : prédire la consommation pour un lundi à 8h en janvier
    print(predict_one("2018-01-15 08:00:00"))