"""
Entraînement du modèle de prévision de consommation énergétique,
avec tracking MLflow.

Usage:
    python -m src.models.train
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = os.getenv("DATA_PROCESSED_PATH", "data/processed")
MODEL_DIR = os.getenv("MODEL_REGISTRY_PATH", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "energy_forecaster")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "driftguard-energy-forecast")

# Features calendaires validées dans l'EDA — définies UNE SEULE FOIS ici,
# réutilisées telles quelles dans predict.py et drift_check.py
FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dayofweek_sin", "dayofweek_cos",
    "is_weekend", "is_holiday",
]
TARGET_COLUMN = "PJME_MW"


def load_train_data(path: str = None) -> pd.DataFrame:
    """
    Objectif dans le pipeline : point d'entrée unique du train, déjà
    nettoyé et enrichi par preprocessing.py. train.py ne refait AUCUN
    nettoyage — c'est le rôle exclusif de preprocessing.py.
    """
    path = path or os.path.join(PROCESSED_DIR, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord `python -m src.data.preprocessing`."
        )
    return pd.read_csv(path)


def train_model(df: pd.DataFrame, random_state: int = 42) -> RandomForestRegressor:
    """
    Objectif dans le pipeline : apprendre la relation entre les features
    calendaires et la consommation. Un RandomForest capture naturellement
    les interactions non-linéaires (ex: l'effet de `hour` dépend de
    `is_weekend`) sans avoir à les coder à la main.

    Choix retenu après comparaison : testé aussi HistGradientBoosting
    (Option B) sur les mêmes 8 features — résultat quasi identique
    (val R² 0.7039 vs 0.7046 ici). Le plafond de performance vient donc
    des features disponibles (purement calendaires, pas de météo), pas
    de l'algorithme. RandomForest est gardé pour sa simplicité.
    """
    X = df[FEATURE_COLUMNS]
    print("Features utilisées :", list(X.columns))
    y = df[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_model(model, df: pd.DataFrame) -> dict:
    """
    Objectif dans le pipeline : quantifier la qualité du modèle avec des
    métriques de régression. MAE = erreur moyenne en MW ; RMSE pénalise
    plus les grosses erreurs ; R² = part de variance expliquée.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    y_pred = model.predict(X)

    return {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": mean_squared_error(y, y_pred) ** 0.5,
        "r2": r2_score(y, y_pred),
    }


def load_reference_data(path: str = None) -> pd.DataFrame:
    """
    Objectif dans le pipeline : fournir un jeu de validation honnête,
    jamais vu pendant l'entraînement (reference.csv = année 2016,
    chronologiquement après train.csv = 2002-2015).
    """
    path = path or os.path.join(PROCESSED_DIR, "reference.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable.")
    return pd.read_csv(path)


def run():
    """
    Objectif dans le pipeline : point d'entrée exécuté par
    `python -m src.models.train` ou par le stage `train` du dvc.yaml.
    Enchaîne chargement → entraînement → évaluation (train + val)
    → logging MLflow → sauvegarde du modèle pour predict.py et app.py.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = load_train_data()
    df_val = load_reference_data()

    with mlflow.start_run():
        model = train_model(df_train)

        train_metrics = evaluate_model(model, df_train)
        val_metrics = evaluate_model(model, df_val)
        os.makedirs("reports", exist_ok=True)
        with open("reports/train_metrics.json", "w") as f:
            json.dump({"train": train_metrics, "val": val_metrics}, f, indent=2)
        mlflow.log_params(
            {
                "model_type": type(model).__name__,
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "features": FEATURE_COLUMNS,
                "n_train_rows": len(df_train),
            }
        )
        # Préfixer pour bien distinguer les deux dans MLflow
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        mlflow.sklearn.log_model(
            model, name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.joblib")
        joblib.dump(model, model_path)

        print(f"[train] TRAIN → MAE: {train_metrics['mae']:.1f} | RMSE: {train_metrics['rmse']:.1f} | R²: {train_metrics['r2']:.4f}")
        print(f"[train] VAL (2016) → MAE: {val_metrics['mae']:.1f} | RMSE: {val_metrics['rmse']:.1f} | R²: {val_metrics['r2']:.4f}")
        print(f"[train] Modèle sauvegardé -> {model_path}")
        print(f"[train] Run MLflow ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run()