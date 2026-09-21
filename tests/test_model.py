import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.train import train_model, evaluate_model, FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def fake_train_df():
    """
    Dataset synthétique avec les 8 features réelles du modèle énergie
    (pas de dépendance au vrai CSV).
    """
    n = 300
    rng = np.random.default_rng(1)
    data = {}
    for col in FEATURE_COLUMNS:
        if "sin" in col or "cos" in col:
            data[col] = rng.uniform(-1, 1, size=n)
        else:
            data[col] = rng.integers(0, 2, size=n)
    data[TARGET_COLUMN] = rng.uniform(20000, 50000, size=n)
    return pd.DataFrame(data)


def test_train_model_returns_fitted_regressor(fake_train_df):
    model = train_model(fake_train_df)
    assert isinstance(model, RandomForestRegressor)

    preds = model.predict(fake_train_df[FEATURE_COLUMNS])
    assert len(preds) == len(fake_train_df)


def test_evaluate_model_returns_expected_metrics(fake_train_df):
    model = train_model(fake_train_df)
    metrics = evaluate_model(model, fake_train_df)

    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1  # peut être négatif sur données random, mais jamais > 1


def test_model_predictions_are_positive_scale(fake_train_df):
    """
    Vérifie que les prédictions restent dans un ordre de grandeur
    plausible (pas de valeur aberrante type NaN ou négative extrême).
    """
    model = train_model(fake_train_df)
    preds = model.predict(fake_train_df[FEATURE_COLUMNS])

    assert not np.isnan(preds).any()
    assert (preds > 0).all()
