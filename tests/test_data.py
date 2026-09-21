import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_data,
    add_calendar_features,
    split_train_reference_current,
)


@pytest.fixture
def fake_raw_df():
    """
    Dataset synthétique couvrant 3 heures avec un doublon et un trou,
    pour tester clean_data() sans dépendre du vrai CSV (16 ans de données
    serait trop lourd pour un test unitaire).
    """
    dates = pd.to_datetime(
        [
            "2016-01-01 00:00:00",
            "2016-01-01 01:00:00",
            "2016-01-01 01:00:00",  # doublon volontaire
            # 2016-01-01 02:00:00 volontairement absent (trou)
            "2016-01-01 03:00:00",
        ]
    )
    return pd.DataFrame({"Datetime": dates, "PJME_MW": [30000, 31000, 31000, 33000]})


@pytest.fixture
def fake_multi_year_df():
    """
    Dataset synthétique couvrant plusieurs années (2010-2017), avec les
    colonnes déjà enrichies, pour tester le split sans dépendre du
    vrai volume de données (16 ans en horaire).
    """
    n_per_year = 100
    frames = []
    for year in range(2010, 2018):
        dates = pd.date_range(start=f"{year}-01-01", periods=n_per_year, freq="h")
        frames.append(pd.DataFrame({"Datetime": dates, "PJME_MW": np.random.uniform(20000, 50000, n_per_year)}))
    df = pd.concat(frames).reset_index(drop=True)
    df["year"] = df["Datetime"].dt.year
    return df


def test_clean_data_removes_duplicates(fake_raw_df):
    cleaned = clean_data(fake_raw_df)
    assert cleaned["Datetime"].duplicated().sum() == 0


def test_clean_data_fills_missing_hours(fake_raw_df):
    cleaned = clean_data(fake_raw_df)
    full_range = pd.date_range(
        start=fake_raw_df["Datetime"].min(), end=fake_raw_df["Datetime"].max(), freq="h"
    )
    assert len(cleaned) == len(full_range)
    assert cleaned["PJME_MW"].isnull().sum() == 0


def test_clean_data_interpolates_reasonably(fake_raw_df):
    """L'heure manquante (02h) doit être interpolée entre 01h (31000) et 03h (33000)."""
    cleaned = clean_data(fake_raw_df)
    row_02h = cleaned[cleaned["Datetime"] == "2016-01-01 02:00:00"]
    assert not row_02h.empty
    assert 31000 <= row_02h["PJME_MW"].iloc[0] <= 33000


def test_add_calendar_features_creates_expected_columns(fake_raw_df):
    cleaned = clean_data(fake_raw_df)
    enriched = add_calendar_features(cleaned)

    expected_columns = {
        "year", "month", "hour", "dayofweek", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofweek_sin", "dayofweek_cos",
    }
    assert expected_columns.issubset(set(enriched.columns))


def test_cyclical_encoding_is_bounded(fake_raw_df):
    """Les sin/cos doivent toujours être dans [-1, 1]."""
    cleaned = clean_data(fake_raw_df)
    enriched = add_calendar_features(cleaned)

    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]:
        assert enriched[col].between(-1, 1).all()


def test_split_train_reference_current_disjoint(fake_multi_year_df):
    """
    Point critique validé dans le pipeline : aucun chevauchement d'années
    entre train, reference et current.
    """
    train, reference, current = split_train_reference_current(
        fake_multi_year_df, train_years=(2010, 2015), reference_year=2016, current_year=2017
    )

    train_years = set(train["year"].unique())
    reference_years = set(reference["year"].unique())
    current_years = set(current["year"].unique())

    assert train_years.isdisjoint(reference_years)
    assert train_years.isdisjoint(current_years)
    assert reference_years.isdisjoint(current_years)


def test_split_sizes(fake_multi_year_df):
    train, reference, current = split_train_reference_current(
        fake_multi_year_df, train_years=(2010, 2015), reference_year=2016, current_year=2017
    )
    assert len(train) == 100 * 6  # 2010-2015 = 6 années
    assert len(reference) == 100
    assert len(current) == 100