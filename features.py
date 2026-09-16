"""Shared data loading and feature engineering for SIT720 8.1D.

Imported by build_notebook.py, app.py and build_report.py. Keeping this in one
place guarantees that the notebook, the deployed application and the report all
construct features identically, and that the train/test split is reproducible.

The notebook imports these functions rather than reimplementing them, because a
model trained on differently-constructed features than it is served is the single
most common cause of silent deployment failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import schema

_INT_COLUMNS = ("postcode", "bedrooms", "bathrooms", "car_spaces", "total_rooms",
                "has_pool", "has_water_view")
_FLOAT_COLUMNS = ("sale_price", "distance_to_cbd_km")


def load_dataset(path=None) -> pd.DataFrame:
    """Load and type the collected dataset."""
    path = path or schema.DATASET_PATH
    df = pd.read_csv(path)
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    for col in _INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in _FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features described in Part 2.

    Notes on what is deliberately NOT done here:

    * Land size and internal floor area are absent from this dataset because
      Domain does not expose them for sold listings. They are not imputed and
      not estimated. `total_rooms` is the dwelling-size proxy instead, and
      `price_per_room` is the size-normalised price.
    * Keywords are computed from the verbatim agent description, so a row with no
      description simply scores zero on every flag rather than being dropped.
    """
    df = df.copy()

    df["total_rooms"] = df["bedrooms"].fillna(0) + df["bathrooms"].fillna(0)
    df["price_per_room"] = (
        df["sale_price"] / df["total_rooms"].replace(0, np.nan)
    )

    description = df["agent_description"].fillna("").astype(str).str.lower()
    for flag, keywords in schema.TEXT_KEYWORDS.items():
        df[flag] = description.apply(
            lambda text: int(any(k in text for k in keywords))
        )

    df["description_words"] = description.str.split().str.len().fillna(0).astype(int)
    df["has_description"] = (df["description_words"] >= 10).astype(int)
    df["sale_quarter"] = df["sale_date"].dt.to_period("Q").astype(str)
    df["sale_year_month"] = df["sale_date"].dt.strftime("%Y-%m")
    df["log_price"] = np.log(df["sale_price"])
    return df


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, categorical, all) modelling feature names present in df."""
    numeric = [
        c for c in [
            "bedrooms", "bathrooms", "car_spaces", "total_rooms",
            "distance_to_cbd_km", "has_pool", "has_water_view",
        ] + list(schema.TEXT_KEYWORDS) if c in df.columns
    ]
    categorical = [
        c for c in ("suburb", "property_type", "sale_method", "condition_rating")
        if c in df.columns
    ]
    return numeric, categorical, numeric + categorical


def prepare(path=None) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load, engineer, and return features.

    Returns (df_with_features, numeric_features, categorical_features, feature_cols).
    """
    df = engineer_features(load_dataset(path))
    numeric, categorical, all_features = feature_lists(df)
    return df, numeric, categorical, all_features


def build_input_frame(values: dict) -> pd.DataFrame:
    """Build a one-row feature frame from raw user input (used by the app).

    Mirrors engineer_features without the columns that only exist after
    collection (sale_price, sale_date, price_per_room).
    """
    row = dict(values)

    bedrooms = row.get("bedrooms") or 0
    bathrooms = row.get("bathrooms") or 0
    row["total_rooms"] = float(bedrooms) + float(bathrooms)

    description = str(row.get("agent_description") or "").lower()
    for flag, keywords in schema.TEXT_KEYWORDS.items():
        row[flag] = int(any(k in description for k in keywords))

    return pd.DataFrame([row])
