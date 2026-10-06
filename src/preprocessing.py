"""
Preprocessing utilities for the Bank Marketing Prediction project.

This module contains reusable functions for preparing
customer input data before making predictions.
"""

import pandas as pd


# Features expected by the trained model
NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def prepare_input(data):
    """
    Prepare customer input data for prediction.

    Parameters
    ----------
    data : dict or pandas.DataFrame
        Customer information.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the features expected by the model.
    """

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    elif not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    missing_columns = [
        column
        for column in MODEL_FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    return data[MODEL_FEATURES].copy()