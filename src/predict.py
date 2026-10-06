"""
Prediction utilities for the Bank Marketing Prediction project.
"""

import os
import joblib

from .preprocessing import prepare_input
# Get the project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Path to the saved model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "bank_marketing_model.joblib"
)


def load_model():
    """
    Load the trained model artifact.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_customer(customer_data):
    """
    Predict whether a customer will subscribe
    to a term deposit.

    Parameters
    ----------
    customer_data : dict or pandas.DataFrame
        Customer information.

    Returns
    -------
    dict
        Prediction result containing probability,
        threshold and predicted class.
    """

    artifact = load_model()

    model = artifact["model"]
    threshold = artifact["threshold"]

    # Prepare input
    input_data = prepare_input(customer_data)

    # Get probability of subscription
    probability = model.predict_proba(input_data)[0][1]

    # Apply optimized threshold
    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "yes"
    else:
        result = "no"

    return {
        "prediction": result,
        "probability": probability,
        "threshold": threshold
    }