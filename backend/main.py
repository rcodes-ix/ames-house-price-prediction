
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"

# If your train.csv is inside backend/data/train.csv
DATA_PATH = BASE_DIR / "data" / "train.csv"


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

try:
    bundle = joblib.load(MODEL_PATH)

except Exception as e:
    raise RuntimeError(
        f"Could not load model bundle from {MODEL_PATH}: {e}"
    )


# ============================================================
# LOAD MODELS
# ============================================================

elastic_model = bundle["elastic_model"]
rf_model = bundle["rf_model"]
gbr_model = bundle["gbr_model"]

stack_model = bundle["stack_model"]

lower_model = bundle["lower_model"]
upper_model = bundle["upper_model"]

calibration_factor = bundle["calibration_factor"]


# ============================================================
# TRAINING FEATURE INFORMATION
# ============================================================

feature_columns = bundle["feature_columns"]

numeric_features = bundle["numeric_features"]

categorical_features = bundle["categorical_features"]


# ============================================================
# USER-FACING FEATURES
# ============================================================
#
# The website only asks the user for these five features.
#
# The remaining original training features are automatically
# filled with NaN and handled by the preprocessing pipelines.
#

USER_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
]


# Make sure the selected features actually exist
# in the trained model.

missing_user_features = [
    feature
    for feature in USER_FEATURES
    if feature not in feature_columns
]

if missing_user_features:
    raise RuntimeError(
        "The following user-facing features are missing "
        f"from the trained model: {missing_user_features}"
    )


# ============================================================
# FEATURE RANGES
# ============================================================

def load_feature_ranges():
    """
    Load the actual minimum and maximum values of the five
    user-facing features from the Ames training dataset.

    This prevents the frontend from using meaningless ranges
    such as 0, 1, 2, 3 for fields like GrLivArea.
    """

    if not DATA_PATH.exists():
        raise RuntimeError(
            f"Training dataset not found at {DATA_PATH}. "
            "Place train.csv inside backend/data/."
        )

    try:
        data = pd.read_csv(DATA_PATH)

    except Exception as e:
        raise RuntimeError(
            f"Could not read training dataset: {e}"
        )

    ranges = {}

    for feature in USER_FEATURES:

        if feature not in data.columns:
            raise RuntimeError(
                f"Feature '{feature}' does not exist "
                "in the training dataset."
            )

        series = pd.to_numeric(
            data[feature],
            errors="coerce"
        ).dropna()

        if series.empty:
            raise RuntimeError(
                f"No valid numeric values found for {feature}."
            )

        minimum = float(series.min())
        maximum = float(series.max())

        ranges[feature] = {
            "min": minimum,
            "max": maximum
        }

    return ranges


feature_ranges = load_feature_ranges()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Ames House Price Prediction API",
    description=(
        "House price prediction using a stacked ensemble "
        "with calibrated prediction intervals."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):
    features: dict[str, Any]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Ames House Price Prediction API",
        "status": "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# FEATURES ENDPOINT
# ============================================================

@app.get("/features")
def get_features():
    """
    Returns only the five features that the website
    should ask the user to enter.
    """

    return {
        "features": [
            {
                "name": "OverallQual",
                "label": "Overall Quality",
                "type": "numeric",
                "min": feature_ranges["OverallQual"]["min"],
                "max": feature_ranges["OverallQual"]["max"],
                "step": 1,
                "description": (
                    "Overall material and finish quality "
                    "of the house."
                )
            },
            {
                "name": "GrLivArea",
                "label": "Above-Ground Living Area",
                "type": "numeric",
                "min": feature_ranges["GrLivArea"]["min"],
                "max": feature_ranges["GrLivArea"]["max"],
                "step": 1,
                "unit": "sq ft",
                "description": (
                    "Above-ground living area of the house."
                )
            },
            {
                "name": "TotalBsmtSF",
                "label": "Total Basement Area",
                "type": "numeric",
                "min": feature_ranges["TotalBsmtSF"]["min"],
                "max": feature_ranges["TotalBsmtSF"]["max"],
                "step": 1,
                "unit": "sq ft",
                "description": (
                    "Total basement area of the house."
                )
            },
            {
                "name": "GarageCars",
                "label": "Garage Capacity",
                "type": "numeric",
                "min": feature_ranges["GarageCars"]["min"],
                "max": feature_ranges["GarageCars"]["max"],
                "step": 1,
                "unit": "cars",
                "description": (
                    "Number of cars the garage can accommodate."
                )
            },
            {
                "name": "YearBuilt",
                "label": "Year Built",
                "type": "numeric",
                "min": feature_ranges["YearBuilt"]["min"],
                "max": feature_ranges["YearBuilt"]["max"],
                "step": 1,
                "description": (
                    "Original construction year of the house."
                )
            }
        ]
    }


# ============================================================
# VALIDATE USER FEATURES
# ============================================================

def validate_user_features(user_features):
    """
    Validate the five values entered by the user.

    Returns a cleaned dictionary containing numeric values.
    """

    missing_features = [
        feature
        for feature in USER_FEATURES
        if feature not in user_features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features
            }
        )

    cleaned_features = {}

    for feature in USER_FEATURES:

        value = user_features.get(feature)

        # ----------------------------------------------------
        # Empty value
        # ----------------------------------------------------

        if value is None or value == "":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"{feature} is required."
                }
            )

        # ----------------------------------------------------
        # Convert to number
        # ----------------------------------------------------

        try:
            numeric_value = float(value)

        except (ValueError, TypeError):

            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        f"{feature} must be a numeric value."
                    )
                }
            )

        # ----------------------------------------------------
        # Check finite number
        # ----------------------------------------------------

        if not np.isfinite(numeric_value):

            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        f"{feature} must be a valid number."
                    )
                }
            )

        # ----------------------------------------------------
        # Range validation
        # ----------------------------------------------------

        minimum = feature_ranges[feature]["min"]

        maximum = feature_ranges[feature]["max"]

        if numeric_value < minimum or numeric_value > maximum:

            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        f"{feature} must be between "
                        f"{minimum:g} and {maximum:g}."
                    ),
                    "feature": feature,
                    "min": minimum,
                    "max": maximum,
                    "received": numeric_value
                }
            )

        # ----------------------------------------------------
        # Integer features
        # ----------------------------------------------------

        if feature in [
            "OverallQual",
            "GarageCars",
            "YearBuilt"
        ]:

            if not numeric_value.is_integer():

                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            f"{feature} must be a whole number."
                        )
                    }
                )

            numeric_value = int(numeric_value)

        else:

            numeric_value = float(numeric_value)

        cleaned_features[feature] = numeric_value

    return cleaned_features


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ====================================================
        # STEP 1 — GET USER INPUT
        # ====================================================

        user_features = request.features

        # ====================================================
        # STEP 2 — VALIDATE THE FIVE INPUTS
        # ====================================================

        cleaned_features = validate_user_features(
            user_features
        )

        # ====================================================
        # STEP 3 — BUILD COMPLETE MODEL INPUT
        # ====================================================
        #
        # The trained model originally expects all training
        # features.
        #
        # We create all of them here.
        #
        # The five user-selected features receive real values.
        #
        # Everything else becomes NaN.
        #
        # The existing preprocessing pipeline handles these
        # missing values using its imputers.
        #

        row = {}

        for feature in feature_columns:

            if feature in cleaned_features:

                row[feature] = cleaned_features[feature]

            else:

                row[feature] = np.nan

        # ====================================================
        # STEP 4 — CREATE DATAFRAME
        # ====================================================

        input_df = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        # ====================================================
        # STEP 5 — BASE MODEL PREDICTIONS
        # ====================================================

        elastic_pred = elastic_model.predict(
            input_df
        )[0]

        rf_pred = rf_model.predict(
            input_df
        )[0]

        gbr_pred = gbr_model.predict(
            input_df
        )[0]

        # ====================================================
        # STEP 6 — STACKING
        # ====================================================

        stack_input = pd.DataFrame({
            "ElasticNet": [elastic_pred],
            "RandomForest": [rf_pred],
            "GradientBoosting": [gbr_pred]
        })

        stack_log_prediction = stack_model.predict(
            stack_input
        )[0]

        # ====================================================
        # STEP 7 — CONVERT LOG PRICE TO DOLLARS
        # ====================================================

        predicted_price = np.expm1(
            stack_log_prediction
        )

        # ====================================================
        # STEP 8 — LOWER QUANTILE
        # ====================================================

        lower_log = lower_model.predict(
            input_df
        )[0]

        # ====================================================
        # STEP 9 — UPPER QUANTILE
        # ====================================================

        upper_log = upper_model.predict(
            input_df
        )[0]

        # ====================================================
        # STEP 10 — CALIBRATION
        # ====================================================

        calibrated_lower_log = (
            lower_log - calibration_factor
        )

        calibrated_upper_log = (
            upper_log + calibration_factor
        )

        # ====================================================
        # STEP 11 — CONVERT INTERVAL TO DOLLARS
        # ====================================================

        lower_price = np.expm1(
            calibrated_lower_log
        )

        upper_price = np.expm1(
            calibrated_upper_log
        )

        # ====================================================
        # STEP 12 — INTERVAL WIDTH
        # ====================================================

        interval_width = (
            upper_price - lower_price
        )

        # ====================================================
        # STEP 13 — SAFETY CHECK
        # ====================================================

        if lower_price < 0:
            lower_price = 0.0

        if predicted_price < 0:
            predicted_price = 0.0

        if upper_price < predicted_price:
            upper_price = predicted_price

        interval_width = (
            upper_price - lower_price
        )

        # ====================================================
        # STEP 14 — RESPONSE
        # ====================================================

        return {
            "predicted_price": round(
                float(predicted_price),
                2
            ),

            "lower_bound": round(
                float(lower_price),
                2
            ),

            "upper_bound": round(
                float(upper_price),
                2
            ),

            "interval_width": round(
                float(interval_width),
                2
            ),

            "inputs": {
                feature: cleaned_features[feature]
                for feature in USER_FEATURES
            }
        }

    # ========================================================
    # EXPECTED HTTP ERRORS
    # ========================================================

    except HTTPException:
        raise

    # ========================================================
    # UNEXPECTED ERRORS
    # ========================================================

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "error": str(e)
            }
        )

