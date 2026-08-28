
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
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
# LOAD TRAINED MODELS
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

USER_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
]


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
    if not DATA_PATH.exists():
        raise RuntimeError(
            f"Training dataset not found at {DATA_PATH}."
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

        ranges[feature] = {
            "min": float(series.min()),
            "max": float(series.max())
        }

    # IMPORTANT:
    # The dataset's historical maximum is 2010,
    # but the UI should allow the current year.
    ranges["YearBuilt"]["max"] = 2026

    return ranges


feature_ranges = load_feature_ranges()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {
    "elastic": {
        "name": "ElasticNet",
        "short_name": "ElasticNet",
        "description": (
            "A regularized linear regression model. "
            "Strong as a simple, interpretable baseline."
        ),
    },
    "random_forest": {
        "name": "Random Forest",
        "short_name": "Random Forest",
        "description": (
            "An ensemble of decision trees that captures "
            "nonlinear relationships and feature interactions."
        ),
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "short_name": "Gradient Boosting",
        "description": (
            "A sequential tree-based model that focuses on "
            "reducing errors made by previous trees."
        ),
    },
    "stacking": {
        "name": "Stacking Ensemble",
        "short_name": "Stacking",
        "description": (
            "Combines ElasticNet, Random Forest, and "
            "Gradient Boosting through a trained meta-model."
        ),
    },
}


# ============================================================
# CALCULATE REAL TEST R² SCORES
# ============================================================

def calculate_model_metrics():
    """
    Recreate the same 70/15/15 split used during training
    and evaluate the already-trained models on the held-out
    test set.

    This gives the frontend real R² values instead of
    invented percentages.
    """

    if not DATA_PATH.exists():
        raise RuntimeError(
            f"Training dataset not found at {DATA_PATH}."
        )

    df = pd.read_csv(DATA_PATH)

    if "SalePrice" not in df.columns:
        raise RuntimeError(
            "SalePrice column was not found in train.csv."
        )

    # Same feature preparation used by the project.
    X = df.drop(
        columns=["SalePrice", "Id"],
        errors="ignore"
    )

    y_log = np.log1p(df["SalePrice"])

    # Keep exactly the features expected by the trained models.
    X = X[feature_columns]

    # Same split used in the project.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_log,
        test_size=0.30,
        random_state=42
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42
    )

    # Base models
    elastic_pred = elastic_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    gbr_pred = gbr_model.predict(X_test)

    # Stacking model
    stack_input = pd.DataFrame({
        "ElasticNet": elastic_pred,
        "RandomForest": rf_pred,
        "GradientBoosting": gbr_pred,
    })

    stack_pred = stack_model.predict(stack_input)

    return {
        "elastic": float(
            r2_score(y_test, elastic_pred)
        ),
        "random_forest": float(
            r2_score(y_test, rf_pred)
        ),
        "gradient_boosting": float(
            r2_score(y_test, gbr_pred)
        ),
        "stacking": float(
            r2_score(y_test, stack_pred)
        ),
    }


try:
    MODEL_METRICS = calculate_model_metrics()
except Exception as e:
    raise RuntimeError(
        f"Could not calculate model metrics: {e}"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Ames House Price Prediction API",
    description=(
        "House price prediction using multiple trained "
        "regression models and a stacking ensemble."
    ),
    version="2.0.0"
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
    model: str = "stacking"


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
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# FEATURES
# ============================================================

@app.get("/features")
def get_features():

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
                    "Above-ground living area."
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
                    "Total basement area."
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
                "min": 1872,
                "max": 2026,
                "step": 1,
                "description": (
                    "Original construction year."
                )
            }
        ]
    }


# ============================================================
# MODELS ENDPOINT
# ============================================================

@app.get("/models")
def get_models():

    return {
        "models": [
            {
                "id": model_id,
                **MODEL_INFO[model_id],
                "r2": round(
                    MODEL_METRICS[model_id],
                    4
                ),
                "r2_percent": round(
                    MODEL_METRICS[model_id] * 100,
                    2
                )
            }
            for model_id in MODEL_INFO
        ]
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_user_features(user_features):

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

        if value is None or value == "":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"{feature} is required."
                }
            )

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

        if not np.isfinite(numeric_value):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        f"{feature} must be a valid number."
                    )
                }
            )

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
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ----------------------------------------------------
        # Validate selected model
        # ----------------------------------------------------

        selected_model = request.model

        if selected_model not in MODEL_INFO:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid model selected.",
                    "allowed_models": list(
                        MODEL_INFO.keys()
                    )
                }
            )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        cleaned_features = validate_user_features(
            request.features
        )

        # ----------------------------------------------------
        # Build complete dataframe
        # ----------------------------------------------------

        row = {}

        for feature in feature_columns:

            if feature in cleaned_features:
                row[feature] = cleaned_features[feature]
            else:
                row[feature] = np.nan

        input_df = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        # ----------------------------------------------------
        # Run all three base models
        #
        # This is important because the stacking model needs
        # all three predictions.
        # ----------------------------------------------------

        elastic_pred = elastic_model.predict(
            input_df
        )[0]

        rf_pred = rf_model.predict(
            input_df
        )[0]

        gbr_pred = gbr_model.predict(
            input_df
        )[0]

        # ----------------------------------------------------
        # Calculate stacking prediction
        # ----------------------------------------------------

        stack_input = pd.DataFrame({
            "ElasticNet": [elastic_pred],
            "RandomForest": [rf_pred],
            "GradientBoosting": [gbr_pred]
        })

        stack_pred = stack_model.predict(
            stack_input
        )[0]

        # ----------------------------------------------------
        # Select the REAL prediction from selected model
        # ----------------------------------------------------

        if selected_model == "elastic":

            selected_log_prediction = elastic_pred

        elif selected_model == "random_forest":

            selected_log_prediction = rf_pred

        elif selected_model == "gradient_boosting":

            selected_log_prediction = gbr_pred

        else:

            selected_log_prediction = stack_pred

        # ----------------------------------------------------
        # Convert log prediction to dollars
        # ----------------------------------------------------

        predicted_price = np.expm1(
            selected_log_prediction
        )

        # ----------------------------------------------------
        # Prediction interval
        #
        # The calibrated quantile models belong to the
        # project's uncertainty pipeline.
        # ----------------------------------------------------

        lower_log = lower_model.predict(
            input_df
        )[0]

        upper_log = upper_model.predict(
            input_df
        )[0]

        calibrated_lower_log = (
            lower_log - calibration_factor
        )

        calibrated_upper_log = (
            upper_log + calibration_factor
        )

        lower_price = np.expm1(
            calibrated_lower_log
        )

        upper_price = np.expm1(
            calibrated_upper_log
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        predicted_price = max(
            0.0,
            float(predicted_price)
        )

        lower_price = max(
            0.0,
            float(lower_price)
        )

        upper_price = max(
            predicted_price,
            float(upper_price)
        )

        interval_width = (
            upper_price - lower_price
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "model": selected_model,
            "model_name": MODEL_INFO[
                selected_model
            ]["name"],

            "model_r2": round(
                MODEL_METRICS[selected_model],
                4
            ),

            "model_r2_percent": round(
                MODEL_METRICS[selected_model] * 100,
                2
            ),

            "predicted_price": round(
                predicted_price,
                2
            ),

            "lower_bound": round(
                lower_price,
                2
            ),

            "upper_bound": round(
                upper_price,
                2
            ),

            "interval_width": round(
                interval_width,
                2
            ),

            "inputs": {
                feature: cleaned_features[feature]
                for feature in USER_FEATURES
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "error": str(e)
            }
        )

