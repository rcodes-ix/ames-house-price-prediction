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


# ============================================================
# CHECK USER FEATURES
# ============================================================

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
# AVAILABLE MODELS
# ============================================================

AVAILABLE_MODELS = {
    "elastic": {
        "model": elastic_model,
        "name": "ElasticNet",
        "description": "Regularized linear regression model.",
    },
    "random_forest": {
        "model": rf_model,
        "name": "Random Forest",
        "description": "Ensemble of decision trees.",
    },
    "gradient_boosting": {
        "model": gbr_model,
        "name": "Gradient Boosting",
        "description": "Sequential tree-based boosting model.",
    },
    "stacking": {
        "model": stack_model,
        "name": "Stacking Ensemble",
        "description": "Combines predictions from all three base models.",
    },
}


# ============================================================
# FEATURE RANGES
# ============================================================

def load_feature_ranges():
    """
    Load the actual min/max values from the training dataset.
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

        ranges[feature] = {
            "min": float(series.min()),
            "max": float(series.max())
        }

    return ranges


feature_ranges = load_feature_ranges()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Ames House Price Prediction API",
    description=(
        "House price prediction using multiple trained "
        "regression models."
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

    # Model selected by the frontend.
    #
    # Possible values:
    #
    # elastic
    # random_forest
    # gradient_boosting
    # stacking

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
        "model_loaded": True,
        "available_models": list(AVAILABLE_MODELS.keys())
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
                "name": model_info["name"],
                "description": model_info["description"],
            }
            for model_id, model_info in AVAILABLE_MODELS.items()
        ]
    }


# ============================================================
# FEATURES ENDPOINT
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
                    "Original construction year."
                )
            }
        ]
    }


# ============================================================
# VALIDATE USER FEATURES
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

    integer_features = [
        "OverallQual",
        "GarageCars",
        "YearBuilt"
    ]

    for feature in USER_FEATURES:

        value = user_features.get(feature)

        # ----------------------------------------------------
        # EMPTY
        # ----------------------------------------------------

        if value is None or value == "":

            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"{feature} is required."
                }
            )

        # ----------------------------------------------------
        # CONVERT TO NUMBER
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
        # FINITE NUMBER
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
        # RANGE
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
        # INTEGER FEATURES
        # ----------------------------------------------------

        if feature in integer_features:

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
# BUILD MODEL INPUT
# ============================================================

def build_input_dataframe(cleaned_features):

    row = {}

    for feature in feature_columns:

        if feature in cleaned_features:

            row[feature] = cleaned_features[feature]

        else:

            row[feature] = np.nan

    return pd.DataFrame(
        [row],
        columns=feature_columns
    )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ====================================================
        # 1. VALIDATE MODEL
        # ====================================================

        selected_model = request.model.lower().strip()

        if selected_model not in AVAILABLE_MODELS:

            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unknown model selected.",
                    "selected_model": selected_model,
                    "available_models": list(
                        AVAILABLE_MODELS.keys()
                    )
                }
            )

        # ====================================================
        # 2. VALIDATE FEATURES
        # ====================================================

        cleaned_features = validate_user_features(
            request.features
        )

        # ====================================================
        # 3. BUILD DATAFRAME
        # ====================================================

        input_df = build_input_dataframe(
            cleaned_features
        )

        # ====================================================
        # 4. RUN SELECTED MODEL
        # ====================================================

        if selected_model == "stacking":

            # ------------------------------------------------
            # Run all three base models
            # ------------------------------------------------

            elastic_pred = elastic_model.predict(
                input_df
            )[0]

            rf_pred = rf_model.predict(
                input_df
            )[0]

            gbr_pred = gbr_model.predict(
                input_df
            )[0]

            # ------------------------------------------------
            # Feed their predictions into stack model
            # ------------------------------------------------

            stack_input = pd.DataFrame({
                "ElasticNet": [elastic_pred],
                "RandomForest": [rf_pred],
                "GradientBoosting": [gbr_pred]
            })

            log_prediction = stack_model.predict(
                stack_input
            )[0]

        else:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Actually run the model selected by the user.
            # ------------------------------------------------

            selected_estimator = AVAILABLE_MODELS[
                selected_model
            ]["model"]

            log_prediction = selected_estimator.predict(
                input_df
            )[0]

        # ====================================================
        # 5. LOG → DOLLARS
        # ====================================================

        predicted_price = np.expm1(
            log_prediction
        )

        # ====================================================
        # 6. PREDICTION INTERVAL
        # ====================================================

        lower_log = lower_model.predict(
            input_df
        )[0]

        upper_log = upper_model.predict(
            input_df
        )[0]

        # Apply calibration

        calibrated_lower_log = (
            lower_log - calibration_factor
        )

        calibrated_upper_log = (
            upper_log + calibration_factor
        )

        # Convert to dollars

        lower_price = np.expm1(
            calibrated_lower_log
        )

        upper_price = np.expm1(
            calibrated_upper_log
        )

        # ====================================================
        # 7. SAFETY
        # ====================================================

        predicted_price = max(
            0.0,
            float(predicted_price)
        )

        lower_price = max(
            0.0,
            float(lower_price)
        )

        upper_price = max(
            0.0,
            float(upper_price)
        )

        if upper_price < predicted_price:

            upper_price = predicted_price

        if lower_price > predicted_price:

            lower_price = predicted_price

        interval_width = (
            upper_price - lower_price
        )

        # ====================================================
        # 8. MODEL NAME
        # ====================================================

        model_name = AVAILABLE_MODELS[
            selected_model
        ]["name"]

        # ====================================================
        # 9. RESPONSE
        # ====================================================

        return {

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

            "model": selected_model,

            "model_name": model_name,

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