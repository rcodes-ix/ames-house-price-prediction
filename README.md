# Ames House Price AI

A full-stack machine learning application that predicts house prices using the **Ames Housing dataset**.

The application combines multiple regression models, allows users to choose which trained model generates the prediction, displays model performance using **R²**, and provides a **90% prediction interval** alongside the estimated house price.

---

## Overview

**Ames House Price AI** turns a trained machine-learning regression pipeline into an interactive web application.

Users provide five important characteristics of a house:

* Overall quality
* Above-ground living area
* Basement area
* Garage capacity
* Year built

They can then select between four trained prediction approaches:

* **ElasticNet**
* **Random Forest**
* **Gradient Boosting**
* **Stacking Ensemble**

The selected model is actually used by the backend to generate the prediction.

The application also displays the **test R² performance** of each model so users can understand how the models compare.

---

## Features

### House Price Prediction

Predict an estimated sale price using five property characteristics:

| Feature       | Description                         |
| ------------- | ----------------------------------- |
| `OverallQual` | Overall material and finish quality |
| `GrLivArea`   | Above-ground living area            |
| `TotalBsmtSF` | Total basement area                 |
| `GarageCars`  | Garage capacity                     |
| `YearBuilt`   | Original construction year          |

---

### Multiple Prediction Models

The application supports four prediction modes:

#### ElasticNet

A regularized linear regression model that combines L1 and L2 regularization.

#### Random Forest

A nonlinear ensemble of randomized decision trees capable of modeling complex relationships between features.

#### Gradient Boosting

A sequential boosting algorithm that builds models iteratively to correct previous prediction errors.

#### Stacking Ensemble

Combines predictions from the three base models:

```text
                 ┌─────────────────┐
                 │   House Data    │
                 └────────┬────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        ElasticNet   Random Forest  Gradient Boosting
             │            │            │
             └────────────┼────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Stacking Model  │
                 └────────┬────────┘
                          ▼
                   Final Prediction
```

---

## Model Performance

The application displays the **test R² score** for every trained model.

R² measures how well a regression model explains variation in the target variable.

A higher R² generally indicates stronger predictive performance on the evaluation data.

The application dynamically displays the model metrics returned by the backend, allowing users to compare the models before making a prediction.

> The exact scores shown in the application are generated from the trained models and should be updated here if the models are retrained.

Example:

| Model             | Test R² |
| ----------------- | ------: |
| ElasticNet        | Dynamic |
| Random Forest     | Dynamic |
| Gradient Boosting | Dynamic |
| Stacking Ensemble | Dynamic |

---

## Prediction Intervals

The application does not only return a single predicted price.

It also provides a **90% prediction interval**:

```text
Lower Bound  ───────── Predicted Price ───────── Upper Bound
```

This gives the user a range around the estimated price rather than presenting the prediction as an exact value.

The backend calculates the lower and upper bounds using trained quantile models and a calibration factor.

---

## Machine Learning Pipeline

The project uses a preprocessing and ensemble-based regression architecture.

### Base Models

```text
House Features
      │
      ├──► ElasticNet
      │
      ├──► Random Forest
      │
      └──► Gradient Boosting
```

Their predictions are then used by the stacking model:

```text
ElasticNet prediction
        │
Random Forest prediction ──► Stacking Model ──► Final Price
        │
Gradient Boosting prediction
```

The final prediction is converted from the model's logarithmic target representation back into the original dollar scale.

---

## Tech Stack

### Machine Learning

* Python
* pandas
* NumPy
* scikit-learn
* Joblib
* LightGBM / quantile regression components
* Ames Housing dataset

### Backend

* FastAPI
* Pydantic
* Uvicorn
* Python

### Frontend

* Next.js
* React
* TypeScript
* CSS

### Development

* Git
* GitHub
* Jupyter Notebook

---

## Project Structure

```text
ames-house-price-prediction/
│
├── backend/
│   ├── app.py
│   ├── model.pkl
│   ├── data/
│   │   └── train.csv
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── globals.css
│   │   └── layout.tsx
│   │
│   ├── public/
│   ├── package.json
│   └── ...
│
├── notebooks/
│   └── ...
│
├── README.md
└── .gitignore
```

> Adjust the structure above if your actual repository folders have different names.

---

# Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/rcodes-ix/ames-house-price-prediction.git
cd ames-house-price-prediction
```

---

# Backend Setup

Move into the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure the trained model exists:

```text
backend/model.pkl
```

and the training data is available at:

```text
backend/data/train.csv
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will run at:

```text
http://localhost:8000
```

---

# Frontend Setup

Open another terminal and move into the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:3000
```

---

# 🔌 API

## `GET /`

Returns the API status.

---

## `GET /health`

Checks whether the backend and trained model are available.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## `GET /models`

Returns the available trained models and their evaluation metrics.

The frontend uses this endpoint to display model performance.

---

## `GET /features`

Returns information about the features accepted by the application, including their valid ranges.

---

## `POST /predict`

Generates a house-price prediction using the model selected by the user.

Example request:

```json
{
  "model": "random_forest",
  "features": {
    "OverallQual": 7,
    "GrLivArea": 2000,
    "TotalBsmtSF": 1000,
    "GarageCars": 2,
    "YearBuilt": 2005
  }
}
```

Example response:

```json
{
  "model": "random_forest",
  "model_name": "Random Forest",
  "model_r2": 0.0,
  "model_r2_percent": 0.0,
  "predicted_price": 250000,
  "lower_bound": 220000,
  "upper_bound": 285000,
  "interval_width": 65000
}
```

> The numeric values above are examples. The actual application returns values generated by the trained models.

---

# Why These Five Features?

The frontend intentionally keeps the user input simple.

Instead of requiring users to enter dozens of Ames Housing features, the application asks for five high-value property characteristics:

* Overall quality
* Living area
* Basement area
* Garage capacity
* Construction year

The backend constructs the model input and handles the remaining training features through the preprocessing pipeline.

This creates a much simpler interface while preserving the trained model architecture.

---

# Model Selection

One of the main features of the application is **real model selection**.

When the user selects a model, the frontend sends the selected model identifier to the backend.

For example:

```json
{
  "model": "elastic",
  "features": {
    "OverallQual": 7,
    "GrLivArea": 2000,
    "TotalBsmtSF": 1000,
    "GarageCars": 2,
    "YearBuilt": 2005
  }
}
```

The backend then runs the corresponding trained model rather than simply changing the label shown in the interface.

This allows users to compare predictions produced by different regression approaches.

---

# Evaluation Metric

The primary model-comparison metric displayed by the application is:

### R² — Coefficient of Determination

R² indicates how much of the variation in the target variable is explained by the regression model.

In general:

* **Closer to 1.0** → stronger fit
* **Closer to 0** → weaker explanatory performance
* **Negative values** → model performs worse than a simple baseline

The application displays R² as a percentage for easier interpretation.

For example:

```text
R² = 0.92

Displayed as:

92%
```

---

# Limitations

This application is a machine-learning demonstration and should not be treated as a professional property valuation system.

The prediction can be affected by:

* Limited user-provided features
* Dataset characteristics
* Historical housing-market patterns
* Differences between the training data and real-world properties
* Model assumptions and preprocessing
* Missing property characteristics

The prediction interval should also be interpreted as an estimate of uncertainty, not a guarantee of the property's actual selling price.

---

# Dataset

This project uses the **Ames Housing dataset**, a widely used dataset for regression and house-price prediction experiments.

The dataset contains detailed information about residential properties and their sale prices.

---

# Project Goal

The goal of this project was not simply to train a regression model.

It was to build an end-to-end machine-learning application that connects:

```text
Dataset
   ↓
Data preprocessing
   ↓
Model training
   ↓
Model evaluation
   ↓
Ensemble learning
   ↓
Prediction intervals
   ↓
FastAPI backend
   ↓
Next.js frontend
   ↓
Interactive prediction
```

This project demonstrates how a machine-learning model can be turned into an actual usable application rather than remaining inside a notebook.

---

## License

This project is available for educational and portfolio purposes.


