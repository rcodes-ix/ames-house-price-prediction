# Ames House Price Prediction

A machine learning regression pipeline for predicting residential house prices using the Ames Housing dataset.

The project goes beyond a single regression model by combining **leak-resistant preprocessing, multiple regression algorithms, out-of-fold stacking, and quantile regression for prediction intervals**.

The final stacking ensemble achieved an **R² of 0.9198** on the held-out test set.

---

## Overview

Predicting house prices is a regression problem where the target variable can be highly skewed and the dataset contains both numerical and categorical features with missing values.

This project builds a complete end-to-end regression workflow that:

* Handles missing numerical and categorical values
* Encodes categorical variables using target encoding
* Standardizes numerical features
* Applies a log transformation to the target
* Trains and compares multiple regression models
* Generates out-of-fold predictions
* Combines models using a stacking ensemble
* Estimates 90% prediction intervals using quantile regression
* Calibrates the prediction intervals using validation data
* Evaluates the final system on an untouched test set

The goal is not only to predict a house's price, but also to estimate **how uncertain that prediction is**.

---

## Project Goals

The project focuses on three main objectives:

1. Build a reliable house-price regression pipeline.
2. Improve predictive performance through ensemble learning.
3. Provide prediction intervals instead of only point estimates.

This makes the system more informative than a model that simply outputs:

> Predicted price: $250,000

Instead, the final system can provide a prediction together with an estimated range of plausible prices.

---

## Machine Learning Approach

### 1. Data Preparation

The Ames Housing dataset contains approximately 2,900 residential properties and a mixture of numerical and categorical features.

The `Id` column is removed because it is an identifier rather than a meaningful predictive feature.

The target variable is transformed using:

```python
y_log = np.log1p(y)
```

This reduces the strong right skew in the original `SalePrice` distribution and allows the models to work with a more stable target distribution.

---

### 2. Train / Validation / Test Split

The dataset is divided into:

* **70% training**
* **15% validation**
* **15% test**

The validation set is used during model development and calibration.

The test set remains untouched until the final evaluation.

---

### 3. Preprocessing

The preprocessing pipeline uses `ColumnTransformer` and separate transformations for numerical and categorical features.

#### Numerical features

```text
Median imputation
        ↓
StandardScaler
```

#### Categorical features

```text
Missing-value imputation
        ↓
Target encoding
```

The preprocessing steps are contained inside scikit-learn pipelines so that transformations are learned from the appropriate training data rather than manually applied beforehand.

---

## Models

Three base regression models were trained:

### ElasticNet

Used as the regularized linear baseline.

```text
ElasticNet
alpha = 0.0005
l1_ratio = 0.5
```

### Random Forest

An ensemble of decision trees used to capture nonlinear relationships.

```text
n_estimators = 500
max_features = sqrt
```

### Gradient Boosting

A sequential boosting model designed to capture complex nonlinear relationships.

```text
n_estimators = 500
learning_rate = 0.03
max_depth = 3
```

---

## Stacking Ensemble

Instead of selecting only one base model, the project combines their predictions.

First, **5-fold out-of-fold predictions** are generated for:

* ElasticNet
* Random Forest
* Gradient Boosting

These predictions become the input features for a second-level linear regression model.

```text
ElasticNet ───────┐
                  │
Random Forest ────┼──→ Linear Regression → Final Prediction
                  │
Gradient Boosting ┘
```

This allows the final model to learn how to combine the strengths of the individual models.

---

## Final Test Performance

The final stacking ensemble was evaluated on the untouched test set.

| Metric                 |         Result |
| ---------------------- | -------------: |
| **R²**                 |     **0.9198** |
| **Variance explained** |     **91.98%** |
| **RMSE**               | **$24,473.46** |
| **MAE**                | **$14,282.78** |

### What does this mean?

The model explains approximately **91.98% of the variance in the log-transformed house prices** on the held-out test set.

The MAE of approximately **$14.3K** means that the model's average absolute prediction error was about $14,283 on the original price scale.

---

## Prediction Intervals

A point prediction alone does not communicate uncertainty.

To address this, the project trains two additional Gradient Boosting models using quantile regression:

```text
5th percentile  → Lower bound
95th percentile → Upper bound
```

Together these form a **nominal 90% prediction interval**.

### Calibration

The original interval achieved:

```text
Validation coverage: 82.65%
```

A calibration step expanded the interval based on validation residuals:

```text
Original coverage:
82.65%

Calibrated validation coverage:
89.95%
```

The calibrated interval was then evaluated on the untouched test set.

### Final test interval results

| Metric                  |         Result |
| ----------------------- | -------------: |
| Nominal coverage        |        **90%** |
| Empirical test coverage |     **86.76%** |
| Mean interval width     | **$77,454.50** |
| Median interval width   | **$63,671.09** |

The **86.76% test coverage** is the empirical coverage observed on the held-out test set. It should not be interpreted as a guarantee that 90% of future predictions will fall inside the interval.

---

## Tech Stack

* Python
* NumPy
* pandas
* scikit-learn
* category_encoders
* Matplotlib
* Jupyter Notebook

### Key techniques

* Log transformation
* Missing-value imputation
* Standardization
* Target encoding
* ElasticNet regression
* Random Forest regression
* Gradient Boosting regression
* K-fold cross-validation
* Out-of-fold predictions
* Stacking ensemble
* Quantile regression
* Prediction interval calibration
* RMSE
* MAE
* R²

---

## Project Structure

```text
ames-house-price-prediction/
│
├── data/
│   └── train.csv
│
├── house_price_prediction.ipynb
│
├── README.md
│
└── ...
```

> The dataset is not included in this repository if it is subject to the original dataset's distribution terms. Download it separately and place the required file inside the `data/` directory.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rcodes-ix/ames-house-price-prediction.git
cd ames-house-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv myenv
```

Activate it on Linux/macOS:

```bash
source myenv/bin/activate
```

On Windows:

```bash
myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn category_encoders jupyter
```

### 4. Add the dataset

Place the training dataset at:

```text
data/train.csv
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open:

```text
house_price_prediction.ipynb
```

and run the cells from top to bottom.

---

## Evaluation Philosophy

A major focus of this project is avoiding misleading evaluation.

The workflow separates:

```text
Training data
      ↓
Model training
      ↓
Validation data
      ↓
Model development + calibration
      ↓
Untouched test data
      ↓
Final evaluation
```

The test set is not used to choose the final model or calibration factor.

This provides a more honest estimate of how the final system performs on unseen data.

---

## Limitations

This project has several limitations:

* The dataset represents a specific housing market and may not generalize to other regions.
* The prediction intervals are empirically calibrated rather than guaranteed to provide exact 90% coverage.
* The final test coverage was **86.76%**, below the nominal 90% level.
* The model predicts based on historical dataset features and cannot account for information unavailable in the dataset.
* The current project is a research/learning implementation rather than a production real-estate valuation system.

---

## Future Improvements

Potential improvements include:

* Hyperparameter optimization with cross-validation
* More advanced gradient-boosting models
* Feature engineering based on housing domain knowledge
* Better uncertainty calibration
* Cross-validation-based model comparison
* Error analysis by house-price range
* Explainability using SHAP
* A prediction API using FastAPI
* A web interface for interactive predictions
* Model monitoring and retraining pipelines

---

## Dataset

This project uses the **Ames Housing dataset**, commonly used for regression and machine learning experimentation.

The project focuses on building the modeling pipeline rather than redistributing the original dataset.

---

## License

See the repository's license file for licensing information.
