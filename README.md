# Ames Housing Price Regression with Prediction Intervals

A production-oriented house price regression project built around the Ames Housing dataset.

The project focuses not only on predicting house prices, but on building a **leak-proof machine learning pipeline** and producing **prediction intervals** that communicate uncertainty around each prediction.

---

## Project Overview

House price prediction is often presented as a simple regression problem: given information about a house, predict its sale price.

This project goes further.

The goal is to build a robust regression system that:

- Handles numerical and categorical features safely
- Prevents data leakage during preprocessing
- Uses target encoding for categorical variables
- Trains multiple base regression models
- Generates honest out-of-fold predictions
- Combines base models using a stacked ensemble
- Uses quantile regression to estimate prediction intervals
- Evaluates predictions using RMSE on log-transformed sale prices
- Produces both point estimates and uncertainty intervals

The project uses the **Ames Housing dataset** from Kaggle's House Prices: Advanced Regression Techniques competition.

---

## Problem Statement

Given a set of characteristics describing a residential property, predict its sale price while also estimating the uncertainty associated with the prediction.

Instead of returning only:

> Predicted price: $250,000

the final system aims to produce something conceptually like:

> Predicted price: $250,000  
> Prediction interval: $220,000 – $285,000

This makes the output more useful for decision-making because it communicates not only the expected value, but also the range of plausible outcomes.

---

## Objectives

The main objectives of this project are:

1. Build a leak-proof preprocessing pipeline.
2. Handle missing numerical and categorical values inside the pipeline.
3. Apply target encoding to categorical features.
4. Train multiple regression models.
5. Generate out-of-fold predictions without leakage.
6. Build a stacked regression ensemble.
7. Train quantile regression models.
8. Produce prediction intervals rather than only point predictions.
9. Evaluate the final system using appropriate regression metrics.
10. Document the complete machine learning workflow.

---

## Dataset

The project uses the Ames Housing dataset from Kaggle:

**House Prices: Advanced Regression Techniques**

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The Kaggle training dataset contains:

- 1,460 houses
- 79 explanatory features
- 1 target variable: `SalePrice`

The features contain a mixture of numerical and categorical variables describing characteristics such as:

- Overall quality
- Living area
- Neighborhood
- Year built
- Garage characteristics
- Basement characteristics
- Kitchen quality
- Lot characteristics
- And many other property attributes

The raw dataset is not included in this repository.

---

## Machine Learning Approach

### 1. Data Preparation

The raw training data is separated into:

- Features `X`
- Target `SalePrice`

The `Id` column is removed because it is an identifier rather than a meaningful property characteristic.

### 2. Target Transformation

House prices are strongly right-skewed.

The target is transformed using:

```python
np.log1p(SalePrice)
```

Models are trained on the transformed target.

Predictions can be converted back to the original price scale using:

```python
np.expm1(prediction)
```

The primary evaluation metric is RMSE on the log-transformed target.

### 3. Leak-Proof Preprocessing

Preprocessing is implemented using scikit-learn `Pipeline` and `ColumnTransformer`.

Numerical features use:

- Median imputation
- Standard scaling

Categorical features use:

- Missing-value handling
- Target encoding

The preprocessing steps are fitted only on the appropriate training data during model training and cross-validation.

This is important because preprocessing the entire dataset before cross-validation can introduce information leakage.

### 4. Baseline Model

An ElasticNet regression model is used as an initial baseline.

The baseline establishes a reference point against which more complex models can be compared.

### 5. Base Models

Multiple regression models will be trained and evaluated.

The base models are intended to capture different relationships within the data.

### 6. Out-of-Fold Predictions

The stacking system will use out-of-fold predictions.

Instead of training a meta-model on predictions generated from data that the base model has already seen, each training observation receives a prediction from a model that did not train on that observation.

This creates honest meta-features for the stacking model.

### 7. Stacked Ensemble

The out-of-fold predictions from the base models are used as inputs to a meta-model.

Conceptually:

```text
                    Training Data
                         |
          +--------------+--------------+
          |              |              |
      Base Model 1   Base Model 2   Base Model 3
          |              |              |
          +--------------+--------------+
                         |
                OOF Predictions
                         |
                         v
                  Meta Model
                         |
                         v
                 Final Prediction
```

### 8. Quantile Regression

Quantile regression is used to estimate prediction intervals.

The project will train models for different quantiles, such as:

- Lower quantile
- Median
- Upper quantile

For example:

```text
Lower Quantile ───────────────┐
                              |
Median Prediction ────────────┼──> Prediction output
                              |
Upper Quantile ───────────────┘
```

The final system can therefore provide both:

- A point estimate
- An uncertainty interval

### 9. Prediction Intervals

The prediction interval will be evaluated using measures such as:

- Coverage
- Interval width
- Calibration

The goal is not simply to create a wide interval that contains almost every house price.

A useful interval should provide a reasonable balance between:

**Coverage**

and

**Sharpness / interval width**

---

## Evaluation

The primary regression metric is:

### RMSE on log-transformed SalePrice

Additional metrics and diagnostics will be used to understand model performance.

For prediction intervals, the project will evaluate:

- Prediction interval coverage
- Average interval width
- Quantile behavior
- Calibration

---

## Project Structure

```text
ames-housing-price-regression/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── house-price-prediction.ipynb
│
├── README.md
│
└── .gitignore
```

The dataset is intentionally excluded from version control.

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Category Encoders
- LightGBM
- Jupyter Notebook
- Git
- GitHub

---

## Learning Goals

This project is being developed as a hands-on machine learning engineering exercise.

The main concepts being practiced are:

- Regression
- Feature preprocessing
- Pipeline design
- ColumnTransformer
- Target encoding
- Cross-validation
- Out-of-fold predictions
- Ensemble learning
- Stacking
- Quantile regression
- Prediction intervals
- Model evaluation
- Data leakage prevention
- Git and GitHub workflow

---

## Lessons Learned

This section will be updated throughout development.

### Data Leakage

Preprocessing operations that learn from the target or feature distribution must be fitted only on the appropriate training data.

### Target Transformation

The strong skew in raw house prices can be substantially reduced using a logarithmic transformation.

### Pipeline Discipline

Keeping preprocessing and modeling inside a single pipeline makes it easier to ensure that transformations are consistently applied and prevents accidental leakage during cross-validation.



