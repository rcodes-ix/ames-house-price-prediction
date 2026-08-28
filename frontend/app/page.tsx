
"use client";

import { useEffect, useMemo, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type FormData = {
  OverallQual: string;
  GrLivArea: string;
  TotalBsmtSF: string;
  GarageCars: string;
  YearBuilt: string;
};

type ModelId =
  | "elastic"
  | "random_forest"
  | "gradient_boosting"
  | "stacking";

type ModelInfo = {
  id: ModelId;
  name: string;
  short_name: string;
  description: string;
  r2: number;
  r2_percent: number;
};

type PredictionResult = {
  model: ModelId;
  model_name: string;
  model_r2: number;
  model_r2_percent: number;
  predicted_price: number;
  lower_bound: number;
  upper_bound: number;
  interval_width: number;
};

const RANGES = {
  OverallQual: {
    min: 1,
    max: 10,
    label: "Overall quality",
  },
  GrLivArea: {
    min: 334,
    max: 5642,
    label: "Above-ground living area",
  },
  TotalBsmtSF: {
    min: 0,
    max: 6110,
    label: "Basement area",
  },
  GarageCars: {
    min: 0,
    max: 5,
    label: "Garage capacity",
  },
  YearBuilt: {
    min: 1872,
    max: 2026,
    label: "Year built",
  },
};

const INITIAL_FORM: FormData = {
  OverallQual: "7",
  GrLivArea: "",
  TotalBsmtSF: "",
  GarageCars: "",
  YearBuilt: "",
};

const MODEL_OPTIONS: {
  id: ModelId;
  name: string;
  description: string;
}[] = [
  {
    id: "elastic",
    name: "ElasticNet",
    description: "Regularized linear model",
  },
  {
    id: "random_forest",
    name: "Random Forest",
    description: "Nonlinear tree ensemble",
  },
  {
    id: "gradient_boosting",
    name: "Gradient Boosting",
    description: "Sequential boosting model",
  },
  {
    id: "stacking",
    name: "Stacking Ensemble",
    description: "Combines all three models",
  },
];

const MODEL_METRIC_LABELS: Record<ModelId, string> = {
  elastic: "ElasticNet",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
  stacking: "Ensemble",
};

export default function Home() {
  const [formData, setFormData] =
    useState<FormData>(INITIAL_FORM);

  const [selectedModel, setSelectedModel] =
    useState<ModelId>("stacking");

  const [modelMetrics, setModelMetrics] =
    useState<Record<ModelId, ModelInfo> | null>(null);

  const [result, setResult] =
    useState<PredictionResult | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const qualityLabel = useMemo(() => {
    const quality = Number(formData.OverallQual);

    const labels: Record<number, string> = {
      1: "Very Poor",
      2: "Poor",
      3: "Fair",
      4: "Below Average",
      5: "Average",
      6: "Above Average",
      7: "Good",
      8: "Very Good",
      9: "Excellent",
      10: "Very Excellent",
    };

    return labels[quality] || "";
  }, [formData.OverallQual]);

  const selectedModelInfo =
    modelMetrics?.[selectedModel];

  const handleChange = (
    field: keyof FormData,
    value: string
  ) => {
    setFormData((previous) => ({
      ...previous,
      [field]: value,
    }));

    setError("");
    setResult(null);
  };

  const handleModelChange = (model: ModelId) => {
    setSelectedModel(model);
    setResult(null);
    setError("");
  };

  const isFieldOutOfRange = (
    field: keyof typeof RANGES
  ) => {
    const value = formData[field];

    if (value === "") {
      return false;
    }

    const number = Number(value);
    const range = RANGES[field];

    return (
      Number.isNaN(number) ||
      number < range.min ||
      number > range.max
    );
  };

  const hasInvalidFields =
    isFieldOutOfRange("OverallQual") ||
    isFieldOutOfRange("GrLivArea") ||
    isFieldOutOfRange("TotalBsmtSF") ||
    isFieldOutOfRange("GarageCars") ||
    isFieldOutOfRange("YearBuilt");

  const hasMissingFields =
    formData.OverallQual === "" ||
    formData.GrLivArea === "" ||
    formData.TotalBsmtSF === "" ||
    formData.GarageCars === "" ||
    formData.YearBuilt === "";

  const canPredict =
    !loading &&
    !hasMissingFields &&
    !hasInvalidFields;

  const formatPrice = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(value);
  };

  const loadModels = async () => {
    try {
      const response = await fetch(
        `${API_URL}/models`
      );

      if (!response.ok) {
        return;
      }

      const data = await response.json();

      if (Array.isArray(data.models)) {
        const mapped: Record<
          ModelId,
          ModelInfo
        > = {} as Record<ModelId, ModelInfo>;

        data.models.forEach(
          (model: ModelInfo) => {
            mapped[model.id] = model;
          }
        );

        setModelMetrics(mapped);
      }
    } catch {
      // Model metrics are supplementary.
      // Prediction can still work through /predict.
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    setError("");
    setResult(null);

    if (hasMissingFields) {
      setError(
        "Please complete all five property details."
      );
      return;
    }

    if (hasInvalidFields) {
      setError(
        "Please correct the values outside their allowed ranges."
      );
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: selectedModel,
            features: {
              OverallQual: Number(
                formData.OverallQual
              ),
              GrLivArea: Number(
                formData.GrLivArea
              ),
              TotalBsmtSF: Number(
                formData.TotalBsmtSF
              ),
              GarageCars: Number(
                formData.GarageCars
              ),
              YearBuilt: Number(
                formData.YearBuilt
              ),
            },
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message ||
                "Prediction failed. Please try again."
        );
      }

      setResult(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(
          "Something went wrong while connecting to the prediction API."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <div className="glow glowOne" />
      <div className="glow glowTwo" />

      <section className="container">
        <header className="hero">
          <div className="eyebrow">
            AMES HOUSE PRICE AI
          </div>

          <h1>
            What is your
            <span> house worth?</span>
          </h1>

          <p>
            Enter a few key characteristics of
            the property. Choose a trained machine
            learning model and estimate its sale
            price.
          </p>
        </header>

        <div className="grid">
          <section className="card formCard">
            <div className="sectionHeader">
              <div>
                <h2>Property details</h2>
                <p>
                  Only five details are required.
                </p>
              </div>

              <div className="requiredBadge">
                5 inputs
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              {/* OVERALL QUALITY */}

              <div className="field qualityField">
                <div className="fieldTop">
                  <label htmlFor="overallQual">
                    Overall quality
                  </label>

                  <strong>
                    {formData.OverallQual}/10
                  </strong>
                </div>

                <input
                  id="overallQual"
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={formData.OverallQual}
                  onChange={(event) =>
                    handleChange(
                      "OverallQual",
                      event.target.value
                    )
                  }
                />

                <div className="rangeLabels">
                  <span>1 — Poor</span>

                  <span>{qualityLabel}</span>

                  <span>10 — Excellent</span>
                </div>
              </div>

              {/* LIVING AREA */}

              <div className="field">
                <div className="fieldTop">
                  <label htmlFor="grLivArea">
                    Above-ground living area
                  </label>

                  <span className="unit">
                    sq ft
                  </span>
                </div>

                <input
                  id="grLivArea"
                  type="number"
                  min={RANGES.GrLivArea.min}
                  max={RANGES.GrLivArea.max}
                  step={1}
                  value={formData.GrLivArea}
                  onChange={(event) =>
                    handleChange(
                      "GrLivArea",
                      event.target.value
                    )
                  }
                  placeholder="e.g. 2000"
                  required
                  className={
                    isFieldOutOfRange("GrLivArea")
                      ? "invalid"
                      : ""
                  }
                />

                <div className="hint">
                  Range: 334–5,642 sq ft
                </div>

                {isFieldOutOfRange("GrLivArea") && (
                  <p className="errorText">
                    Living area must be between
                    334 and 5,642 sq ft.
                  </p>
                )}
              </div>

              {/* BASEMENT */}

              <div className="field">
                <div className="fieldTop">
                  <label htmlFor="totalBsmtSF">
                    Basement area
                  </label>

                  <span className="unit">
                    sq ft
                  </span>
                </div>

                <input
                  id="totalBsmtSF"
                  type="number"
                  min={RANGES.TotalBsmtSF.min}
                  max={RANGES.TotalBsmtSF.max}
                  step={1}
                  value={formData.TotalBsmtSF}
                  onChange={(event) =>
                    handleChange(
                      "TotalBsmtSF",
                      event.target.value
                    )
                  }
                  placeholder="e.g. 1000"
                  required
                  className={
                    isFieldOutOfRange(
                      "TotalBsmtSF"
                    )
                      ? "invalid"
                      : ""
                  }
                />

                <div className="hint">
                  Range: 0–6,110 sq ft
                </div>

                {isFieldOutOfRange(
                  "TotalBsmtSF"
                ) && (
                  <p className="errorText">
                    Basement area must be between
                    0 and 6,110 sq ft.
                  </p>
                )}
              </div>

              {/* GARAGE */}

              <div className="field">
                <div className="fieldTop">
                  <label htmlFor="garageCars">
                    Garage capacity
                  </label>

                  <span className="unit">
                    cars
                  </span>
                </div>

                <input
                  id="garageCars"
                  type="number"
                  min={RANGES.GarageCars.min}
                  max={RANGES.GarageCars.max}
                  step={1}
                  value={formData.GarageCars}
                  onChange={(event) =>
                    handleChange(
                      "GarageCars",
                      event.target.value
                    )
                  }
                  placeholder="e.g. 2"
                  required
                  className={
                    isFieldOutOfRange(
                      "GarageCars"
                    )
                      ? "invalid"
                      : ""
                  }
                />

                <div className="hint">
                  Range: 0–5 cars
                </div>

                {isFieldOutOfRange(
                  "GarageCars"
                ) && (
                  <p className="errorText">
                    Garage capacity must be
                    between 0 and 5 cars.
                  </p>
                )}
              </div>

              {/* YEAR BUILT */}

              <div className="field">
                <div className="fieldTop">
                  <label htmlFor="yearBuilt">
                    Year built
                  </label>

                  <span className="unit">
                    year
                  </span>
                </div>

                <input
                  id="yearBuilt"
                  type="number"
                  min={RANGES.YearBuilt.min}
                  max={RANGES.YearBuilt.max}
                  step={1}
                  value={formData.YearBuilt}
                  onChange={(event) =>
                    handleChange(
                      "YearBuilt",
                      event.target.value
                    )
                  }
                  placeholder="e.g. 2005"
                  required
                  className={
                    isFieldOutOfRange(
                      "YearBuilt"
                    )
                      ? "invalid"
                      : ""
                  }
                />

                <div className="hint">
                  Range: 1872–2026
                </div>

                {isFieldOutOfRange(
                  "YearBuilt"
                ) && (
                  <p className="errorText">
                    Year built must be between
                    1872 and 2026.
                  </p>
                )}
              </div>

              {/* MODEL */}

              <div className="field modelField">
                <div className="fieldTop">
                  <label>
                    Prediction model
                  </label>

                  <span className="unit">
                    trained model
                  </span>
                </div>

                <div className="modelOptions">
                  {MODEL_OPTIONS.map(
                    (model) => (
                      <button
                        key={model.id}
                        type="button"
                        className={
                          selectedModel ===
                          model.id
                            ? "modelOption selected"
                            : "modelOption"
                        }
                        onClick={() =>
                          handleModelChange(
                            model.id
                          )
                        }
                      >
                        <span>
                          {model.name}
                        </span>

                        <small>
                          {model.description}
                        </small>

                        {modelMetrics?.[
                          model.id
                        ] && (
                          <strong>
                            {modelMetrics[
                              model.id
                            ].r2_percent.toFixed(
                              2
                            )}
                            % R²
                          </strong>
                        )}
                      </button>
                    )
                  )}
                </div>
              </div>

              {error && (
                <div className="errorBox">
                  <span>!</span>
                  <p>{error}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={!canPredict}
                className="predictButton"
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Predicting...
                  </>
                ) : (
                  "Predict House Price"
                )}
              </button>
            </form>
          </section>

          {/* RESULT CARD */}

          <section className="card resultCard">
            <div className="resultLabel">
              ESTIMATED VALUE
            </div>

            {result ? (
              <>
                <div className="price">
                  {formatPrice(
                    result.predicted_price
                  )}
                </div>

                <p className="resultDescription">
                  Estimated sale price generated
                  by{" "}
                  <strong>
                    {result.model_name}
                  </strong>
                  .
                </p>

                <div className="intervalCard">
                  <div className="intervalHeader">
                    <div>
                      <span>
                        Prediction range
                      </span>

                      <strong>
                        90% prediction interval
                      </strong>
                    </div>
                  </div>

                  <div className="intervalValues">
                    <div>
                      <span>LOWER</span>

                      <strong>
                        {formatPrice(
                          result.lower_bound
                        )}
                      </strong>
                    </div>

                    <div className="intervalArrow">
                      →
                    </div>

                    <div className="right">
                      <span>UPPER</span>

                      <strong>
                        {formatPrice(
                          result.upper_bound
                        )}
                      </strong>
                    </div>
                  </div>

                  <div className="widthText">
                    Interval width:{" "}
                    <strong>
                      {formatPrice(
                        result.interval_width
                      )}
                    </strong>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="emptyPrice">
                  $
                </div>

                <h3>
                  Your estimate will appear here
                </h3>

                <p className="resultDescription">
                  Fill in the property details,
                  choose a model, and get an
                  estimated house price.
                </p>
              </>
            )}

            {/* MODEL PERFORMANCE */}

            <div className="performance">
              <div className="performanceHeader">
                <span>
                  MODEL PERFORMANCE
                </span>

                <span className="modelTag">
                  {selectedModelInfo?.short_name ||
                    MODEL_OPTIONS.find(
                      (m) =>
                        m.id === selectedModel
                    )?.name}
                </span>
              </div>

              <div className="modelSummary">
                {selectedModelInfo?.description ||
                  MODEL_OPTIONS.find(
                    (m) =>
                      m.id === selectedModel
                  )?.description}
              </div>

              <div className="stats">
                <div className="stat">
                  <strong>
                    {result
                      ? `${result.model_r2_percent.toFixed(
                          2
                        )}%`
                      : selectedModelInfo
                      ? `${selectedModelInfo.r2_percent.toFixed(
                          2
                        )}%`
                      : "—"}
                  </strong>

                  <span>Test R²</span>
                </div>

                <div className="stat">
                  <strong>
                    {
                      MODEL_METRIC_LABELS[
                        selectedModel
                      ]
                    }
                  </strong>

                  <span>Model</span>
                </div>

                <div className="stat">
                  <strong>
                    {selectedModel ===
                    "stacking"
                      ? "3"
                      : "1"}
                  </strong>

                  <span>Models used</span>
                </div>
              </div>

              <div className="modelComparison">
                <div className="comparisonTitle">
                  TEST R² COMPARISON
                </div>

                {MODEL_OPTIONS.map(
                  (model) => {
                    const metric =
                      modelMetrics?.[
                        model.id
                      ];

                    return (
                      <div
                        className={
                          selectedModel ===
                          model.id
                            ? "comparisonRow active"
                            : "comparisonRow"
                        }
                        key={model.id}
                      >
                        <span>
                          {model.name}
                        </span>

                        <strong>
                          {metric
                            ? `${metric.r2_percent.toFixed(
                                2
                              )}%`
                            : "—"}
                        </strong>
                      </div>
                    );
                  }
                )}
              </div>
            </div>
          </section>
        </div>

        <footer>
          <span>
            Ames House Price Prediction
          </span>

          <span>•</span>

          <span>
            Machine Learning Regression
          </span>
        </footer>
      </section>
    </main>
  );
}

