"use client";

import { useMemo, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ModelType = "stacking" | "elastic" | "random_forest" | "gradient_boosting";

type FormData = {
  OverallQual: string;
  GrLivArea: string;
  TotalBsmtSF: string;
  GarageCars: string;
  YearBuilt: string;
};

type PredictionResult = {
  predicted_price: number;
  lower_bound: number;
  upper_bound: number;
  interval_width: number;
  model?: string;
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
  value: ModelType;
  title: string;
  description: string;
}[] = [
  {
    value: "stacking",
    title: "Stacking Ensemble",
    description: "Combines the three trained models.",
  },
  {
    value: "elastic",
    title: "ElasticNet",
    description: "Regularized linear regression model.",
  },
  {
    value: "random_forest",
    title: "Random Forest",
    description: "Tree-based ensemble model.",
  },
  {
    value: "gradient_boosting",
    title: "Gradient Boosting",
    description: "Sequential boosting model.",
  },
];

export default function Home() {
  const [formData, setFormData] =
    useState<FormData>(INITIAL_FORM);

  const [selectedModel, setSelectedModel] =
    useState<ModelType>("stacking");

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

  const selectedModelInfo = MODEL_OPTIONS.find(
    (model) => model.value === selectedModel
  );

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

  const handleModelChange = (model: ModelType) => {
    setSelectedModel(model);
    setError("");
    setResult(null);
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
    isFieldOutOfRange("GrLivArea") ||
    isFieldOutOfRange("TotalBsmtSF") ||
    isFieldOutOfRange("GarageCars") ||
    isFieldOutOfRange("YearBuilt");

  const hasMissingFields =
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
      /*
       * The backend receives:
       *
       * 1. The five user-selected property features
       * 2. The selected trained model
       *
       * The backend is responsible for actually running
       * that model and returning its prediction.
       */
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
            : "Prediction failed. Please try again."
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
    <>
      <style jsx global>{`
        * {
          box-sizing: border-box;
        }

        html {
          scroll-behavior: smooth;
        }

        body {
          margin: 0;
          background: #050505;
          color: #f5f5f5;
          font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
        }

        button,
        input {
          font: inherit;
        }

        button {
          cursor: pointer;
        }

        /*
         * Keeps the prediction visible while the user
         * scrolls through the property form.
         */
        .resultSticky {
          position: sticky;
          top: 24px;
          align-self: start;
        }

        @media (max-width: 800px) {
          .resultSticky {
            position: static;
          }
        }
      `}</style>

      <main className="page">
        <div className="glow glowOne" />
        <div className="glow glowTwo" />

        <section className="container">

          {/* HEADER */}

          <header className="hero">
            <div className="eyebrow">
              AMES HOUSE PRICE AI
            </div>

            <h1>
              What is your
              <span> house worth?</span>
            </h1>

            <p>
              Enter a few key characteristics of the
              property. Choose a trained machine-learning
              model and get an estimated sale price with
              a prediction range.
            </p>
          </header>

          <div className="grid">

            {/* FORM CARD */}

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

                {/* MODEL SELECTOR */}

                <div className="field qualityField">

                  <div className="fieldTop">
                    <label>
                      Prediction model
                    </label>

                    <span className="unit">
                      choose one
                    </span>
                  </div>

                  <div className="modelOptions">

                    {MODEL_OPTIONS.map((model) => (
                      <button
                        key={model.value}
                        type="button"
                        className={
                          selectedModel === model.value
                            ? "modelOption active"
                            : "modelOption"
                        }
                        onClick={() =>
                          handleModelChange(model.value)
                        }
                      >
                        <span className="modelOptionTitle">
                          {model.title}
                        </span>

                        <span className="modelOptionDescription">
                          {model.description}
                        </span>
                      </button>
                    ))}

                  </div>

                  <div className="hint">
                    Selected:{" "}
                    <strong>
                      {selectedModelInfo?.title}
                    </strong>
                  </div>
                </div>

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
                    <span>
                      1 — Poor
                    </span>

                    <span>
                      {qualityLabel}
                    </span>

                    <span>
                      10 — Excellent
                    </span>
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
                      isFieldOutOfRange(
                        "GrLivArea"
                      )
                        ? "invalid"
                        : ""
                    }
                  />

                  <div className="hint">
                    Range: 334–5,642 sq ft
                  </div>

                  {isFieldOutOfRange(
                    "GrLivArea"
                  ) && (
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
                      Garage capacity must be between
                      0 and 5 cars.
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

                {/* GENERAL ERROR */}

                {error && (
                  <div className="errorBox">
                    <span>!</span>
                    <p>{error}</p>
                  </div>
                )}

                {/* PREDICT BUTTON */}

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

            <section
              className="card resultCard resultSticky"
            >

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
                    Estimated sale price using{" "}
                    <strong>
                      {result.model ||
                        selectedModelInfo?.title}
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
                          90% confidence interval
                        </strong>

                      </div>

                    </div>

                    <div className="intervalValues">

                      <div>

                        <span>
                          LOWER
                        </span>

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

                        <span>
                          UPPER
                        </span>

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
                    choose a model, and let the
                    trained model estimate the
                    home's potential sale price.
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
                    {selectedModelInfo?.title}
                  </span>

                </div>

                <div className="stats">

                  <div className="stat">

                    <strong>
                      91.98%
                    </strong>

                    <span>
                      Test R²
                    </span>

                  </div>

                  <div className="stat">

                    <strong>
                      $14.3K
                    </strong>

                    <span>
                      Test MAE
                    </span>

                  </div>

                  <div className="stat">

                    <strong>
                      86.76%
                    </strong>

                    <span>
                      Interval coverage
                    </span>

                  </div>

                </div>

              </div>

            </section>

          </div>

          <footer>

            <span>
              Ames House Price Prediction
            </span>

            <span>
              •
            </span>

            <span>
              {selectedModelInfo?.title}
            </span>

          </footer>

        </section>

      </main>
    </>
  );
}