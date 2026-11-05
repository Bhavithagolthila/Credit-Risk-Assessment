from __future__ import annotations

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
STATIC_DIR = BASE_DIR / "static"

FEATURE_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("credit-ledger")


class LoanApplication(BaseModel):
    """Applicant fields in the exact schema used to train the saved pipeline."""

    model_config = ConfigDict(extra="forbid")

    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, le=6_000_000)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(ge=500, le=35_000)
    loan_int_rate: float = Field(ge=5.42, le=23.22)
    loan_percent_income: float = Field(ge=0, le=0.83)
    cb_person_default_on_file: Literal["N", "Y"]
    cb_person_cred_hist_length: int = Field(ge=2, le=30)

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.person_emp_length > self.person_age:
            raise ValueError("Employment length cannot exceed applicant age.")
        expected_ratio = self.loan_amnt / self.person_income
        if abs(self.loan_percent_income - expected_ratio) > 0.015:
            raise ValueError("Loan-to-income ratio does not match loan amount and annual income.")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise RuntimeError("Model artifacts are missing. Keep both .pkl files beside main.py.")
    try:
        app.state.model = joblib.load(MODEL_PATH)
        with THRESHOLD_PATH.open("rb") as file:
            app.state.threshold = float(pickle.load(file))
    except Exception as exc:
        logger.exception("Could not load the trained model artifacts")
        raise RuntimeError(
            "Could not load model artifacts. Install the versions in requirements.txt; "
            "the model was serialized with scikit-learn 1.6.1."
        ) from exc

    if not 0.0 < app.state.threshold < 1.0:
        raise RuntimeError("The saved classification threshold must be between 0 and 1.")
    logger.info("Credit model loaded; decision threshold=%.4f", app.state.threshold)
    yield
    for attr in ("model", "threshold"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


app = FastAPI(
    title="Credit Ledger API",
    description="Explainable credit-risk assessment demo powered by a calibrated XGBoost pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_global_feature_importance(model) -> list[dict]:
    """Average XGBoost global feature importance across calibration folds.

    This is a global model-level summary, not a causal or applicant-specific explanation.
    """
    totals: dict[str, float] = {}
    count = 0
    calibrated = getattr(model, "calibrated_classifiers_", [])
    for item in calibrated:
        estimator = getattr(item, "estimator", None)
        if estimator is None and hasattr(item, "base_estimator"):
            estimator = item.base_estimator
        if estimator is None or not hasattr(estimator, "named_steps"):
            continue
        pipeline = estimator.named_steps
        classifier = pipeline.get("classifier")
        preprocessor = pipeline.get("preprocessor")
        if classifier is None or preprocessor is None or not hasattr(classifier, "feature_importances_"):
            continue
        try:
            names = preprocessor.get_feature_names_out()
            values = classifier.feature_importances_
        except Exception:
            continue
        for name, value in zip(names, values):
            clean = str(name).replace("num__", "").replace("cat__", "")
            base = next((col for col in FEATURE_COLUMNS if clean == col or clean.startswith(col + "_")), clean)
            totals[base] = totals.get(base, 0.0) + float(value)
        count += 1
    if not count or not totals:
        return []
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    grand_total = sum(value for _, value in ranked) or 1.0
    return [
        {"feature": name.replace("person_", "").replace("cb_", "credit ").replace("_", " ").title(),
         "importance": round(value / grand_total, 4)}
        for name, value in ranked[:6]
    ]


@app.get("/health")
def health():
    loaded = hasattr(app.state, "model") and hasattr(app.state, "threshold")
    return {"status": "ok" if loaded else "starting", "model_loaded": loaded}


@app.get("/model-info")
def model_info():
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not ready.")
    return {
        "model_name": "Calibrated XGBoost",
        "threshold": round(float(app.state.threshold), 4),
        "features": FEATURE_COLUMNS,
        "global_feature_importance": get_global_feature_importance(app.state.model),
        "explanation_note": "Feature importance is a global summary across model folds. It does not explain causation or an individual applicant's outcome.",
    }


@app.post("/predict")
def predict(application: LoanApplication):
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not ready.")
    try:
        row = application.model_dump()
        input_df = pd.DataFrame([[row[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        probabilities = app.state.model.predict_proba(input_df)[0]
        classes = list(app.state.model.classes_)
        positive_index = classes.index(1) if 1 in classes else 1
        default_probability = float(probabilities[positive_index])
        repayment_probability = 1.0 - default_probability
        threshold = float(app.state.threshold)
        high_risk = default_probability >= threshold
        return {
            "default_probability": round(default_probability, 6),
            "repayment_probability": round(repayment_probability, 6),
            "threshold": round(threshold, 6),
            "default_prediction": int(high_risk),
            "risk_level": "Higher risk" if high_risk else "Lower risk",
            "result": "HIGHER RISK" if high_risk else "LOWER RISK",
            "model_name": "Calibrated XGBoost",
            "disclaimer": "This is an educational machine-learning estimate, not a credit decision, financial advice, or a guarantee of repayment.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed. Check the model environment and input values.") from exc


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
