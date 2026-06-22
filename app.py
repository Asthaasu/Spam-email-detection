import os
import sys
import pickle
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel


# Project root folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Allow app.py to import gmail_features.py from src/
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from gmail_features import extract_email_features


# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "model", "baseline_rf.pkl")
FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "baseline_feature_names.pkl",
)

with open(MODEL_PATH, "rb") as file:
    MODEL = pickle.load(file)

with open(FEATURE_NAMES_PATH, "rb") as file:
    FEATURE_NAMES = pickle.load(file)


app = FastAPI(
    title="Spam Detector v2.0",
    description="Live Gmail-trained spam detection API",
)


class EmailInput(BaseModel):
    subject: str
    sender: str
    body: str = ""


class PredictionOutput(BaseModel):
    is_spam: int
    confidence: float
    spam_probability: float
    message: str


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/predict", response_model=PredictionOutput)
def predict(email: EmailInput):
    # Extract the same 11 features used during training
    features = extract_email_features(
        email.subject,
        email.sender,
        email.body,
    )

    # Keep exact feature order used while training
    X = np.array(
        [[features[name] for name in FEATURE_NAMES]]
    )

    prediction = MODEL.predict(X)[0]
    probabilities = MODEL.predict_proba(X)[0]

    spam_probability = float(probabilities[1])
    confidence = float(probabilities[int(prediction)])

    message = (
        "LIKELY SPAM"
        if prediction == 1
        else "LIKELY LEGITIMATE"
    )

    return PredictionOutput(
        is_spam=int(prediction),
        confidence=confidence,
        spam_probability=spam_probability,
        message=message,
    )
