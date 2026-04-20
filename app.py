
import pickle
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title       = "Breast Cancer Prediction API",
    description = "MLOps pipeline — XAI minor project",
    version     = "1.0.0"
)

# Load model and scaler on startup
try:
    model  = pickle.load(open("best_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    meta   = json.load(open("model_metadata.json"))
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found: {e}")

class PatientFeatures(BaseModel):
    features: List[float]
    class Config:
        json_schema_extra = {
            "example": {"features": [17.99, 10.38, 122.8, 1001.0, 0.1184,
                                      0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                                      1.095, 0.9053, 8.589, 153.4, 0.006399,
                                      0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                                      25.38, 17.33, 184.6, 2019.0, 0.1622,
                                      0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}
        }

class PredictionResponse(BaseModel):
    prediction        : int
    diagnosis         : str
    malignant_prob    : float
    benign_prob       : float
    model_name        : str
    confidence        : str

@app.get("/")
def root():
    return {
        "message"    : "Breast Cancer Prediction API is running",
        "model"      : meta["model_name"],
        "test_auc"   : meta["test_auc"],
        "n_features" : len(meta["features"]),
        "features"   : meta["features"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    if len(patient.features) != len(meta["features"]):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(meta['features'])} features, got {len(patient.features)}"
        )
    X       = np.array(patient.features).reshape(1, -1)
    X_sc    = scaler.transform(X)
    pred    = int(model.predict(X_sc)[0])
    proba   = model.predict_proba(X_sc)[0]
    mal_p   = float(proba[1])
    ben_p   = float(proba[0])
    conf    = "High" if max(mal_p, ben_p) > 0.85 else "Medium" if max(mal_p, ben_p) > 0.65 else "Low"
    return PredictionResponse(
        prediction     = pred,
        diagnosis      = "Malignant" if pred == 1 else "Benign",
        malignant_prob = round(mal_p, 4),
        benign_prob    = round(ben_p, 4),
        model_name     = meta["model_name"],
        confidence     = conf
    )

@app.post("/predict/batch")
def predict_batch(patients: List[PatientFeatures]):
    results = []
    for i, patient in enumerate(patients):
        X     = np.array(patient.features).reshape(1, -1)
        X_sc  = scaler.transform(X)
        pred  = int(model.predict(X_sc)[0])
        proba = model.predict_proba(X_sc)[0]
        results.append({
            "patient_index" : i,
            "prediction"    : pred,
            "diagnosis"     : "Malignant" if pred == 1 else "Benign",
            "malignant_prob": round(float(proba[1]), 4)
        })
    return {"predictions": results, "total": len(results)}
