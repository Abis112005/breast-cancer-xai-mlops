
import pickle
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def test_model_files_exist():
    import os
    assert os.path.exists("models/best_model.pkl"),      "Model file missing"
    assert os.path.exists("models/scaler.pkl"),           "Scaler file missing"
    assert os.path.exists("models/model_metadata.json"), "Metadata file missing"

def test_model_accuracy():
    raw    = load_breast_cancer()
    model  = pickle.load(open("models/best_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl",     "rb"))
    X = raw.data
    y = raw.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)
    y_pred    = model.predict(X_test_sc)
    acc       = accuracy_score(y_test, y_pred)
    assert acc >= 0.90, f"Accuracy too low: {acc:.4f}"
    print(f"Accuracy test passed: {acc:.4f}")

def test_model_recall():
    raw    = load_breast_cancer()
    model  = pickle.load(open("models/best_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl",     "rb"))
    X = raw.data
    y = raw.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)
    y_pred    = model.predict(X_test_sc)
    rec       = recall_score(y_test, y_pred)
    assert rec >= 0.90, f"Recall too low: {rec:.4f} — too many malignant cases missed!"
    print(f"Recall test passed: {rec:.4f}")

def test_prediction_output_shape():
    raw    = load_breast_cancer()
    model  = pickle.load(open("models/best_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl",     "rb"))
    sample = scaler.transform(raw.data[:5])
    preds  = model.predict(sample)
    probas = model.predict_proba(sample)
    assert preds.shape  == (5,),    f"Wrong prediction shape: {preds.shape}"
    assert probas.shape == (5, 2),  f"Wrong probability shape: {probas.shape}"
    assert all(p in [0, 1] for p in preds), "Predictions must be 0 or 1"
    print("Output shape test passed")

def test_metadata_keys():
    meta = json.load(open("models/model_metadata.json"))
    for key in ["model_name", "run_id", "test_auc", "features"]:
        assert key in meta, f"Missing key in metadata: {key}"
    assert meta["test_auc"] >= 0.95, f'AUC in metadata too low: {meta["test_auc"]}'
    print(f'Metadata test passed — model: {meta["model_name"]}, AUC: {meta["test_auc"]}')

if __name__ == "__main__":
    test_model_files_exist()
    test_model_accuracy()
    test_model_recall()
    test_prediction_output_shape()
    test_metadata_keys()
    print("Testing passed")
