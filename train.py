
import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

def load_data():
    raw = load_breast_cancer()
    df  = pd.DataFrame(raw.data, columns=raw.feature_names)
    df["diagnosis"] = raw.target
    return df, raw

def preprocess(df, raw, seed=SEED):
    X = df[list(raw.feature_names)]
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler

def get_models(seed=SEED):
    return {
        "LogisticRegression" : LogisticRegression(max_iter=1000, random_state=seed),
        "RandomForest"       : RandomForestClassifier(n_estimators=200, random_state=seed),
        "ExtraTrees"         : ExtraTreesClassifier(n_estimators=200, random_state=seed),
        "SVM_RBF"            : SVC(kernel="rbf", probability=True, random_state=seed),
        "LightGBM"           : LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1),
        "XGBoost"            : XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric="logloss", random_state=seed),
    }

def train_and_log(experiment_name="breast-cancer-classification"):
    mlflow.set_experiment(experiment_name)
    df, raw = load_data()
    X_train_sc, X_test_sc, y_train, y_test, scaler = preprocess(df, raw)
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            cv_auc = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="roc_auc")
            mlflow.log_metric("cv_auc_mean", cv_auc.mean())
            model.fit(X_train_sc, y_train)
            y_pred  = model.predict(X_test_sc)
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            test_auc = roc_auc_score(y_test, y_proba)
            test_acc = accuracy_score(y_test, y_pred)
            test_rec = recall_score(y_test, y_pred)
            mlflow.log_metric("test_auc",      test_auc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_recall",   test_rec)
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            results.append({"name": name, "auc": test_auc, "run_id": run_id, "model": model})
            print(f"{name:22s} | AUC={test_auc:.4f}")

    best = max(results, key=lambda x: x["auc"])
    os.makedirs("models", exist_ok=True)
    pickle.dump(best["model"], open("models/best_model.pkl", "wb"))
    pickle.dump(scaler,        open("models/scaler.pkl",     "wb"))
    meta = {"model_name": best["name"], "run_id": best["run_id"],
            "test_auc": best["auc"], "features": list(raw.feature_names)}
    json.dump(meta, open("models/model_metadata.json", "w"), indent=2)
    print(f"
Best model: {best["name"]} (AUC={best["auc"]:.4f})")
    return best

if __name__ == "__main__":
    train_and_log()
