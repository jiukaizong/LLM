import os, joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from .data_loader import get_hr_df, get_cc_df

MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.makedirs(MODELS_DIR, exist_ok=True)

HR_MODEL_PATH = os.path.join(MODELS_DIR, "hr_attrition_logreg.joblib")
CC_MODEL_PATH = os.path.join(MODELS_DIR, "credit_default_logreg.joblib")

# HR: Attrition 
def train_hr_model(random_state=42) -> Dict[str, Any]:
    df = get_hr_df().copy()
    target = "Attrition"
    y = (df[target] == "Yes").astype(int)

    num_cols = ["Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome","NumCompaniesWorked",
                "PercentSalaryHike","TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany",
                "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","JobInvolvement"]
    cat_cols = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus",
                "OverTime","Education","EnvironmentSatisfaction","JobSatisfaction","WorkLifeBalance"]

    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    X = df[num_cols + cat_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=random_state)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "acc": float(accuracy_score(y_test, y_pred)),
        "pos_rate_test": float(y_test.mean())
    }

    joblib.dump({"pipe": pipe, "num_cols": num_cols, "cat_cols": cat_cols, "metrics": metrics}, HR_MODEL_PATH)
    return {"path": HR_MODEL_PATH, "metrics": metrics}

def load_hr_model():
    if not os.path.exists(HR_MODEL_PATH):
        return None
    return joblib.load(HR_MODEL_PATH)

def hr_top_features(k=8) -> Dict[str, float]:
    model = load_hr_model()
    if model is None: 
        return {}
    pipe = model["pipe"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    pre: ColumnTransformer = pipe.named_steps["pre"]

    num_cols = model["num_cols"]
    cat_cols = model["cat_cols"]

    ohe: OneHotEncoder = pre.named_transformers_["cat"]
    cat_names = []
    if cat_cols:
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    feat_names = num_cols + cat_names

    coefs = clf.coef_.ravel()
    pairs = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:k]
    return {name: float(weight) for name, weight in pairs}

# Credit: Default
def train_credit_model(random_state=42) -> Dict[str, Any]:
    df = get_cc_df().copy()
    target = "default.payment.next.month"
    y = df[target].astype(int)

    drop_cols = [target, "ID"] if "ID" in df.columns else [target]
    X = df.drop(columns=drop_cols, errors="ignore")

    num_cols = X.columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(with_mean=False), num_cols)])
    clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=random_state)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "acc": float(accuracy_score(y_test, y_pred)),
        "pos_rate_test": float(y_test.mean())
    }

    joblib.dump({"pipe": pipe, "num_cols": num_cols, "metrics": metrics}, CC_MODEL_PATH)
    return {"path": CC_MODEL_PATH, "metrics": metrics}

def load_credit_model():
    if not os.path.exists(CC_MODEL_PATH):
        return None
    return joblib.load(CC_MODEL_PATH)

def credit_top_features(k=8) -> Dict[str, float]:
    model = load_credit_model()
    if model is None: 
        return {}
    pipe = model["pipe"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    num_cols = model["num_cols"]
    coefs = clf.coef_.ravel()
    pairs = sorted(zip(num_cols, coefs), key=lambda x: abs(x[1]), reverse=True)[:k]
    return {name: float(weight) for name, weight in pairs}

# Ensure trained
def ensure_models_trained() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hr_info, cc_info = {}, {}
    if not os.path.exists(HR_MODEL_PATH):
        hr_info = train_hr_model()
    else:
        hr_info = {"path": HR_MODEL_PATH, "metrics": load_hr_model()["metrics"]}
    if not os.path.exists(CC_MODEL_PATH):
        cc_info = train_credit_model()
    else:
        cc_info = {"path": CC_MODEL_PATH, "metrics": load_credit_model()["metrics"]}
    return hr_info, cc_info
