from typing import Dict, Any, List, Tuple
import re
import math
import numpy as np
import pandas as pd

from .data_loader import (
    hr_basic_metrics, credit_basic_metrics,
    get_hr_df, get_cc_df
)
from .ml import (
    ensure_models_trained,
    hr_top_features, credit_top_features
)

# Intent router 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_INTENT_LABELS = [
    "hr_overview",
    "hr_by_dept",
    "hr_overtime",
    "hr_tenure",
    "credit_overview",
    "credit_by_pay",
    "credit_by_limit",
    "credit_policy",
    "sensitive_fire",
]

# Seed examples
_SEED: List[Tuple[str, str]] = [
    ("overall employee attrition and retention", "hr_overview"),
    ("organization wide attrition summary", "hr_overview"),

    ("which department has highest attrition", "hr_by_dept"),
    ("attrition by department ranking", "hr_by_dept"),

    ("does overtime increase attrition", "hr_overtime"),
    ("overtime impact on turnover", "hr_overtime"),

    ("attrition by tenure years at company", "hr_tenure"),
    ("turnover by years of service", "hr_tenure"),

    ("overall credit default situation", "credit_overview"),
    ("portfolio default summary", "credit_overview"),

    ("default by PAY_0 buckets", "credit_by_pay"),
    ("impact of recent payment status on default", "credit_by_pay"),

    ("default by credit limit", "credit_by_limit"),
    ("risk by limit segments", "credit_by_limit"),

    ("approval threshold or policy suggestion", "credit_policy"),
    ("which segments to tighten approval", "credit_policy"),

    ("who should I fire", "sensitive_fire"),
    ("which employees to terminate", "sensitive_fire"),
]

_CORPUS = [q for q, _ in _SEED]
_LABELS = [lbl for _, lbl in _SEED]
_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
_M = _vectorizer.fit_transform(_CORPUS)

def classify_intents_tfidf(questions: List[str], k: int = 4, thresh: float = 0.08) -> List[str]:
    text = " ".join([s or "" for s in questions]).strip().lower()
    if not text:
        return ["hr_overview", "credit_overview"]
    qv = _vectorizer.transform([text])
    sims = cosine_similarity(qv, _M)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    chosen: List[str] = []
    for idx, score in ranked[:max(k, 1)]:
        if score >= thresh:
            lbl = _LABELS[idx]
            if lbl not in chosen:
                chosen.append(lbl)
    if not chosen:
        chosen = ["hr_overview", "credit_overview"]
    return chosen

# Optional light regex fallback (English-only)
_INTENT_PATTERNS = {
    "hr_overview":  r"\b(employee|hr|attrition|retention|turnover)\b",
    "hr_by_dept":   r"\b(dept|department)\b",
    "hr_overtime":  r"\b(overtime)\b",
    "hr_tenure":    r"\b(tenure|years at company|years of service)\b",

    "credit_overview": r"\b(credit|default|portfolio)\b",
    "credit_by_pay":   r"\b(pay[_ ]?0|recent status|payment status)\b",
    "credit_by_limit": r"\b(limit|limit[_ ]?bal|credit limit)\b",
    "credit_policy":   r"\b(policy|threshold|approval)\b",

    "sensitive_fire":  r"\b(fire|terminate|layoff|who should i fire)\b",
}

def classify_intents_regex(questions: List[str]) -> List[str]:
    q = " ".join([s or "" for s in questions]).lower()
    intents = []
    for name, pat in _INTENT_PATTERNS.items():
        if re.search(pat, q, re.I):
            intents.append(name)
    if not intents:
        intents = ["hr_overview", "credit_overview"]
    return intents

def route_intents(questions: List[str]) -> List[str]:
    try:
        cand = classify_intents_tfidf(questions)
        if cand:
            return cand
    except Exception:
        pass
    return classify_intents_regex(questions)

# Helpers
def coef_fmt(w: float) -> str:
    s = f"{w:.3f}"
    return s if w == 0 else (s + (" (↑)" if w > 0 else " (↓)"))

def format_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df is None or len(df) == 0:
        return "no data"
    d = df.head(max_rows).reset_index()
    items = []
    for _, r in d.iterrows():
        key = str(r.iloc[0])
        rate = float(r.get("rate", np.nan))
        count = r.get("count", None)
        if pd.notna(rate):
            if pd.notna(count):
                items.append(f"{key}: {rate*100:.1f}% (n={int(count)})")
            else:
                items.append(f"{key}: {rate*100:.1f}%")
    return "; ".join(items) if items else "no data"

def rate_by(df: pd.DataFrame, by: str, target: str, positive) -> pd.DataFrame:
    y = df[target].apply(positive) if callable(positive) else df[target].isin(positive)
    out = (
        pd.DataFrame({"y": y})
        .join(df[[by]])
        .groupby(by)["y"].mean()
        .sort_values(ascending=False)
        .rename("rate")
        .to_frame()
    )
    out["count"] = df.groupby(by)[target].count()
    return out

def bucket_numeric(df: pd.DataFrame, col: str, bins: List[float], labels: List[str] | None = None) -> pd.Series:
    s = df[col].astype(float)
    if labels is None:
        labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)

# Main entry
def generate_insights(questions: List[str], controls: Dict[str, Any] | None) -> Dict[str, Any]:
    # Ensure models are trained 
    hr_info, cc_info = ensure_models_trained()

    # Global summaries
    hr_m = hr_basic_metrics()
    cr_m = credit_basic_metrics()
    hr_feats = hr_top_features(k=6)
    cc_feats = credit_top_features(k=6)

    summary = (
        f"Employee attrition rate: {hr_m['attrition_rate']*100:.2f}% (rows={hr_m['rowcount']}), "
        f"job involvement avg={hr_m['job_involvement_mean']:.2f}. "
        f"Credit default rate: {cr_m['default_rate']*100:.2f}% (rows={cr_m['rowcount']}), "
        f"avg limit={cr_m['avg_limit']:.0f}."
    )

    risks: List[str] = []
    if hr_m["attrition_rate"] > 0.15:
        risks.append("High employee attrition rate (>15%) indicates potential retention issues.")
    if cr_m["default_rate"] > 0.20:
        risks.append("Elevated credit default rate (>20%) increases portfolio risk.")
    if hr_feats:
        risks.append("Attrition drivers (model coef): " + ", ".join([f"{k}:{coef_fmt(v)}" for k, v in hr_feats.items()]))
    if cc_feats:
        risks.append("Default drivers (model coef): " + ", ".join([f"{k}:{coef_fmt(v)}" for k, v in cc_feats.items()]))

    actions: List[str] = [
        "HR: Launch targeted retention interviews in top-attrition departments; KPI: reduce attrition by 2pp within one quarter.",
        "HR: Manager coaching on workload/work-life balance where JobInvolvement is low; KPI: +0.2 average improvement in one quarter.",
        "Credit: Tighten approval policy for high-risk PAY_0 segments; KPI: -3pp default while maintaining approval targets.",
        "Credit: Early warning and outreach for worsening PAY_X trajectories; KPI: +15% on-time payment next cycle.",
    ]
    caveats: List[str] = [
        f"Models are LogisticRegression trained locally; HR AUC={hr_info['metrics']['auc']:.3f}, Credit AUC={cc_info['metrics']['auc']:.3f}.",
        "Coefficients indicate linear associations, not strict causality.",
        "Feature coverage is simplified; consider adding more features/regularization or alternative models.",
    ]
    if questions:
        caveats.append(f"Questions considered: {questions}")

    # Intent-specific answers
    hr_df = get_hr_df()
    cc_df = get_cc_df()
    answers: List[str] = []
    intents = route_intents(questions)

    # HR: by department
    if "hr_by_dept" in intents and {"Department", "Attrition"}.issubset(hr_df.columns):
        tbl = rate_by(hr_df, "Department", "Attrition", positive={"Yes"}).head(5)
        answers.append("HR by Department (top attrition): " + format_table(tbl))

    # HR: overtime vs attrition
    if "hr_overtime" in intents and {"OverTime", "Attrition"}.issubset(hr_df.columns):
        tbl = rate_by(hr_df, "OverTime", "Attrition", positive={"Yes"})
        answers.append("HR OverTime vs Attrition: " + format_table(tbl))

    # HR: attrition by tenure
    if "hr_tenure" in intents and {"YearsAtCompany", "Attrition"}.issubset(hr_df.columns):
        hr_tmp = hr_df.copy()
        bins = [0, 1, 3, 5, 10, 40]
        labels = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]
        hr_tmp["TenureBin"] = bucket_numeric(hr_tmp, "YearsAtCompany", bins, labels)
        tbl = rate_by(hr_tmp, "TenureBin", "Attrition", positive={"Yes"})
        answers.append("HR Attrition by Tenure: " + format_table(tbl))

    # Credit: default by PAY_0
    if "credit_by_pay" in intents and {"PAY_0", "default.payment.next.month"}.issubset(cc_df.columns):
        tbl = rate_by(cc_df, "PAY_0", "default.payment.next.month", positive={1}).head(10)
        answers.append("Credit Default by PAY_0: " + format_table(tbl))

    # Credit: default by LIMIT_BAL buckets
    if "credit_by_limit" in intents and {"LIMIT_BAL", "default.payment.next.month"}.issubset(cc_df.columns):
        cc_tmp = cc_df.copy()
        bins = [0, 50_000, 100_000, 200_000, 400_000, 1_000_000]
        labels = ["<50k", "50-100k", "100-200k", "200-400k", "400k+"]
        cc_tmp["LimitBin"] = bucket_numeric(cc_tmp, "LIMIT_BAL", bins, labels)
        tbl = rate_by(cc_tmp, "LimitBin", "default.payment.next.month", positive={1})
        answers.append("Credit Default by Limit: " + format_table(tbl))

    # Credit: simple policy hint
    if "credit_policy" in intents and {"PAY_0", "LIMIT_BAL", "default.payment.next.month"}.issubset(cc_df.columns):
        high_risk = (cc_df["PAY_0"] >= 1) & (cc_df["LIMIT_BAL"] < 100_000)
        rate_high = cc_df.loc[high_risk, "default.payment.next.month"].mean()
        rate_others = cc_df.loc[~high_risk, "default.payment.next.month"].mean()
        answers.append(
            f"Policy hint: PAY_0>=1 & LIMIT_BAL<100k default={rate_high*100:.1f}% vs others {rate_others*100:.1f}% → consider a higher approval threshold or manual review."
        )

    # Sensitive ask
    if "sensitive_fire" in intents:
        answers.append(
            "Ethics & Compliance: Do not use models to make firing decisions. Use fair process: performance goals, coaching, PIP, and documented reviews; focus on workload, role fit, and manager support."
        )

    # If nothing matched, at least surface model drivers
    if not answers:
        if hr_feats:
            answers.append("HR drivers: " + ", ".join([f"{k}:{coef_fmt(v)}" for k, v in hr_feats.items()]))
        if cc_feats:
            answers.append("Credit drivers: " + ", ".join([f"{k}:{coef_fmt(v)}" for k, v in cc_feats.items()]))

    return {
        "summary": summary,
        "risks": risks,
        "actions": actions,
        "caveats": caveats + ["Answer blocks are intent-specific."],
        "answers": answers,
    }
