import os
from pathlib import Path
from textwrap import dedent
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import math

# save path
ROOT = Path(__file__).resolve().parents[1]
HR_PATH = os.getenv("HR_CSV", str(ROOT / "data" / "Employee_Attrition.csv"))
CC_PATH = os.getenv("CC_CSV", str(ROOT / "data" / "UCI_Credit_Card.csv"))

_hr_df = None
_cc_df = None

def _json_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame to be JSON-safe: Inf -> NaN, then NaN -> None."""
    safe = df.replace([np.inf, -np.inf], np.nan)
    return safe.where(pd.notna(safe), None)

def _json_safe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:

    df2 = df.replace([np.inf, -np.inf], np.nan)

    raw_records: List[Dict[str, Any]] = df2.to_dict(orient="records")

    def _fix_val(v):
        if isinstance(v, (np.generic,)):
            v = v.item()

        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if v is None:
            return None
        return v

    out: List[Dict[str, Any]] = []
    for rec in raw_records:
        out.append({k: _fix_val(v) for k, v in rec.items()})
    return out

def _pct(x: float) -> str:
    try:
        return f"{round(float(x) * 100, 2)}%"
    except Exception:
        return "n/a"

# preprocessing
def get_hr_df() -> pd.DataFrame:
    global _hr_df
    if _hr_df is not None:
        return _hr_df
    df = pd.read_csv(HR_PATH)
    df = df.copy()

    if "YearsAtCompany" in df.columns:
        df["TenureBin"] = pd.cut(df["YearsAtCompany"], bins=[-1,1,3,5,10,100], labels=["<=1","2-3","4-5","6-10","10+" ])
    if "MonthlyIncome" in df.columns:
        q = df["MonthlyIncome"].quantile([.25,.5,.75]).to_dict()
        df["IncomeBin"] = pd.cut(df["MonthlyIncome"], bins=[-1,q[.25],q[.5],q[.75],df["MonthlyIncome"].max()+1],
                                 labels=["Q1","Q2","Q3","Q4"])
    if "JobSatisfaction" in df.columns:
        df["LowSatisfaction"] = (df["JobSatisfaction"] <= 2).astype(int)
    if "OverTime" in df.columns:
        df["OverTimeFlag"] = (df["OverTime"].astype(str).str.lower().eq("yes")).astype(int)
    if "Attrition" in df.columns:
        df["AttritionFlag"] = df["Attrition"].astype(str).str.lower().eq("yes").astype(int)

    _hr_df = df
    return _hr_df

def get_cc_df() -> pd.DataFrame:
    global _cc_df
    if _cc_df is not None:
        return _cc_df
    df = pd.read_csv(CC_PATH)
    df = df.copy()

    if "LIMIT_BAL" in df.columns:
        try:
            df["LimitBin"] = pd.qcut(df["LIMIT_BAL"], q=5, labels=[1,2,3,4,5])
        except Exception:
            df["LimitBin"] = pd.cut(df["LIMIT_BAL"], bins=5, labels=[1,2,3,4,5])

    if "PAY_0" in df.columns:
        df["LateRecent"] = (df["PAY_0"] > 0).astype(int)

    if "PAY_AMT1" in df.columns and "BILL_AMT1" in df.columns:
        denom = df["BILL_AMT1"].replace(0, np.nan)
        repay_ratio = (df["PAY_AMT1"] / denom).replace([np.inf, -np.inf], np.nan).clip(upper=5)
        df["RepayRatio1"] = repay_ratio
        df["RepayBucket1"] = pd.cut(repay_ratio, bins=[-0.01, 0.2, 0.5, 1.0, 10],
                                    labels=["<20%", "20-50%", "50-100%", ">=100%"])
    if "default.payment.next.month" in df.columns:
        df["DefaultFlag"] = df["default.payment.next.month"].astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    _cc_df = df
    return _cc_df


def hr_snapshot_text() -> str:
    df = get_hr_df()
    total = len(df)
    if total == 0:
        return "[HR SNAPSHOT] No data rows."

    attr_rate = df["AttritionFlag"].mean() if "AttritionFlag" in df.columns else np.nan

    by_dept = None
    if "Department" in df.columns and "AttritionFlag" in df.columns:
        by_dept = (df.groupby("Department")["AttritionFlag"].mean().sort_values(ascending=False))

    by_ot = None
    if "OverTime" in df.columns and "AttritionFlag" in df.columns:
        by_ot = (df.groupby("OverTime")["AttritionFlag"].mean().sort_values(ascending=False))

    risky_cov = []
    if "LowSatisfaction" in df.columns:
        risky_cov.append(("Low JobSatisfaction (1-2)", df["LowSatisfaction"].mean()))
    if "OverTimeFlag" in df.columns:
        risky_cov.append(("OverTime=Yes", df["OverTimeFlag"].mean()))
    if "IncomeBin" in df.columns:
        risky_cov.append(("Income Q1 (lowest)", (df["IncomeBin"] == "Q1").mean()))
    risky_cov = sorted(risky_cov, key=lambda t: t[1], reverse=True)[:3]

    top_dept = "n/a"
    if isinstance(by_dept, pd.Series) and len(by_dept) > 0:
        top_dept = "; ".join([f"{k}={_pct(v)}" for k, v in by_dept.head(3).items()])

    top_ot = "n/a"
    if isinstance(by_ot, pd.Series) and len(by_ot) > 0:
        top_ot = "; ".join([f"{k}={_pct(v)}" for k, v in by_ot.items()])

    risky_lines = []
    for name, share in risky_cov:
        risky_lines.append(f"  - {name}: {_pct(share)} of employees")

    return dedent(f"""
    [HR SNAPSHOT]
    Rows: {total}
    Overall Attrition Rate: {_pct(attr_rate) if not np.isnan(attr_rate) else 'n/a'}
    Highest Attrition by Department (top 3): {top_dept}
    Attrition by OverTime: {top_ot}
    High-coverage segments:
{os.linesep.join(risky_lines) if risky_lines else '  - n/a'}
    """).strip()

def cc_snapshot_text() -> str:
    df = get_cc_df()
    total = len(df)
    if total == 0:
        return "[CREDIT SNAPSHOT] No data rows."

    def_rate = df["DefaultFlag"].mean() if "DefaultFlag" in df.columns else np.nan

    by_limit = None
    if "LimitBin" in df.columns and "DefaultFlag" in df.columns:
        by_limit = (df.groupby("LimitBin")["DefaultFlag"].mean().sort_values(ascending=False))

    late_recent = df["LateRecent"].mean() if "LateRecent" in df.columns else np.nan
    good = poor = np.nan
    if "RepayRatio1" in df.columns:
        good = (df["RepayRatio1"] >= 1).mean()
        poor = (df["RepayRatio1"] < 0.2).mean()

    top_limit = "n/a"
    if isinstance(by_limit, pd.Series) and len(by_limit) > 0:
        top_limit = "; ".join([f"Bin{int(k)}={_pct(v)}" for k, v in by_limit.head(3).items()])

    return dedent(f"""
    [CREDIT SNAPSHOT]
    Rows: {total}
    Overall Default Rate (next month): {_pct(def_rate) if not np.isnan(def_rate) else 'n/a'}
    Highest Default by Limit Bin (top 3): {top_limit}
    Late payment (recent cycle, PAY_0>0): {_pct(late_recent) if not np.isnan(late_recent) else 'n/a'}
    Repayment ratio (recent cycle):
      - Good (>=100% of bill): {_pct(good) if not np.isnan(good) else 'n/a'}
      - Poor (<20% of bill): {_pct(poor) if not np.isnan(poor) else 'n/a'}
    """).strip()


def rate_by_column_text(df: pd.DataFrame, target_col: str, group_col: str, topn: int = 10) -> str:
    if target_col not in df.columns or group_col not in df.columns:
        return f"[GROUP STATS] Missing column(s): target={target_col}, group={group_col}"
    g = (df.groupby(group_col)[target_col].mean().sort_values(ascending=False)).head(topn)
    if g.empty:
        return f"[GROUP STATS] No groups for {group_col}"
    return f"[GROUP STATS] {target_col} by {group_col} (top {min(topn, len(g))}): " + \
           "; ".join([f"{k}={_pct(v)}" for k, v in g.items()])


_HR_KEYS = [
    "attrition","employee","employees","hr","turnover","satisfaction","overtime",
    "department","tenure","income","promotion","performance"
]
_CC_KEYS = [
    "credit","card","default","limit","repay","bill","payment","risk","late","overdue","loan"
]

_HR_COLUMN_HINTS = {
    r"\bdepartment(s)?\b": ("AttritionFlag", "Department"),
    r"\bovertime\b": ("AttritionFlag", "OverTime"),
    r"\btenure|year(s)?\b": ("AttritionFlag", "TenureBin"),
    r"\bincome\b": ("AttritionFlag", "IncomeBin"),
    r"\bsatisfaction\b": ("AttritionFlag", "JobSatisfaction"),
}
_CC_COLUMN_HINTS = {
    r"\blimit(s)?\b": ("DefaultFlag", "LimitBin"),
    r"\blate|overdue|pay_0|delinquen": ("DefaultFlag", "LateRecent"),
    r"\brepay|payment\b": ("DefaultFlag", "RepayBucket1"),
    r"\beducation\b": ("DefaultFlag", "EDUCATION"),
    r"\bmarriage\b": ("DefaultFlag", "MARRIAGE"),
    r"\bage\b": ("DefaultFlag", "AGE"),
}

def _match_hints(text: str, hint_map: dict) -> list[tuple[str, str]]:
    matched = []
    for pat, cols in hint_map.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            matched.append(cols)
    return matched

def hr_context_for(question: str) -> str:
    df = get_hr_df()
    parts = [hr_snapshot_text()]
    for target, group in _match_hints(question, _HR_COLUMN_HINTS):
        if group in df.columns and target in df.columns:
            parts.append(rate_by_column_text(df, target, group))
    for group in ["Department","OverTime","TenureBin","IncomeBin"]:
        if group in df.columns and "AttritionFlag" in df.columns:
            parts.append(rate_by_column_text(df, "AttritionFlag", group, topn=5))
    return "\n".join(dict.fromkeys(parts)) 

def cc_context_for(question: str) -> str:
    df = get_cc_df()
    parts = [cc_snapshot_text()]
    for target, group in _match_hints(question, _CC_COLUMN_HINTS):
        if group in df.columns and target in df.columns:
            parts.append(rate_by_column_text(df, target, group))
    for group in ["LimitBin","RepayBucket1","LateRecent","EDUCATION","MARRIAGE"]:
        if group in df.columns and "DefaultFlag" in df.columns:
            parts.append(rate_by_column_text(df, "DefaultFlag", group, topn=5))
    return "\n".join(dict.fromkeys(parts))

def build_data_context(user_question: str, extra_context: str | None = None) -> str:

    uq = (user_question or "").lower()
    use_hr = any(k in uq for k in _HR_KEYS)
    use_cc = any(k in uq for k in _CC_KEYS)

    ctx_parts = []
    if use_hr: ctx_parts.append(hr_context_for(uq))
    if use_cc: ctx_parts.append(cc_context_for(uq))
    if not ctx_parts:  
        ctx_parts = [hr_context_for(uq), cc_context_for(uq)]

    if extra_context:
        ctx_parts.append(dedent(f"[EXTRA CONTEXT]\n{extra_context.strip()}"))

    return "\n\n".join(ctx_parts).strip()

def hr_basic_metrics():
    """Return a compact HR metrics dict for /api/employee/metrics and insights.py."""
    df = get_hr_df()
    out = {"rows": int(len(df))}
    if "AttritionFlag" in df.columns:
        out["attrition_rate"] = float(df["AttritionFlag"].mean())
    # Top departments by attrition
    if {"Department", "AttritionFlag"} <= set(df.columns):
        dep = (
            df.groupby("Department")["AttritionFlag"]
              .mean()
              .sort_values(ascending=False)
              .head(5)
              .round(4)
              .to_dict()
        )
        out["attrition_by_department"] = dep
    # OverTime split
    if {"OverTime", "AttritionFlag"} <= set(df.columns):
        ot = (
            df.groupby("OverTime")["AttritionFlag"]
              .mean()
              .sort_values(ascending=False)
              .round(4)
              .to_dict()
        )
        out["attrition_by_overtime"] = ot
    # Tenure / Income bins if available
    if {"TenureBin", "AttritionFlag"} <= set(df.columns):
        out["attrition_by_tenure"] = (
            df.groupby("TenureBin")["AttritionFlag"].mean().round(4).to_dict()
        )
    if {"IncomeBin", "AttritionFlag"} <= set(df.columns):
        out["attrition_by_incomebin"] = (
            df.groupby("IncomeBin")["AttritionFlag"].mean().round(4).to_dict()
        )
    return out


def credit_basic_metrics():
    """Return a compact Credit metrics dict for /api/credit/metrics and insights.py."""
    df = get_cc_df()
    out = {"rows": int(len(df))}
    out["rowcount"] = out["rows"]

    if "LIMIT_BAL" in df.columns:
        out["avg_limit"] = float(pd.to_numeric(df["LIMIT_BAL"], errors="coerce").mean())

    if "DefaultFlag" in df.columns:
        out["default_rate"] = float(df["DefaultFlag"].mean())

    if {"LimitBin", "DefaultFlag"} <= set(df.columns):
        out["default_by_limitbin"] = (
            df.groupby("LimitBin")["DefaultFlag"]
              .mean()
              .sort_values(ascending=False)
              .round(4)
              .to_dict()
        )

    # Late recent
    if "LateRecent" in df.columns:
        out["late_recent_share"] = float(df["LateRecent"].mean())

    # Repayment buckets
    if {"RepayBucket1", "DefaultFlag"} <= set(df.columns):
        out["default_by_repaybucket1"] = (
            df.groupby("RepayBucket1")["DefaultFlag"]
              .mean()
              .sort_values(ascending=False)
              .round(4)
              .to_dict()
        )

    # PAY_0 分布（前端用 pay0_dist_top5）
    if "PAY_0" in df.columns:
        s = (
            pd.to_numeric(df["PAY_0"], errors="coerce")
              .value_counts(normalize=True, dropna=True)
              .sort_values(ascending=False)
              .head(5)
              .round(4)
              .to_dict()
        )
        out["pay0_dist_top5"] = s

    return out


def _page_dict(df: pd.DataFrame, page: int, page_size: int) -> Dict[str, Any]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 1000))
    total = int(len(df))
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    slice_df = df.iloc[start:end].copy()
    rows: List[Dict[str, Any]] = _json_safe_records(slice_df)
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "columns": list(df.columns),
        "rows": rows,
        "has_next": end < total,
        "has_prev": start > 0,
    }

def _page_slice(df: pd.DataFrame, offset: int, limit: int) -> Dict[str, Any]:
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 1000))
    total = int(len(df))
    end = min(offset + limit, total)
    slice_df = df.iloc[offset:end].copy()
    rows: List[Dict[str, Any]] = _json_safe_records(slice_df)
    return {
        "offset": offset,
        "limit": limit,
        "total": total,
        "columns": list(df.columns),
        "rows": rows,
        "has_next": end < total,
        "has_prev": offset > 0,
    }

def get_hr_page(
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Supports either offset/limit (frontend) or page/page_size (older code)."""
    df = get_hr_df()
    if offset is not None or limit is not None:
        return _page_slice(df, offset or 0, limit or 50)
    # fallback to page/page_size
    return _page_dict(df, page or 1, page_size or 50)

def get_cc_page(
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Supports either offset/limit (frontend) or page/page_size (older code)."""
    df = get_cc_df()
    if offset is not None or limit is not None:
        return _page_slice(df, offset or 0, limit or 50)
    # fallback to page/page_size
    return _page_dict(df, page or 1, page_size or 50)