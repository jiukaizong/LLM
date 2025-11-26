import os, json, math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HR_PATH = os.getenv("HR_CSV", str(ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"))
CC_PATH = os.getenv("CC_CSV", str(ROOT / "data" / "UCI_Credit_Card.csv"))
OUT_PATH = ROOT / "train_data" / "chat_train.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def q(o):  
    return json.dumps(o, ensure_ascii=False)

def build_hr_pairs(df: pd.DataFrame):
    rows = []
    # Overall attrition
    if "Attrition" in df.columns:
        rate = (df["Attrition"] == "Yes").mean()*100
        rows.append({
            "instruction": "What is the overall employee attrition rate?",
            "input": "",
            "output": f"The overall attrition rate is {rate:.2f}%."
        })
    # By department
    if {"Department", "Attrition"}.issubset(df.columns):
        by = (df["Attrition"].eq("Yes").groupby(df["Department"]).mean()*100).sort_values(ascending=False)
        top = "; ".join([f"{k}: {v:.2f}%" for k,v in by.head(5).items()])
        rows.append({
            "instruction": "Which department has the highest attrition rate?",
            "input": "",
            "output": f"Top departments by attrition: {top}."
        })
    # Overtime vs attrition
    if {"OverTime","Attrition"}.issubset(df.columns):
        t = df.groupby("OverTime")["Attrition"].apply(lambda s: (s=="Yes").mean()*100)
        detail = "; ".join([f"{k}: {v:.2f}%" for k,v in t.items()])
        rows.append({
            "instruction": "Does overtime increase employee attrition?",
            "input": "",
            "output": f"Attrition by overtime: {detail}. Employees with OverTime=Yes show higher turnover if the percentage is higher."
        })
    # Tenure buckets
    if {"YearsAtCompany","Attrition"}.issubset(df.columns):
        bins = [0,1,3,5,10,40]; labels = ["<1y","1-3y","3-5y","5-10y","10y+"]
        tmp = df.copy()
        tmp["TenureBin"] = pd.cut(tmp["YearsAtCompany"], bins=bins, labels=labels, include_lowest=True)
        g = tmp.groupby("TenureBin")["Attrition"].apply(lambda s: (s=="Yes").mean()*100)
        detail = "; ".join([f"{str(k)}: {v:.2f}%" for k,v in g.items()])
        rows.append({
            "instruction": "How does attrition vary by tenure?",
            "input": "",
            "output": f"Attrition by tenure buckets: {detail}."
        })
    # Ethics
    rows.append({
        "instruction": "Who should I fire?",
        "input": "",
        "output": "Firing decisions must not be automated. Use a fair process: clear goals, coaching, performance improvement plans, and documented reviews."
    })
    return rows

def build_cc_pairs(df: pd.DataFrame):
    rows = []
    # Overall default rate
    if "default.payment.next.month" in df.columns:
        rate = df["default.payment.next.month"].mean()*100
        rows.append({
            "instruction": "What is the current credit default rate?",
            "input": "",
            "output": f"The portfolio default rate is {rate:.2f}%."
        })
    # PAY_0
    if {"PAY_0","default.payment.next.month"}.issubset(df.columns):
        g = df.groupby("PAY_0")["default.payment.next.month"].mean().sort_values(ascending=False)*100
        detail = "; ".join([f"{int(k)}: {v:.2f}%" for k,v in g.items()])
        rows.append({
            "instruction": "How does default vary by PAY_0?",
            "input": "",
            "output": f"Default by PAY_0: {detail}."
        })
    # LIMIT_BAL buckets
    if {"LIMIT_BAL","default.payment.next.month"}.issubset(df.columns):
        bins = [0,50_000,100_000,200_000,400_000,1_000_000]
        labels = ["<50k","50-100k","100-200k","200-400k","400k+"]
        tmp = df.copy()
        tmp["LimitBin"] = pd.cut(tmp["LIMIT_BAL"], bins=bins, labels=labels, include_lowest=True)
        g = tmp.groupby("LimitBin")["default.payment.next.month"].mean()*100
        detail = "; ".join([f"{k}: {v:.2f}%" for k,v in g.items()])
        rows.append({
            "instruction": "Which credit-limit segments are riskiest?",
            "input": "",
            "output": f"Default by credit limit: {detail}."
        })
        rows.append({
            "instruction": "Which segments should we tighten approval for?",
            "input": "",
            "output": "Customers with PAY_0 >= 1 and LIMIT_BAL < 100k show significantly higher default; consider higher approval thresholds or manual review."
        })
    return rows

def main():
    hr = pd.read_csv(HR_PATH)
    cc = pd.read_csv(CC_PATH)
    rows = build_hr_pairs(hr) + build_cc_pairs(cc)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(q(r) + "\n")
    print(f"Wrote {len(rows)} samples to {OUT_PATH}")

if __name__ == "__main__":
    main()
