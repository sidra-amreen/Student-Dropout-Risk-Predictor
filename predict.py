import json
import sys
import joblib
import numpy as np
import pandas as pd

from features import FEATURES, LABELS

bundle = joblib.load("dropout_model.joblib")
model, explainer = bundle["model"], bundle["explainer"]


def tier(p: float) -> str:
    return "HIGH" if p >= 0.6 else "MEDIUM" if p >= 0.35 else "LOW"


def explain(row: pd.DataFrame, top: int = 3):
    """Top factors raising/lowering risk, using the linear model's contributions."""
    scaler, lr = explainer[0], explainer[-1]
    contrib = lr.coef_[0] * scaler.transform(row)[0]
    order = np.argsort(contrib)
    up = [(LABELS[FEATURES[i]], row.iloc[0, i]) for i in order[::-1][:top] if contrib[i] > 0]
    down = [(LABELS[FEATURES[i]], row.iloc[0, i]) for i in order[:top] if contrib[i] < 0]
    return up, down


def assess(student: dict):
    row = pd.DataFrame([student])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    up, down = explain(row)
    print(f"Dropout probability: {p:.1%}  ->  {tier(p)} RISK")
    print("  Raising risk :", "; ".join(f"{n} = {v}" for n, v in up) or "-")
    print("  Protective   :", "; ".join(f"{n} = {v}" for n, v in down) or "-")


EXAMPLES = {
    "Struggling, financial stress": dict(age=24, gpa=1.7, attendance_rate=0.55, failed_courses=4,
        credits_passed_ratio=0.45, assignments_submitted_rate=0.4, lms_logins_per_week=3,
        scholarship=0, tuition_paid_on_time=0, family_income_level=1, works_part_time=1,
        first_generation=1, commute_km=35),
    "Average student": dict(age=20, gpa=2.9, attendance_rate=0.85, failed_courses=1,
        credits_passed_ratio=0.8, assignments_submitted_rate=0.8, lms_logins_per_week=12,
        scholarship=0, tuition_paid_on_time=1, family_income_level=2, works_part_time=0,
        first_generation=0, commute_km=10),
    "Strong student": dict(age=19, gpa=3.8, attendance_rate=0.96, failed_courses=0,
        credits_passed_ratio=1.0, assignments_submitted_rate=0.98, lms_logins_per_week=18,
        scholarship=1, tuition_paid_on_time=1, family_income_level=3, works_part_time=0,
        first_generation=0, commute_km=4),
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        assess(json.load(open(sys.argv[1])))
    else:
        for name, s in EXAMPLES.items():
            print(f"\n== {name} ==")
            assess(s)
