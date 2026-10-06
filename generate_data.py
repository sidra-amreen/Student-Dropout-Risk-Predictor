import numpy as np
import pandas as pd


def generate(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gpa = np.clip(rng.normal(2.9, 0.65, n), 0.5, 4.0)
    df = pd.DataFrame({
        "age": np.clip(rng.gamma(9, 2.3, n), 17, 45).round().astype(int),
        "gpa": gpa.round(2),
        "attendance_rate": np.clip(0.55 + 0.1 * gpa + rng.normal(0, 0.12, n), 0.2, 1.0).round(2),
        "failed_courses": rng.poisson(np.clip(2.2 - 0.5 * gpa, 0.05, None)),
        "credits_passed_ratio": np.clip(0.4 + 0.15 * gpa + rng.normal(0, 0.12, n), 0.1, 1.0).round(2),
        "assignments_submitted_rate": np.clip(0.5 + 0.12 * gpa + rng.normal(0, 0.15, n), 0.1, 1.0).round(2),
        "lms_logins_per_week": np.clip(rng.normal(8 + 1.5 * gpa, 4, n), 0, 30).round(1),
        "scholarship": rng.binomial(1, 0.3, n),
        "tuition_paid_on_time": rng.binomial(1, 0.8, n),
        "family_income_level": rng.choice([1, 2, 3], n, p=[0.35, 0.45, 0.2]),
        "works_part_time": rng.binomial(1, 0.35, n),
        "first_generation": rng.binomial(1, 0.4, n),
        "commute_km": np.clip(rng.exponential(12, n), 0, 80).round(1),
    })
    z = (
        -0.4
        - 1.1 * (df.gpa - 2.9)
        - 3.0 * (df.attendance_rate - 0.85)
        + 0.45 * df.failed_courses
        - 2.5 * (df.credits_passed_ratio - 0.8)
        - 1.5 * (df.assignments_submitted_rate - 0.8)
        - 0.05 * (df.lms_logins_per_week - 12)
        - 0.7 * df.scholarship
        - 1.0 * df.tuition_paid_on_time
        - 0.25 * (df.family_income_level - 2)
        + 0.35 * df.works_part_time
        + 0.3 * df.first_generation
        + 0.01 * df.commute_km
        + 0.03 * (df.age - 21)
        + rng.normal(0, 0.8, n)
    )
    df["dropout"] = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return df


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/students.csv", index=False)
    print(f"Saved data/students.csv ({len(d)} rows, dropout rate {d.dropout.mean():.1%})")
