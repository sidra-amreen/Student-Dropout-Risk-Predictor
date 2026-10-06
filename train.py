import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, LABELS
from generate_data import generate

DATA = "data/students.csv"


def main():
    if not os.path.exists(DATA):
        os.makedirs("data", exist_ok=True)
        generate().to_csv(DATA, index=False)
        print("Generated synthetic dataset -> data/students.csv")
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["dropout"]
    print(f"Students: {len(df)} | dropout rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    fitted, best_name, best_auc = {}, None, -1
    print("\n5-fold CV ROC-AUC (train set) | test ROC-AUC")
    for name, m in models.items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        m.fit(X_tr, y_tr)
        te_auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        fitted[name] = m
        print(f"  {name:<20} {cv_auc:.3f} | {te_auc:.3f}")
        if cv_auc > best_auc:
            best_name, best_auc = name, cv_auc

    best = fitted[best_name]
    proba = best.predict_proba(X_te)[:, 1]
    threshold = 0.4
    pred = (proba >= threshold).astype(int)
    print(f"\nBest model: {best_name} (decision threshold = {threshold})\n")
    print(classification_report(y_te, pred, target_names=["stayed", "dropped out"]))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, m in fitted.items():
        RocCurveDisplay.from_estimator(m, X_te, y_te, ax=ax[0], name=name)
    ax[0].set_title("ROC curves")
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["stayed", "dropped"], cmap="Blues", ax=ax[1])
    ax[1].set_title(f"Confusion matrix - {best_name}")
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=42)
    order = np.argsort(imp.importances_mean)
    ax[2].barh([LABELS[FEATURES[i]] for i in order], imp.importances_mean[order], color="#4C78A8")
    ax[2].set_title("What drives dropout risk (permutation importance)")
    plt.tight_layout()
    plt.savefig("results.png", dpi=120)

    explainer = fitted["Logistic Regression"]
    joblib.dump({"model": best, "explainer": explainer, "threshold": threshold, "name": best_name}, "dropout_model.joblib")
    print("Saved dropout_model.joblib and results.png")


if __name__ == "__main__":
    main()
