"""
Zepto Inventory — Out-of-Stock Risk Predictor
==============================================
Uses a Random Forest Classifier to predict whether a product is
likely to go out of stock based on pricing, discount, and weight features.

Model output:
  - Classification report (precision, recall, F1)
  - Confusion matrix plot
  - Feature importance chart
  - Risk-scored product table saved to outputs/

Author: Aditya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
import os

# ─────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "zepto_v2.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

# Clean: remove zero-price records
df = df[(df["mrp"] > 0) & (df["discountedSellingPrice"] > 0)].copy()

# Convert paise → rupees
df["mrp"] = df["mrp"] / 100.0
df["discountedSellingPrice"] = df["discountedSellingPrice"] / 100.0

# Derived features
df["savings_rs"] = df["mrp"] - df["discountedSellingPrice"]
df["price_per_gram"] = df["discountedSellingPrice"] / df["weightInGms"].replace(0, np.nan)
df["price_per_gram"].fillna(df["price_per_gram"].median(), inplace=True)

# Encode category
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["Category"].astype(str))

# Target & features
TARGET = "outOfStock"
df[TARGET] = df[TARGET].astype(bool).astype(int)

FEATURES = [
    "mrp",
    "discountPercent",
    "discountedSellingPrice",
    "availableQuantity",
    "weightInGms",
    "savings_rs",
    "price_per_gram",
    "category_encoded",
]

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"Dataset: {len(df):,} SKUs | Stockout rate: {y.mean():.1%}")

# ─────────────────────────────────────────────
# 2. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. Train Random Forest
# ─────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Cross-validation score
cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

print("\n── Model Performance ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["In Stock", "Out of Stock"]))
print(f"ROC-AUC (test):   {roc_auc_score(y_test, y_prob):.4f}")
print(f"ROC-AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Zepto — Out-of-Stock Risk Predictor", fontsize=16, fontweight="bold", color="white")

# 4a. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["In Stock", "Out of Stock"],
    yticklabels=["In Stock", "Out of Stock"],
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix", color="white")
axes[0].set_xlabel("Predicted", color="white")
axes[0].set_ylabel("Actual", color="white")

# 4b. Feature Importance
importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
importances.plot(kind="barh", ax=axes[1], color=colors)
axes[1].set_title("Feature Importance", color="white")
axes[1].set_xlabel("Importance Score", color="white")
axes[1].tick_params(colors="white")

# 4c. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[2].plot(fpr, tpr, color="#00d4ff", lw=2, label=f"AUC = {auc:.3f}")
axes[2].plot([0, 1], [0, 1], "w--", lw=1, label="Random Classifier")
axes[2].fill_between(fpr, tpr, alpha=0.15, color="#00d4ff")
axes[2].set_title("ROC Curve", color="white")
axes[2].set_xlabel("False Positive Rate", color="white")
axes[2].set_ylabel("True Positive Rate", color="white")
axes[2].legend(facecolor="#222", labelcolor="white")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "stockout_predictor.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
print(f"\n✅ Plot saved → {out_path}")
plt.show()

# ─────────────────────────────────────────────
# 5. Score All Products & Save Risk Table
# ─────────────────────────────────────────────
df["stockout_risk_score"] = clf.predict_proba(X.fillna(0))[:, 1]
df["risk_label"] = pd.cut(
    df["stockout_risk_score"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

risk_table = (
    df[["name", "Category", "mrp", "discountPercent", "availableQuantity",
        "stockout_risk_score", "risk_label"]]
    .sort_values("stockout_risk_score", ascending=False)
)

risk_table.to_csv(os.path.join(OUTPUT_DIR, "stockout_risk_scores.csv"), index=False)
print("\nTop 10 High-Risk Products:")
print(risk_table.head(10).to_string(index=False))
print(f"\n✅ Full risk table saved → outputs/stockout_risk_scores.csv")
