"""
Zepto Inventory — Price Elasticity & Discount Impact Analysis
=============================================================
Models the relationship between discount percentage and sales volume
proxy (availableQuantity × discountedSellingPrice) using polynomial
regression to identify price-elastic vs inelastic product categories.

Key outputs:
  - Category-level elasticity coefficients
  - Revenue vs Discount scatter with regression fits
  - Elasticity ranking bar chart

Author: Aditya
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "zepto_v2.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

df = df[(df["mrp"] > 0) & (df["discountedSellingPrice"] > 0)].copy()
df["mrp"] = df["mrp"] / 100.0
df["discountedSellingPrice"] = df["discountedSellingPrice"] / 100.0

# Proxy for sales volume: revenue proxy = price × available stock
df["revenue_proxy"] = df["discountedSellingPrice"] * df["availableQuantity"]
df["log_revenue"] = np.log1p(df["revenue_proxy"])

# ─────────────────────────────────────────────
# 2. Category-Level Elasticity Estimation
# ─────────────────────────────────────────────
categories = df["Category"].dropna().unique()
results = []

for cat in sorted(categories):
    subset = df[df["Category"] == cat].dropna(subset=["discountPercent", "revenue_proxy"])
    if len(subset) < 10:
        continue

    X = subset[["discountPercent"]].values
    y = subset["log_revenue"].values

    # Polynomial degree-2 regression
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Elasticity proxy: slope of revenue w.r.t. discount at mean discount
    mean_disc = X.mean()
    delta = 1.0  # 1% increase in discount
    y_at_mean = model.predict([[mean_disc]])[0]
    y_at_mean_plus = model.predict([[mean_disc + delta]])[0]
    elasticity = (y_at_mean_plus - y_at_mean) / y_at_mean  # % change in log-revenue

    results.append({
        "category": cat,
        "n_skus": len(subset),
        "avg_discount": subset["discountPercent"].mean().round(2),
        "avg_revenue_proxy": subset["revenue_proxy"].mean().round(2),
        "elasticity_coeff": round(elasticity, 4),
        "r2_score": round(r2, 3),
    })

elasticity_df = pd.DataFrame(results).sort_values("elasticity_coeff", ascending=False)
print("── Price Elasticity by Category ──────────────────────────────")
print(elasticity_df.to_string(index=False))

# ─────────────────────────────────────────────
# 3. Visualisations
# ─────────────────────────────────────────────
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("Zepto — Price Elasticity & Discount Impact Analysis",
             fontsize=16, fontweight="bold", color="white")

# 3a. Elasticity Ranking
colors = ["#FF6B6B" if e > 0 else "#4ECDC4" for e in elasticity_df["elasticity_coeff"]]
elasticity_df.set_index("category")["elasticity_coeff"].sort_values().plot(
    kind="barh", ax=axes[0], color=colors[::-1]
)
axes[0].axvline(0, color="white", lw=1, linestyle="--")
axes[0].set_title("Elasticity Coefficient by Category\n(+ve = discount boosts revenue)", color="white")
axes[0].set_xlabel("Elasticity (Δ log-revenue per 1% discount)", color="white")
axes[0].tick_params(colors="white", labelsize=8)

# 3b. Discount % vs Revenue Proxy scatter (top 6 categories)
top_cats = elasticity_df.head(6)["category"].tolist()
palette = cm.tab10(np.linspace(0, 1, len(top_cats)))
for i, cat in enumerate(top_cats):
    subset = df[df["Category"] == cat]
    axes[1].scatter(
        subset["discountPercent"],
        np.log1p(subset["revenue_proxy"]),
        s=15, alpha=0.5, color=palette[i], label=cat
    )
axes[1].set_title("Discount % vs Log-Revenue Proxy\n(Top 6 Categories)", color="white")
axes[1].set_xlabel("Discount %", color="white")
axes[1].set_ylabel("Log(Revenue Proxy)", color="white")
axes[1].legend(fontsize=7, facecolor="#222", labelcolor="white")

# 3c. Average Discount vs Average Revenue Bubble Chart
bubble = elasticity_df.copy()
sizes = (bubble["n_skus"] / bubble["n_skus"].max() * 800).values
scatter = axes[2].scatter(
    bubble["avg_discount"],
    bubble["avg_revenue_proxy"],
    s=sizes,
    c=bubble["elasticity_coeff"],
    cmap="RdYlGn",
    alpha=0.85,
    edgecolors="white",
    linewidths=0.5,
)
plt.colorbar(scatter, ax=axes[2], label="Elasticity Coefficient")
for _, row in bubble.iterrows():
    axes[2].annotate(
        row["category"].split(" ")[0],
        (row["avg_discount"], row["avg_revenue_proxy"]),
        fontsize=6, color="white", ha="center"
    )
axes[2].set_title("Discount vs Revenue by Category\n(Bubble size = SKU count)", color="white")
axes[2].set_xlabel("Avg Discount %", color="white")
axes[2].set_ylabel("Avg Revenue Proxy (₹)", color="white")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "price_elasticity.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
print(f"\n✅ Plot saved → {out_path}")
plt.close()

# Save results
elasticity_df.to_csv(os.path.join(OUTPUT_DIR, "price_elasticity_results.csv"), index=False)
print("✅ Elasticity results saved → outputs/price_elasticity_results.csv")
