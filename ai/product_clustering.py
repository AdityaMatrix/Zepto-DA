# -*- coding: utf-8 -*-
"""
Zepto Inventory — Product Segmentation via KMeans Clustering
=============================================================
Groups 3,700+ SKUs into strategic inventory tiers using unsupervised ML.
Clusters are visualised in 2D using PCA for dimensionality reduction.

Output segments help answer:
  - Which products are high-value slow-movers?
  - Which are budget fast-movers?
  - Where are discount-heavy items concentrated?

Author: Aditya
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import sys

# Fix Windows terminal encoding
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

# Derived features
df["savings_rs"] = df["mrp"] - df["discountedSellingPrice"]
df["price_per_gram"] = (
    df["discountedSellingPrice"] / df["weightInGms"].replace(0, np.nan)
)
df["price_per_gram"] = df["price_per_gram"].fillna(df["price_per_gram"].median())

CLUSTER_FEATURES = [
    "mrp",
    "discountPercent",
    "availableQuantity",
    "weightInGms",
    "savings_rs",
    "price_per_gram",
]

X = df[CLUSTER_FEATURES].fillna(0)

# ─────────────────────────────────────────────
# 2. Find Optimal k (Elbow + Silhouette)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 10)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=1000, random_state=42))

BEST_K = 4  # chosen after elbow inspection

# ─────────────────────────────────────────────
# 3. Fit Final KMeans
# ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Segment naming based on centroid characteristics
cluster_summary = df.groupby("cluster")[CLUSTER_FEATURES].mean()
print("\nCluster Centroids (standardised feature means):")
print(cluster_summary.round(2))

SEGMENT_NAMES = {
    cluster_summary["mrp"].idxmax(): "[Premium] High-MRP Products",
    cluster_summary["discountPercent"].idxmax(): "[Discount] Deep Discount Items",
    cluster_summary["availableQuantity"].idxmax(): "[Staples] High-Stock Staples",
    cluster_summary["price_per_gram"].idxmin(): "[Value] Best Price-per-Gram",
}
# Fill any unmapped cluster
for c in range(BEST_K):
    if c not in SEGMENT_NAMES:
        SEGMENT_NAMES[c] = f"Cluster {c}"

df["segment"] = df["cluster"].map(SEGMENT_NAMES)
print("\nSegment distribution:")
print(df["segment"].value_counts())

# ─────────────────────────────────────────────
# 4. PCA for 2D Visualisation
# ─────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df["pca_x"] = coords[:, 0]
df["pca_y"] = coords[:, 1]

explained = pca.explained_variance_ratio_
print(f"\nPCA explained variance: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%}")

# ─────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────
PALETTE = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF"]
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Zepto — Product Segmentation via KMeans Clustering",
             fontsize=16, fontweight="bold", color="white")

# 5a. Elbow plot
axes[0].plot(list(k_range), inertias, "o-", color="#FF6B6B", lw=2)
axes[0].axvline(BEST_K, color="white", linestyle="--", alpha=0.5)
axes[0].set_title("Elbow Method (Inertia)", color="white")
axes[0].set_xlabel("Number of Clusters (k)", color="white")
axes[0].set_ylabel("Inertia", color="white")

# 5b. Silhouette scores
axes[1].bar(list(k_range), silhouettes, color=PALETTE * 3)
axes[1].axvline(BEST_K, color="white", linestyle="--", alpha=0.5)
axes[1].set_title("Silhouette Score by k", color="white")
axes[1].set_xlabel("Number of Clusters (k)", color="white")
axes[1].set_ylabel("Silhouette Score", color="white")

# 5c. PCA scatter
sample = df.sample(min(2000, len(df)), random_state=42)
for i, (cluster_id, seg_name) in enumerate(SEGMENT_NAMES.items()):
    mask = sample["cluster"] == cluster_id
    axes[2].scatter(
        sample.loc[mask, "pca_x"],
        sample.loc[mask, "pca_y"],
        c=PALETTE[i % len(PALETTE)],
        s=20, alpha=0.7, label=seg_name,
    )
axes[2].set_title(
    f"Product Clusters (PCA) — {explained[0]+explained[1]:.0%} variance explained",
    color="white",
)
axes[2].set_xlabel(f"PC1 ({explained[0]:.1%})", color="white")
axes[2].set_ylabel(f"PC2 ({explained[1]:.1%})", color="white")
axes[2].legend(fontsize=8, facecolor="#222", labelcolor="white")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "product_clustering.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
print(f"\n✅ Plot saved → {out_path}")
plt.close()

# ─────────────────────────────────────────────
# 6. Segment Report
# ─────────────────────────────────────────────
report = (
    df.groupby("segment")
    .agg(
        sku_count=("name", "count"),
        avg_mrp=("mrp", "mean"),
        avg_discount=("discountPercent", "mean"),
        avg_stock=("availableQuantity", "mean"),
        oos_rate=("outOfStock", "mean"),
    )
    .round(2)
)
print("\n── Segment Report ──────────────────────────────────────")
print(report.to_string())

df[["name", "Category", "mrp", "discountPercent", "availableQuantity",
    "cluster", "segment"]].to_csv(
    os.path.join(OUTPUT_DIR, "product_segments.csv"), index=False
)
print("\n✅ Segment data saved → outputs/product_segments.csv")
