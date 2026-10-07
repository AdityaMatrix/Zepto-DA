<div align="center">

# 🛒 Zepto Inventory Analytics

### AI-Powered Business Intelligence for Quick Commerce

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SQL](https://img.shields.io/badge/PostgreSQL-SQL%20Analysis-336791?style=for-the-badge&logo=postgresql&logoColor=white)](sql/zepto_analysis.sql)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](dashboard/zepto_dashboard.py)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](ai/)
[![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](notebooks/)

*A real-world data analytics portfolio project built on Zepto's live inventory dataset*

</div>

---

## 📌 Project Overview

This project simulates the end-to-end workflow of a **Business Analyst at a quick-commerce company** — from raw data exploration to AI-powered business insights. Built on a dataset of **3,732 SKUs** scraped from [Zepto](https://www.zeptonow.com/), one of India's fastest-growing 10-minute delivery platforms.

### What This Project Covers

| Layer | Techniques Used |
|---|---|
| **Data Engineering** | SQL schema design, data cleaning, paise→rupee conversion |
| **Exploratory Analysis** | Python (Pandas, Matplotlib, Seaborn), PostgreSQL |
| **Advanced SQL** | CTEs, Window Functions, Subqueries, RANK(), DENSE_RANK() |
| **Machine Learning** | Random Forest (stockout prediction), KMeans clustering, Polynomial Regression |
| **AI Integration** | Gemini LLM for executive insight generation |
| **Visualization** | Interactive Streamlit dashboard with Plotly |

---

## 📁 Repository Structure

```
Zepto-DA/
├── 📂 data/
│   └── zepto_v2.csv                    # Raw dataset (3,732 SKUs, 14 categories)
├── 📂 sql/
│   └── zepto_analysis.sql              # SQL: EDA + Cleaning + Business Queries + Advanced SQL
├── 📂 notebooks/
│   └── zepto_eda.ipynb                 # Python EDA notebook with visualizations
├── 📂 ai/
│   ├── stockout_predictor.py           # Random Forest stockout risk classifier
│   ├── product_clustering.py           # KMeans product segmentation + PCA
│   ├── price_elasticity.py             # Price sensitivity regression analysis
│   └── insight_narrator.py            # Gemini LLM insight generator
├── 📂 dashboard/
│   └── zepto_dashboard.py              # Streamlit interactive dashboard
├── 📂 outputs/                         # Generated charts, CSVs, reports
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

| Field | Description |
|---|---|
| `sku_id` | Unique product identifier (Synthetic PK) |
| `Category` | Product category (14 categories) |
| `name` | Product name as shown on the Zepto app |
| `mrp` | Maximum Retail Price (converted from paise to ₹) |
| `discountPercent` | Discount applied on MRP |
| `discountedSellingPrice` | Final price after discount (₹) |
| `availableQuantity` | Units in inventory |
| `weightInGms` | Product weight in grams |
| `outOfStock` | Boolean stock availability flag |
| `quantity` | Units per package |

**Source:** [Kaggle — Zepto Inventory Dataset](https://www.kaggle.com/datasets/palvinder2006/zepto-inventory-dataset)

---

## 🔧 SQL Analysis

The SQL file ([`sql/zepto_analysis.sql`](sql/zepto_analysis.sql)) covers 5 sections:

1. **Database Setup** — Schema creation and data import
2. **EDA** — Null audit, category distribution, stock status analysis
3. **Data Cleaning** — Zero-price removal, paise→rupee conversion
4. **Business Queries** — 8 targeted business intelligence queries
5. **Advanced SQL** — Window functions, CTEs, correlated subqueries

```sql
-- Category Health Scorecard (CTE + Window Function)
WITH category_stats AS (
    SELECT category,
        COUNT(*) AS total_skus,
        SUM(CASE WHEN outOfStock THEN 1 ELSE 0 END) AS out_of_stock,
        ROUND(AVG(discountPercent), 2) AS avg_discount,
        ROUND(SUM(discountedSellingPrice * availableQuantity), 2) AS est_revenue
    FROM zepto GROUP BY category
)
SELECT *, ROUND(100.0 * out_of_stock / total_skus, 1) AS stockout_rate_pct,
    RANK() OVER (ORDER BY est_revenue DESC) AS revenue_rank
FROM category_stats ORDER BY revenue_rank;
```

---

## 🤖 AI / ML Models

### 1. Out-of-Stock Risk Predictor
**File:** [`ai/stockout_predictor.py`](ai/stockout_predictor.py)
- **Algorithm:** Random Forest Classifier (200 trees, balanced class weights)
- **Target:** `outOfStock` boolean
- **Features:** MRP, discount %, available quantity, weight, category, price/gram
- **Output:** Per-product risk score (Low / Medium / High) + ROC curve + feature importance chart

### 2. Product Segmentation (KMeans Clustering)
**File:** [`ai/product_clustering.py`](ai/product_clustering.py)
- **Algorithm:** KMeans (k=4, selected via elbow method + silhouette score)
- **Segments:** Premium Products · Deep Discount Items · High-Stock Staples · Best Value
- **Output:** PCA scatter visualization + segment-tagged product CSV

### 3. Price Elasticity Analysis
**File:** [`ai/price_elasticity.py`](ai/price_elasticity.py)
- **Algorithm:** Degree-2 Polynomial Regression per category
- **Output:** Elasticity coefficient per category, bubble chart (discount vs revenue vs SKU count)

### 4. AI Insight Narrator
**File:** [`ai/insight_narrator.py`](ai/insight_narrator.py)
- Uses **Google Gemini API** to auto-generate executive-level business insight reports
- Falls back to a rule-based narrator if no API key is provided
- To use: set `GEMINI_API_KEY` environment variable (free key at [ai.google.dev](https://ai.google.dev))

---

## 📈 Interactive Dashboard

**File:** [`dashboard/zepto_dashboard.py`](dashboard/zepto_dashboard.py)

A dark-themed Streamlit dashboard with 5 tabs:
- **Overview** — Revenue by category, stock status, weight segmentation
- **Pricing & Discounts** — Discount rankings, MRP vs selling price scatter
- **Inventory & Stock** — Warehouse weight, stockout rates, high-MRP OOS products
- **AI Insights** — ML risk scores, product segments, LLM-generated narratives
- **Product Explorer** — Searchable, filterable, sortable product table

```bash
streamlit run dashboard/zepto_dashboard.py
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL (for SQL analysis)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Zepto-DA.git
cd Zepto-DA

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard
```bash
streamlit run dashboard/zepto_dashboard.py
```

### Run ML Models
```bash
# Stockout Risk Predictor
python ai/stockout_predictor.py

# Product Clustering
python ai/product_clustering.py

# Price Elasticity
python ai/price_elasticity.py

# AI Insight Report
python ai/insight_narrator.py
```

### Run EDA Notebook
```bash
jupyter notebook notebooks/zepto_eda.ipynb
```

---

## 🔑 Key Findings

- **12.1% stockout rate** — ~453 SKUs unavailable, with highest concentration in Meats & Dairy
- **Munchies & Chocolates** offer the deepest discounts (avg 25%+), while Fruits & Vegetables show the lowest discount depth
- **Packaged Food** leads estimated revenue potential at ~₹12M inventory value
- **ML model** identified 300+ SKUs as high-risk for stockout based on pricing and stock patterns
- **Price elasticity** analysis shows Beverages and Ice Cream are most responsive to discounts

---

## 🛠️ Tech Stack

`Python 3.9+` · `PostgreSQL` · `Pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit` · `Google Gemini API`

---

## 📄 License

This project is licensed under the MIT License. Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/palvinder2006/zepto-inventory-dataset).
