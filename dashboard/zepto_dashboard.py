"""
Zepto Inventory Analytics Dashboard
=====================================
Run: streamlit run dashboard/zepto_dashboard.py
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Path setup ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "zepto_v2.csv")
sys.path.insert(0, ROOT)

# ─── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zepto Inventory Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f0f13; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { color: #888; font-size: 0.85rem; margin-top: 4px; }

    .section-header {
        background: linear-gradient(90deg, #667eea22, transparent);
        border-left: 4px solid #667eea;
        padding: 8px 16px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
        font-weight: 600;
        font-size: 1.1rem;
        color: #e0e0e0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d4e;
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #aaa !important;
        font-size: 0.85rem;
    }

    .risk-high   { color: #ff6b6b; font-weight: 600; }
    .risk-medium { color: #ffd93d; font-weight: 600; }
    .risk-low    { color: #6bcb77; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Cache Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[(df["mrp"] > 0) & (df["discountedSellingPrice"] > 0)].copy()
    df["mrp"] = df["mrp"] / 100.0
    df["discountedSellingPrice"] = df["discountedSellingPrice"] / 100.0
    df["revenue_proxy"] = df["discountedSellingPrice"] * df["availableQuantity"]
    df["savings_rs"] = df["mrp"] - df["discountedSellingPrice"]
    df["price_per_gram"] = df["discountedSellingPrice"] / df["weightInGms"].replace(0, np.nan)
    df["weight_segment"] = pd.cut(
        df["weightInGms"],
        bins=[0, 999, 4999, 1e9],
        labels=["Low (< 1 kg)", "Medium (1–5 kg)", "Bulk (5 kg+)"],
    )
    return df

df = load_data()
ALL_CATEGORIES = sorted(df["Category"].dropna().unique())

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Zepto Analytics")
    st.markdown("*Inventory Intelligence Dashboard*")
    st.divider()

    selected_cats = st.multiselect(
        "Filter by Category",
        options=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
    )
    oos_filter = st.selectbox(
        "Stock Status",
        ["All", "In Stock Only", "Out of Stock Only"],
    )
    mrp_range = st.slider(
        "MRP Range (₹)",
        min_value=int(df["mrp"].min()),
        max_value=int(df["mrp"].max()),
        value=(int(df["mrp"].min()), int(df["mrp"].max())),
    )
    st.divider()
    st.markdown("**Dataset:** Zepto Inventory")
    st.markdown("**SKUs:** 3,732 | **Categories:** 14")
    st.markdown("**Source:** [Kaggle](https://www.kaggle.com/datasets/palvinder2006/zepto-inventory-dataset)")

# ─── Apply Filters ──────────────────────────────────────────────────────────
filtered = df[df["Category"].isin(selected_cats)].copy()
filtered = filtered[(filtered["mrp"] >= mrp_range[0]) & (filtered["mrp"] <= mrp_range[1])]
if oos_filter == "In Stock Only":
    filtered = filtered[filtered["outOfStock"] == False]
elif oos_filter == "Out of Stock Only":
    filtered = filtered[filtered["outOfStock"] == True]

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px 0;">
    <h1 style="font-size:2.4rem; font-weight:700; background:linear-gradient(135deg,#667eea,#764ba2);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        🛒 Zepto Inventory Analytics
    </h1>
    <p style="color:#777; margin-top:6px;">AI-Powered Quick Commerce Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Cards ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

metrics = [
    (k1, f"{len(filtered):,}", "Total SKUs"),
    (k2, f"{filtered['outOfStock'].mean():.1%}", "Stockout Rate"),
    (k3, f"₹{filtered['mrp'].mean():.0f}", "Avg MRP"),
    (k4, f"{filtered['discountPercent'].mean():.1f}%", "Avg Discount"),
    (k5, f"₹{filtered['revenue_proxy'].sum()/1e6:.1f}M", "Est. Revenue"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "💰 Pricing & Discounts",
    "📦 Inventory & Stock",
    "🤖 AI Insights",
    "🔍 Product Explorer",
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📈 Revenue by Category</div>', unsafe_allow_html=True)

    cat_rev = (
        filtered.groupby("Category")
        .agg(revenue=("revenue_proxy", "sum"), skus=("name", "count"))
        .reset_index()
        .sort_values("revenue", ascending=True)
    )
    fig_rev = px.bar(
        cat_rev, x="revenue", y="Category", orientation="h",
        text=cat_rev["revenue"].apply(lambda x: f"₹{x:,.0f}"),
        color="revenue",
        color_continuous_scale="Purples",
        template="plotly_dark",
    )
    fig_rev.update_traces(textposition="outside")
    fig_rev.update_layout(
        paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
        coloraxis_showscale=False, height=420,
        margin=dict(l=0, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">🍩 Stock Status Breakdown</div>', unsafe_allow_html=True)
        stock_counts = filtered["outOfStock"].value_counts().reset_index()
        stock_counts.columns = ["status", "count"]
        stock_counts["label"] = stock_counts["status"].map({True: "Out of Stock", False: "In Stock"})
        fig_pie = px.pie(
            stock_counts, values="count", names="label",
            color="label",
            color_discrete_map={"In Stock": "#6bcb77", "Out of Stock": "#ff6b6b"},
            hole=0.55,
            template="plotly_dark",
        )
        fig_pie.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13", height=300,
                              margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">⚖️ Weight Segment Distribution</div>', unsafe_allow_html=True)
        ws = filtered["weight_segment"].value_counts().reset_index()
        ws.columns = ["segment", "count"]
        fig_ws = px.bar(
            ws, x="segment", y="count", color="segment",
            color_discrete_sequence=["#667eea", "#764ba2", "#a855f7"],
            template="plotly_dark",
        )
        fig_ws.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13", height=300,
                             showlegend=False, margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig_ws, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 2: PRICING & DISCOUNTS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">🏷️ Avg Discount by Category</div>', unsafe_allow_html=True)
        disc = (
            filtered.groupby("Category")["discountPercent"]
            .mean()
            .reset_index()
            .sort_values("discountPercent", ascending=False)
        )
        fig_disc = px.bar(
            disc, x="Category", y="discountPercent",
            color="discountPercent",
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
            labels={"discountPercent": "Avg Discount %"},
        )
        fig_disc.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
                               coloraxis_showscale=False, height=360,
                               xaxis_tickangle=-35, margin=dict(l=0,r=0,t=20,b=60))
        st.plotly_chart(fig_disc, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">💸 MRP vs Selling Price (Scatter)</div>', unsafe_allow_html=True)
        sample = filtered.sample(min(800, len(filtered)), random_state=42)
        fig_scat = px.scatter(
            sample, x="mrp", y="discountedSellingPrice",
            color="Category", size="discountPercent",
            opacity=0.7, template="plotly_dark",
            labels={"mrp": "MRP (₹)", "discountedSellingPrice": "Selling Price (₹)"},
            hover_data=["name", "discountPercent"],
        )
        fig_scat.add_shape(type="line", x0=0, y0=0,
                           x1=filtered["mrp"].max(), y1=filtered["mrp"].max(),
                           line=dict(color="white", dash="dash", width=1))
        fig_scat.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
                               height=360, margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown('<div class="section-header">🏆 Top 20 Products by Discount %</div>', unsafe_allow_html=True)
    top_disc = (
        filtered.sort_values("discountPercent", ascending=False)
        .drop_duplicates("name")
        .head(20)[["name", "Category", "mrp", "discountedSellingPrice", "discountPercent"]]
        .rename(columns={
            "name": "Product", "Category": "Category",
            "mrp": "MRP (₹)", "discountedSellingPrice": "Selling Price (₹)",
            "discountPercent": "Discount %"
        })
    )
    st.dataframe(top_disc.reset_index(drop=True), use_container_width=True, height=300)

# ───────────────────────────────────────────────────────────────────────────
# TAB 3: INVENTORY & STOCK
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📦 Total Inventory Weight per Category (tonnes)</div>', unsafe_allow_html=True)
    inv = (
        filtered.groupby("Category")
        .apply(lambda x: (x["weightInGms"] * x["availableQuantity"]).sum() / 1e6)
        .reset_index(name="weight_tonnes")
        .sort_values("weight_tonnes", ascending=False)
    )
    fig_inv = px.bar(
        inv, x="Category", y="weight_tonnes",
        color="weight_tonnes",
        color_continuous_scale="Blues",
        template="plotly_dark",
        labels={"weight_tonnes": "Weight (tonnes)"},
    )
    fig_inv.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
                          coloraxis_showscale=False, height=380,
                          xaxis_tickangle=-35, margin=dict(l=0,r=0,t=20,b=60))
    st.plotly_chart(fig_inv, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🚨 Stockout Rate by Category</div>', unsafe_allow_html=True)
        oos_cat = (
            filtered.groupby("Category")["outOfStock"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .sort_values("outOfStock", ascending=False)
        )
        oos_cat.columns = ["Category", "Stockout Rate (%)"]
        fig_oos = px.bar(
            oos_cat, x="Stockout Rate (%)", y="Category", orientation="h",
            color="Stockout Rate (%)",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig_oos.update_layout(paper_bgcolor="#0f0f13", plot_bgcolor="#0f0f13",
                              coloraxis_showscale=False, height=380,
                              margin=dict(l=0,r=20,t=20,b=20))
        st.plotly_chart(fig_oos, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">💎 High-MRP Out-of-Stock Products</div>', unsafe_allow_html=True)
        high_oos = (
            filtered[(filtered["outOfStock"] == True) & (filtered["mrp"] > 300)]
            .drop_duplicates("name")
            .sort_values("mrp", ascending=False)
            .head(15)[["name", "Category", "mrp"]]
            .rename(columns={"name": "Product", "Category": "Category", "mrp": "MRP (₹)"})
        )
        st.dataframe(high_oos.reset_index(drop=True), use_container_width=True, height=380)

# ───────────────────────────────────────────────────────────────────────────
# TAB 4: AI INSIGHTS
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🤖 AI-Powered Business Insights</div>', unsafe_allow_html=True)

    # Load pre-computed risk scores if available
    risk_path = os.path.join(ROOT, "outputs", "stockout_risk_scores.csv")
    cluster_path = os.path.join(ROOT, "outputs", "product_segments.csv")
    insights_path = os.path.join(ROOT, "outputs", "insights_report.md")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        if os.path.exists(insights_path):
            with open(insights_path, encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("💡 Run `python ai/insight_narrator.py` to generate AI insights, then refresh.")
            # Show rule-based insights inline
            oos_pct = filtered["outOfStock"].mean() * 100
            top_cat = filtered.groupby("Category")["revenue_proxy"].sum().idxmax()
            high_disc_cat = filtered.groupby("Category")["discountPercent"].mean().idxmax()
            avg_disc = filtered["discountPercent"].mean()

            st.markdown(f"""
**📊 Live Snapshot Insights**

- **{oos_pct:.1f}%** of filtered SKUs are currently out of stock
- **{top_cat}** leads in estimated revenue potential
- **{high_disc_cat}** offers the deepest average discounts at **{filtered.groupby('Category')['discountPercent'].mean().max():.1f}%**
- Platform-wide average discount: **{avg_disc:.1f}%**

**🎯 Recommendations:**
1. Restock **{filtered[filtered['outOfStock']==True].groupby('Category').size().idxmax() if filtered['outOfStock'].any() else 'N/A'}** — highest stockout concentration
2. Review discount ceilings in **{high_disc_cat}** to protect margins
3. Run the ML models for predictive stockout alerts
            """)

    with col2:
        if os.path.exists(risk_path):
            st.markdown("**⚠️ High-Risk Stockout Products (ML Model)**")
            risk_df = pd.read_csv(risk_path)
            high_risk = risk_df[risk_df["risk_label"] == "High Risk"].head(10)
            for _, row in high_risk.iterrows():
                score = row["stockout_risk_score"]
                color = "#ff6b6b" if score > 0.7 else "#ffd93d"
                st.markdown(
                    f'<div style="background:#1a1a2e;border-left:3px solid {color};'
                    f'padding:8px 12px;margin:4px 0;border-radius:0 8px 8px 0;font-size:0.85rem;">'
                    f'<b>{row["name"][:40]}</b><br>'
                    f'<span style="color:#888">{row.get("Category","")}</span> · '
                    f'<span style="color:{color}">Risk: {score:.0%}</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Run `python ai/stockout_predictor.py` to generate risk scores")

        if os.path.exists(cluster_path):
            st.markdown("**🎯 Product Segment Distribution**")
            seg_df = pd.read_csv(cluster_path)
            seg_counts = seg_df["segment"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig_seg = px.pie(
                seg_counts, values="count", names="segment",
                color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF"],
                hole=0.5, template="plotly_dark",
            )
            fig_seg.update_layout(paper_bgcolor="#0f0f13", height=280, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_seg, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 5: PRODUCT EXPLORER
# ───────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔍 Interactive Product Explorer</div>', unsafe_allow_html=True)

    search = st.text_input("🔎 Search product name", placeholder="e.g. Amul, Chips, Milk...")
    sort_col = st.selectbox("Sort by", ["mrp", "discountPercent", "discountedSellingPrice", "availableQuantity"])
    sort_asc = st.checkbox("Ascending", value=False)

    explorer_df = filtered.copy()
    if search:
        explorer_df = explorer_df[explorer_df["name"].str.contains(search, case=False, na=False)]

    explorer_df = explorer_df.sort_values(sort_col, ascending=sort_asc)

    display_cols = ["name", "Category", "mrp", "discountPercent",
                    "discountedSellingPrice", "availableQuantity", "outOfStock"]
    display = explorer_df[display_cols].rename(columns={
        "name": "Product", "Category": "Category",
        "mrp": "MRP (₹)", "discountPercent": "Discount %",
        "discountedSellingPrice": "Selling Price (₹)",
        "availableQuantity": "Available Qty",
        "outOfStock": "Out of Stock",
    }).reset_index(drop=True)

    st.dataframe(
        display,
        use_container_width=True,
        height=480,
    )
    st.caption(f"Showing {len(display):,} products")

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 16px 0; color:#444; font-size:0.8rem;">
    Built with ❤️ using Python · Streamlit · Plotly · scikit-learn<br>
    Data: Zepto Inventory Dataset (Kaggle) · <a href="https://github.com/aditya/Zepto-DA" style="color:#667eea">GitHub</a>
</div>
""", unsafe_allow_html=True)
