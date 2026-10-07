"""
Zepto Inventory — AI Insight Narrator
======================================
Reads key analytics results and generates plain-English executive-level
business insights using Google's Gemini API (gemini-1.5-flash, free tier).

Falls back to a rule-based insight generator if no API key is provided,
so the script always produces useful output.

Usage:
  1. Set GEMINI_API_KEY environment variable (get a free key at ai.google.dev)
  2. Run: python ai/insight_narrator.py
  3. Finds outputs from other AI scripts and summarises them

Author: Aditya
"""

import os
import sys
import json
import pandas as pd
import textwrap

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "zepto_v2.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Collect Data Signals
# ─────────────────────────────────────────────
def load_signals():
    """Pull key stats from the dataset and any saved output CSVs."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[(df["mrp"] > 0) & (df["discountedSellingPrice"] > 0)]
    df["mrp"] = df["mrp"] / 100.0
    df["discountedSellingPrice"] = df["discountedSellingPrice"] / 100.0
    df["revenue_proxy"] = df["discountedSellingPrice"] * df["availableQuantity"]

    signals = {
        "total_skus": len(df),
        "total_categories": df["Category"].nunique(),
        "out_of_stock_pct": round(df["outOfStock"].mean() * 100, 1),
        "avg_discount_pct": round(df["discountPercent"].mean(), 2),
        "highest_discount_category": (
            df.groupby("Category")["discountPercent"].mean().idxmax()
        ),
        "highest_discount_value": round(
            df.groupby("Category")["discountPercent"].mean().max(), 2
        ),
        "top_revenue_category": (
            df.groupby("Category")["revenue_proxy"].sum().idxmax()
        ),
        "top_revenue_value": round(
            df.groupby("Category")["revenue_proxy"].sum().max(), 2
        ),
        "most_oos_category": (
            df[df["outOfStock"] == True].groupby("Category").size().idxmax()
            if df["outOfStock"].any() else "N/A"
        ),
        "avg_mrp": round(df["mrp"].mean(), 2),
        "most_stocked_category": (
            df.groupby("Category")["availableQuantity"].sum().idxmax()
        ),
    }

    # Augment with risk scores if available
    risk_path = os.path.join(OUTPUT_DIR, "stockout_risk_scores.csv")
    if os.path.exists(risk_path):
        risk_df = pd.read_csv(risk_path)
        signals["high_risk_products"] = int((risk_df["risk_label"] == "High Risk").sum())
        signals["top_risk_product"] = risk_df.iloc[0]["name"]

    # Augment with elasticity if available
    elast_path = os.path.join(OUTPUT_DIR, "price_elasticity_results.csv")
    if os.path.exists(elast_path):
        e_df = pd.read_csv(elast_path).sort_values("elasticity_coeff", ascending=False)
        signals["most_elastic_category"] = e_df.iloc[0]["category"]
        signals["most_inelastic_category"] = e_df.iloc[-1]["category"]

    return signals, df


# ─────────────────────────────────────────────
# 2. Gemini API Insight Generation
# ─────────────────────────────────────────────
def generate_with_gemini(signals: dict) -> str:
    """Call Gemini API if key is available."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
You are a Senior Business Analyst at a quick-commerce company.
Based on the following data signals from Zepto's inventory dataset, 
write a concise executive-level insight report (4-5 paragraphs).
Focus on actionable business recommendations.

Data Signals:
{json.dumps(signals, indent=2)}

Format:
- Start with an executive summary sentence
- Cover: inventory health, pricing strategy, category performance, risk areas
- End with 3 specific, numbered recommendations
- Use ₹ for Indian Rupees, % for percentages
- Keep the tone professional and data-driven
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️  Gemini API call failed: {e}")
        return None


# ─────────────────────────────────────────────
# 3. Rule-Based Fallback Narrator
# ─────────────────────────────────────────────
def generate_rule_based(signals: dict) -> str:
    """Generate insights without an API key using signal thresholds."""
    oos = signals["out_of_stock_pct"]
    avg_disc = signals["avg_discount_pct"]
    top_cat = signals["top_revenue_category"]
    high_disc_cat = signals["highest_discount_category"]
    high_disc_val = signals["highest_discount_value"]
    total_skus = signals["total_skus"]

    oos_health = "concerning" if oos > 15 else "manageable" if oos > 8 else "healthy"
    disc_signal = (
        "aggressively discounting" if avg_disc > 20 else
        "moderately discounting" if avg_disc > 10 else
        "conservatively pricing"
    )

    insights = f"""
╔══════════════════════════════════════════════════════════════════════╗
║         ZEPTO INVENTORY — EXECUTIVE INSIGHT REPORT                  ║
╚══════════════════════════════════════════════════════════════════════╝

📊 EXECUTIVE SUMMARY
Zepto's inventory spans {total_skus:,} SKUs across {signals['total_categories']} product 
categories. With a stockout rate of {oos}%, inventory health is {oos_health}. 
The platform is {disc_signal} at an average of {avg_disc}% across all categories.

📦 INVENTORY HEALTH
• {oos}% of SKUs are currently out of stock — {"immediate restocking recommended" if oos > 15 else "within acceptable operational range"}.
• Category most affected by stockouts: {signals['most_oos_category']}.
• {signals['most_stocked_category']} leads in available inventory depth.
{"• ⚠️  " + str(signals.get('high_risk_products','N/A')) + " products flagged as HIGH RISK for stockout by the ML model." if 'high_risk_products' in signals else ""}

💰 PRICING & DISCOUNT STRATEGY
• {high_disc_cat} offers the highest average discount at {high_disc_val}%.
• Average platform-wide discount: {avg_disc}% — {"scope exists to reduce discounts on inelastic categories." if avg_disc > 20 else "discount strategy appears balanced."}
{"• Most discount-responsive category: " + signals['most_elastic_category'] if 'most_elastic_category' in signals else ""}
{"• Least price-sensitive category: " + signals['most_inelastic_category'] if 'most_inelastic_category' in signals else ""}

📈 REVENUE PERFORMANCE
• {top_cat} is the top revenue-generating category (₹{signals['top_revenue_value']:,.0f} estimated revenue).
• Average product MRP: ₹{signals['avg_mrp']}.

🎯 RECOMMENDATIONS
1. Restock {signals['most_oos_category']} immediately — highest stockout concentration
   represents direct lost revenue opportunity.
2. Re-evaluate discount depth in {high_disc_cat} ({high_disc_val}% avg) —
   test price floor experiments to protect margins without impacting conversion.
3. {"Prioritize ML-flagged high-risk products (" + str(signals.get('high_risk_products','')) + " SKUs) for supply chain alerts to prevent future stockouts." if 'high_risk_products' in signals else "Implement real-time inventory alerts for categories with >10% stockout rates."}
"""
    return textwrap.dedent(insights)


# ─────────────────────────────────────────────
# 4. Main Runner
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Zepto AI Insight Narrator")
    print("─" * 50)

    signals, df = load_signals()
    print(f"📊 Signals collected: {len(signals)} data points\n")

    # Try Gemini first, fallback to rule-based
    narrative = generate_with_gemini(signals)
    source = "Gemini 1.5 Flash"

    if not narrative:
        print("ℹ️  No GEMINI_API_KEY found — using rule-based narrator.\n")
        print("   (Set GEMINI_API_KEY env variable for AI-generated insights)\n")
        narrative = generate_rule_based(signals)
        source = "Rule-Based Narrator"

    print(narrative)
    print(f"\n[Generated by: {source}]")

    # Save to file
    out_path = os.path.join(OUTPUT_DIR, "insights_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Zepto Inventory — AI Insight Report\n\n")
        f.write(f"*Generated by: {source}*\n\n")
        f.write(narrative)

    print(f"\n✅ Insights saved → {out_path}")
