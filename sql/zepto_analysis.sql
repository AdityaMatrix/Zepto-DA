-- =============================================================================
-- ZEPTO INVENTORY DATA ANALYSIS
-- Platform: PostgreSQL
-- Dataset: 3,732 SKUs across 14 product categories scraped from Zepto
-- Author: Aditya
-- =============================================================================

-- =============================================================================
-- SECTION 1: DATABASE & TABLE SETUP
-- =============================================================================

DROP TABLE IF EXISTS zepto;

CREATE TABLE zepto (
    sku_id                 SERIAL PRIMARY KEY,
    category               VARCHAR(120),
    name                   VARCHAR(150) NOT NULL,
    mrp                    NUMERIC(8, 2),
    discountPercent        NUMERIC(5, 2),
    availableQuantity      INTEGER,
    discountedSellingPrice NUMERIC(8, 2),
    weightInGms            INTEGER,
    outOfStock             BOOLEAN,
    quantity               INTEGER
);

-- Import via pgAdmin GUI, or use psql:
-- \copy zepto(category,name,mrp,discountPercent,availableQuantity,
--             discountedSellingPrice,weightInGms,outOfStock,quantity)
-- FROM 'data/zepto_v2.csv'
-- WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ENCODING 'UTF8');


-- =============================================================================
-- SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
-- =============================================================================

-- Total record count
SELECT COUNT(*) AS total_skus FROM zepto;

-- Sample rows to understand structure
SELECT * FROM zepto LIMIT 10;

-- Null value audit across all columns
SELECT
    SUM(CASE WHEN name               IS NULL THEN 1 ELSE 0 END) AS null_name,
    SUM(CASE WHEN category           IS NULL THEN 1 ELSE 0 END) AS null_category,
    SUM(CASE WHEN mrp                IS NULL THEN 1 ELSE 0 END) AS null_mrp,
    SUM(CASE WHEN discountPercent    IS NULL THEN 1 ELSE 0 END) AS null_discountPercent,
    SUM(CASE WHEN discountedSellingPrice IS NULL THEN 1 ELSE 0 END) AS null_dsp,
    SUM(CASE WHEN weightInGms        IS NULL THEN 1 ELSE 0 END) AS null_weight,
    SUM(CASE WHEN availableQuantity  IS NULL THEN 1 ELSE 0 END) AS null_qty,
    SUM(CASE WHEN outOfStock         IS NULL THEN 1 ELSE 0 END) AS null_oos
FROM zepto;

-- Distinct product categories
SELECT DISTINCT category FROM zepto ORDER BY category;

-- In-stock vs out-of-stock count
SELECT
    outOfStock,
    COUNT(sku_id) AS product_count,
    ROUND(100.0 * COUNT(sku_id) / SUM(COUNT(sku_id)) OVER (), 1) AS pct_share
FROM zepto
GROUP BY outOfStock;

-- Products appearing as multiple SKUs (same name, different pack sizes or variants)
SELECT name, COUNT(sku_id) AS sku_count
FROM zepto
GROUP BY name
HAVING COUNT(sku_id) > 1
ORDER BY sku_count DESC
LIMIT 20;

-- Category-level summary: SKU count, avg MRP, avg discount
SELECT
    category,
    COUNT(sku_id)                       AS total_skus,
    ROUND(AVG(mrp), 2)                  AS avg_mrp,
    ROUND(AVG(discountPercent), 2)      AS avg_discount_pct,
    SUM(CASE WHEN outOfStock THEN 1 ELSE 0 END) AS out_of_stock_count
FROM zepto
GROUP BY category
ORDER BY total_skus DESC;


-- =============================================================================
-- SECTION 3: DATA CLEANING
-- =============================================================================

-- Identify zero-price records
SELECT * FROM zepto WHERE mrp = 0 OR discountedSellingPrice = 0;

-- Remove zero-MRP records (invalid data)
DELETE FROM zepto WHERE mrp = 0;

-- Convert prices from paise to rupees
UPDATE zepto
SET
    mrp                    = mrp / 100.0,
    discountedSellingPrice = discountedSellingPrice / 100.0;

-- Verify conversion
SELECT
    MIN(mrp)                    AS min_mrp,
    MAX(mrp)                    AS max_mrp,
    ROUND(AVG(mrp), 2)          AS avg_mrp,
    MIN(discountedSellingPrice) AS min_dsp,
    MAX(discountedSellingPrice) AS max_dsp
FROM zepto;


-- =============================================================================
-- SECTION 4: BUSINESS ANALYSIS QUERIES
-- =============================================================================

-- Q1. Top 10 best-value products by discount percentage
SELECT DISTINCT name, mrp, discountPercent
FROM zepto
ORDER BY discountPercent DESC
LIMIT 10;

-- Q2. High-MRP products currently out of stock (lost revenue risk)
SELECT DISTINCT name, mrp, category
FROM zepto
WHERE outOfStock = TRUE AND mrp > 300
ORDER BY mrp DESC;

-- Q3. Estimated revenue potential per category
SELECT
    category,
    ROUND(SUM(discountedSellingPrice * availableQuantity), 2) AS estimated_revenue,
    COUNT(DISTINCT name)                                       AS unique_products
FROM zepto
GROUP BY category
ORDER BY estimated_revenue DESC;

-- Q4. Premium products with minimal discounts (pricing strategy insight)
SELECT DISTINCT name, mrp, discountPercent
FROM zepto
WHERE mrp > 500 AND discountPercent < 10
ORDER BY mrp DESC, discountPercent DESC;

-- Q5. Top 5 categories by average discount percentage
SELECT
    category,
    ROUND(AVG(discountPercent), 2) AS avg_discount
FROM zepto
GROUP BY category
ORDER BY avg_discount DESC
LIMIT 5;

-- Q6. Best value-for-money products: price per gram (products above 100g)
SELECT DISTINCT
    name,
    weightInGms,
    discountedSellingPrice,
    ROUND(discountedSellingPrice / weightInGms, 2) AS price_per_gram
FROM zepto
WHERE weightInGms >= 100
ORDER BY price_per_gram
LIMIT 20;

-- Q7. Product weight segmentation: Low / Medium / Bulk
SELECT DISTINCT
    name,
    category,
    weightInGms,
    CASE
        WHEN weightInGms < 1000 THEN 'Low (< 1 kg)'
        WHEN weightInGms < 5000 THEN 'Medium (1–5 kg)'
        ELSE 'Bulk (5 kg+)'
    END AS weight_segment
FROM zepto
ORDER BY weightInGms DESC;

-- Q8. Total inventory weight per category (logistics & warehousing insight)
SELECT
    category,
    SUM(weightInGms * availableQuantity) / 1000.0 AS total_weight_kg
FROM zepto
GROUP BY category
ORDER BY total_weight_kg DESC;


-- =============================================================================
-- SECTION 5: ADVANCED SQL — WINDOW FUNCTIONS, CTEs & SUBQUERIES
-- =============================================================================

-- A. Discount rank within each category (window function)
SELECT
    name,
    category,
    discountPercent,
    RANK() OVER (PARTITION BY category ORDER BY discountPercent DESC) AS discount_rank
FROM zepto
ORDER BY category, discount_rank;

-- B. Running total of estimated revenue within each category (window function)
SELECT
    category,
    name,
    discountedSellingPrice * availableQuantity AS product_revenue,
    SUM(discountedSellingPrice * availableQuantity)
        OVER (PARTITION BY category ORDER BY discountedSellingPrice * availableQuantity DESC)
        AS running_revenue
FROM zepto
ORDER BY category, product_revenue DESC;

-- C. Category health scorecard (CTE)
WITH category_stats AS (
    SELECT
        category,
        COUNT(*)                                                          AS total_skus,
        SUM(CASE WHEN outOfStock THEN 1 ELSE 0 END)                      AS out_of_stock,
        ROUND(AVG(discountPercent), 2)                                    AS avg_discount,
        ROUND(SUM(discountedSellingPrice * availableQuantity), 2)         AS est_revenue,
        ROUND(AVG(discountedSellingPrice), 2)                             AS avg_selling_price
    FROM zepto
    GROUP BY category
)
SELECT
    *,
    ROUND(100.0 * out_of_stock / total_skus, 1) AS stockout_rate_pct,
    RANK() OVER (ORDER BY est_revenue DESC)      AS revenue_rank
FROM category_stats
ORDER BY revenue_rank;

-- D. Products priced ABOVE their category average (subquery)
SELECT
    name,
    category,
    discountedSellingPrice,
    ROUND(
        discountedSellingPrice - (
            SELECT AVG(discountedSellingPrice)
            FROM zepto sub
            WHERE sub.category = z.category
        ), 2
    ) AS premium_over_avg
FROM zepto z
WHERE discountedSellingPrice > (
    SELECT AVG(discountedSellingPrice)
    FROM zepto
    WHERE category = z.category
)
ORDER BY category, premium_over_avg DESC;

-- E. Discount effectiveness: compare discount % vs actual savings in rupees (CTE)
WITH discount_analysis AS (
    SELECT
        name,
        category,
        mrp,
        discountedSellingPrice,
        discountPercent,
        ROUND(mrp - discountedSellingPrice, 2)                AS savings_rs,
        ROUND((mrp - discountedSellingPrice) / mrp * 100, 2) AS effective_discount_pct
    FROM zepto
    WHERE mrp > 0
)
SELECT *,
    CASE
        WHEN effective_discount_pct >= 30 THEN 'High Discount'
        WHEN effective_discount_pct >= 10 THEN 'Moderate Discount'
        ELSE 'Low Discount'
    END AS discount_tier
FROM discount_analysis
ORDER BY savings_rs DESC
LIMIT 30;

-- F. Top 3 products per category by revenue (dense_rank window function)
WITH ranked AS (
    SELECT
        name,
        category,
        discountedSellingPrice * availableQuantity AS revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY discountedSellingPrice * availableQuantity DESC
        ) AS rank_in_category
    FROM zepto
)
SELECT * FROM ranked WHERE rank_in_category <= 3
ORDER BY category, rank_in_category;
