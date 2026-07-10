-- Snowflake SQL template: leakage-safe historical features for marketplace risk scoring
-- Replace table names with your warehouse schema. In production, run this in Snowflake
-- and export to a stage / Databricks for model training.

SELECT
    t.transaction_id,
    t.event_date,
    t.ticket_price,
    t.quantity,
    b.buyer_age,
    t.venue_type,
    t.event_category,
    t.section_code,

    -- Only history BEFORE this transaction (no future leakage)
    -- is_disputed = whether that past transaction was disputed (not the training label)
    AVG(CASE WHEN t.is_disputed THEN 1 ELSE 0 END) OVER (
        PARTITION BY t.seller_id
        ORDER BY t.event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS seller_prior_dispute_rate,

    COUNT(*) OVER (
        PARTITION BY t.buyer_id
        ORDER BY t.event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS buyer_prior_purchases,

    t.is_high_risk AS label

FROM marketplace.transactions t
JOIN marketplace.buyers b ON b.buyer_id = t.buyer_id
WHERE t.event_date >= '2024-01-01'
  AND t.event_date < CURRENT_DATE();

-- Databricks note: load result into a Delta table, then train with MLflow on a cluster.
-- Example (Python):
--   df = spark.read.table("marketplace.features_training")
--   # feature engineering + XGBoost in a Databricks notebook using the same src/ modules
