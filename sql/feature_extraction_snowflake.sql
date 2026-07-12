-- Point-in-time feature extraction for marketplace risk scoring
--
-- Run inside Snowflake (MARKETPLACE.ANALYTICS schema).
-- Each row sees only history BEFORE its own event_date — no leakage.
-- Result snapshot is pulled to Python (via snowflake_data.extract_features)
-- and fed directly into the same train pipeline used locally.

SELECT
    t.transaction_id,
    t.event_date,
    t.ticket_price,
    t.quantity,
    b.buyer_age,
    t.venue_type,
    t.event_category,
    t.section_code,

    -- Seller dispute rate: only rows strictly BEFORE this transaction
    AVG(CASE WHEN t.is_disputed THEN 1.0 ELSE 0.0 END) OVER (
        PARTITION BY t.seller_id
        ORDER BY t.event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS seller_prior_dispute_rate,

    -- Buyer purchase count: rows strictly BEFORE this transaction
    COUNT(*) OVER (
        PARTITION BY t.buyer_id
        ORDER BY t.event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS buyer_prior_purchases,

    t.is_high_risk AS label

FROM MARKETPLACE.ANALYTICS.TRANSACTIONS t
JOIN MARKETPLACE.ANALYTICS.BUYERS b ON b.buyer_id = t.buyer_id
ORDER BY t.event_date;
