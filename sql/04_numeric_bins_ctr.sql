-- CTR across decile bins for two numeric features with different
-- distributions (I1 has ~45% missing, I2 is mostly present).
WITH binned AS (
    SELECT 'I1' AS feature, I1 AS value, label, NTILE(10) OVER (ORDER BY I1) AS decile
    FROM impressions WHERE I1 IS NOT NULL
    UNION ALL
    SELECT 'I2', I2, label, NTILE(10) OVER (ORDER BY I2)
    FROM impressions WHERE I2 IS NOT NULL
)
SELECT
    feature,
    decile,
    COUNT(*)      AS n,
    MIN(value)    AS min_value,
    MAX(value)    AS max_value,
    AVG(value)    AS avg_value,
    SUM(label)    AS clicks,
    AVG(label)    AS ctr
FROM binned
GROUP BY feature, decile
ORDER BY feature, decile;
