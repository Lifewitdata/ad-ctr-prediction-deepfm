-- Category frequency and CTR for every categorical column, with a
-- minimum-support filter so low-count categories are not ranked by chance.
WITH per_category AS (
    SELECT 'C1' AS feature, C1 AS category_id, COUNT(*) AS n, SUM(label) AS clicks FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C2', C2, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C3', C3, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C4', C4, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C5', C5, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C6', C6, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C7', C7, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C8', C8, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C9', C9, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C10', C10, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C11', C11, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C12', C12, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C13', C13, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C14', C14, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C15', C15, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C16', C16, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C17', C17, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C18', C18, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C19', C19, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C20', C20, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C21', C21, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C22', C22, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C23', C23, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C24', C24, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C25', C25, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C26', C26, COUNT(*), SUM(label) FROM impressions GROUP BY 1, 2
)
SELECT
    feature,
    category_id,
    n,
    clicks,
    clicks * 1.0 / n AS ctr
FROM per_category
WHERE n >= $min_support
ORDER BY n DESC
LIMIT 100;
