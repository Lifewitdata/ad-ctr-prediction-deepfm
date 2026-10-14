-- How much of the traffic sits in category values below a minimum-support
-- threshold, per categorical column. Motivation for the OOV vocabulary.
WITH per_category AS (
    SELECT 'C1' AS feature, C1 AS category_id, COUNT(*) AS n FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C2', C2, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C3', C3, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C4', C4, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C5', C5, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C6', C6, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C7', C7, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C8', C8, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C9', C9, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C10', C10, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C11', C11, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C12', C12, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C13', C13, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C14', C14, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C15', C15, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C16', C16, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C17', C17, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C18', C18, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C19', C19, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C20', C20, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C21', C21, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C22', C22, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C23', C23, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C24', C24, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C25', C25, COUNT(*) FROM impressions GROUP BY 1, 2
    UNION ALL SELECT 'C26', C26, COUNT(*) FROM impressions GROUP BY 1, 2
)
SELECT
    feature,
    COUNT(*)                                             AS distinct_values,
    COUNT(*) FILTER (WHERE n >= $min_support)            AS values_meeting_support,
    SUM(n)                                               AS total_impressions,
    SUM(n) FILTER (WHERE n < $min_support)               AS impressions_from_rare_values,
    ROUND(100.0 * SUM(n) FILTER (WHERE n < $min_support) / SUM(n), 3) AS pct_impressions_rare
FROM per_category
GROUP BY feature
ORDER BY feature;
