-- Overall impression and click counts with the observed CTR.
SELECT
    COUNT(*)   AS impressions,
    SUM(label) AS clicks,
    AVG(label) AS ctr
FROM impressions;
