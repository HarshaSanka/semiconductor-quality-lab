-- sqlite3 artifacts/research.sqlite < scripts/query_results.sql
SELECT experiment, ROUND(cv_ap_mean, 4) AS cv_average_precision,
       ROUND(cv_ap_std, 4) AS fold_standard_deviation
FROM experiments ORDER BY cv_ap_mean DESC;

SELECT label AS actual_failure, prediction AS predicted_failure, COUNT(*) AS runs
FROM holdout_predictions GROUP BY label, prediction ORDER BY label, prediction;
