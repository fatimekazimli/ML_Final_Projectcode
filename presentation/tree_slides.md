# Decision Tree slides — Ilaha Gadirli

## Slide 3: Our Decision Tree

- NumPy implementation for price regression and premium/standard classification.
- Splits: sort each feature; test midpoints of adjacent distinct values; maximize weighted impurity decrease.
- Criteria: MSE for log(price), Gini or entropy for tier.
- Controls: max depth, minimum split size, minimum leaf size, minimum impurity decrease.
- Leaves: mean log(price), or class frequencies for `predict_proba`.
- Visual: `report/figures/tree_classification_depth.png`.

Speaker note: A deeper tree fits training rows better, but validation F1 eventually falls. The final Gini tree uses depth 9 and at least 1 row per leaf. Explain that this choice came from validation, before looking at test.

## Slide 6: Tree results and failure modes

| Task | Our tree | Matched sklearn tree |
|---|---:|---:|
| A: test RMSE (AZN) | 391,531 | 391,531 |
| A: test MAE (AZN) | 99,840 | 99,840 |
| B: test F1 | 0.8952 | 0.8956 |
| B: test ROC-AUC | 0.9251 | 0.9246 |

- Training used the same 5,000 sampled train rows for both implementations; evaluation used all 9,406 test rows.
- 35.9% tier error on 1,053 prices within 10% of the 212,000 AZN threshold, versus 10.6% overall.
- Luxury listings: 547,009 AZN MAE for 939 cases; land plots: 663,872 AZN MAE for 275 cases.
- Visual: `report/figures/tree_regression_importance.png` or `tree_classification_depth.png`.

Speaker note: The small difference from sklearn is expected because both optimize the same impurity on the same features but can break ties differently. Our Python implementation fits more slowly. Do not present the single-split metrics as cross-validation mean ± standard deviation; the team benchmark reports those separately.
