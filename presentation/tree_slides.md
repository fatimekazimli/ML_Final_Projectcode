# Decision Tree slides — Ilaha Gadirli

## Slide 3: Our Decision Tree

- NumPy implementation for price regression and premium/standard classification.
- Splits: sort each feature; test midpoints of adjacent distinct values; maximize weighted impurity decrease.
- Criteria: MSE for log(price), Gini or entropy for tier.
- Controls: max depth, minimum split size, minimum leaf size, minimum impurity decrease.
- Leaves: mean log(price), or class frequencies for `predict_proba`.
- Visual: `report/figures/tree_classification_depth.png`.

Speaker note: A deeper tree fits training rows better, but validation F1 peaks around depth 8 and then falls. The final entropy tree uses depth 8 and at least 20 rows per leaf. Explain that the 20-row choice came from validation, before looking at test.

## Slide 6: Tree results and failure modes

| Task | Our tree | Matched sklearn tree |
|---|---:|---:|
| A: test RMSE (AZN) | 389,483 | 389,485 |
| A: test MAE (AZN) | 103,067 | 103,057 |
| B: test F1 | 0.8853 | 0.8853 |
| B: test ROC-AUC | 0.9513 | 0.9513 |

- Training used the same 5,000 sampled train rows for both implementations; evaluation used all 9,406 test rows.
- 42.1% tier error on 1,040 prices within 10% of the 212,000 AZN threshold, versus 11.3% overall.
- Luxury listings: 566,294 AZN MAE for 933 cases; land plots: 714,521 AZN MAE for 279 cases.
- Visual: `report/figures/tree_regression_importance.png` or `tree_classification_depth.png`.

Speaker note: The tiny difference from sklearn is expected because both optimize the same impurity on the same features. Our Python implementation fits more slowly. Do not present the single-split metrics as cross-validation mean ± standard deviation; the team benchmark must add those separately.
