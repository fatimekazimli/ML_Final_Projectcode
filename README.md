# MLE-AI-201 Final Project — bina.az price prediction from scratch

A decision tree and a Pegasos SVM written in NumPy, applied to bina.az sale
listings and benchmarked against scikit-learn.

- **Task A — regression:** predict sale price (trained on `log(price)`, scored in AZN).
- **Task B — classification:** premium vs standard tier, split at the training-set median price.

## Reproduce everything (one command)

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 or 3.12
pip install -r requirements.txt
# put the dataset at data/bina_az_sale.csv (see data/README.md)
python -m src.run_all          # ~5–10 min: every number, table and figure
python -m pytest -q tests/     # unit tests (tree, SVM, metrics, data prep)
cd report && pdflatex report.tex && pdflatex report.tex
```

`run_all` writes figures to `report/figures/`, CSV/JSON results to `results/`,
and `report/numbers.tex` + `report/results_table.tex`, which the report reads,
so no number in the report is typed by hand. All randomness uses seed 42.

## Layout

| Path | What | Owner |
|---|---|---|
| `src/data_prep.py` | load, clean (dedup scrapes, AZN, parsing, bounds), leakage-free split + encoding | Fatime |
| `src/evaluate.py` | RMSE, MAE, R², confusion matrix, precision/recall/F1, ROC-AUC, `compare()` | Fatime |
| `src/eda.py` | EDA figures (train split only) | Fatime |
| `src/run_all.py` | the one-command pipeline and final tables | Fatime |
| `src/decision_tree.py` | NumPy decision tree (Gini / entropy / MSE) | Ilaha |
| `src/tree_experiments.py` | tree tuning, matched sklearn baseline, error analysis | Ilaha |
| `src/svm.py` | NumPy Pegasos SVM + random Fourier features | Yunis |
| `src/svm_experiments.py` | SVM tuning, margin/SV analysis, CV benchmark for all models | Yunis |
| `tests/` | tests against scikit-learn and hand-checked examples | all |
| `report/` | IEEE report (`report.tex` + section files) | all |
| `presentation/` | slides PDF | all |

`python -m src.tree_experiments` still runs the tree part on its own.

## Rules we follow

- The tree and the SVM use NumPy only; scikit-learn is used for baselines and tests.
- Target-derived columns, identifiers, URLs and free text are excluded (explicit feature allow-list).
- The split is made first; imputation medians, top districts, one-hot columns, scaling and the tier threshold come from the training split only.
- Hyperparameters are chosen on the validation split; the test split is scored once.
