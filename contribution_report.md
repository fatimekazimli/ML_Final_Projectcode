# Contribution Report

| Member (Surname, Name) | GitHub handle | Main responsibilities |
|---|---|---|
| Allahverdi, Yunis | Yunis-Allahverdi | NumPy Pegasos SVM + random Fourier features, SVM tests, SVM tuning and margin/support-vector analysis, cross-validated benchmark of all models, SVM report section, slides. |
| Gadirli, Ilaha | ii1ahe | NumPy Decision Tree, toy tests, tree tuning, matched sklearn benchmark, feature importance, error analysis, report and slide content, final real-data run. |
| Kazimli, Fatime | fatimekazimli | Data cleaning and leakage-free split (`data_prep.py`), metrics (`evaluate.py`), EDA figures, `run_all.py` integration, data/metric tests, README, report assembly. |

## Per-member summary

**Allahverdi, Yunis —** Yunis Allahverdi implemented the soft-margin SVM in `src/svm.py`: the Pegasos sub-gradient solver with the 1/(λt) schedule, projection onto the 1/√λ ball, epoch averaging, margin and support-vector reporting, and the random-Fourier-feature extension. He wrote `tests/test_svm.py` (9 tests, including one showing the solver reaches the same objective as scikit-learn's `LinearSVC`) and `src/svm_experiments.py`, which runs the λ and RFF sweeps, the convergence and margin figures, the weight and error analysis, and the 5-fold cross-validated benchmark with timings for all models against matched scikit-learn baselines. He also wrote `report/svm_section.tex` and the slides. His code is in commit `f2af1e7`. While checking the real-data run he found the RFF model's decision threshold was unstable between runs and fixed it with epoch averaging. AI assistance (Claude) was used for parts of this code and text and is disclosed in the report.

**Gadirli, Ilaha —** Ilaha Gadirli implemented `src/decision_tree.py` and verified its split thresholds and predictions against scikit-learn on deterministic toy data in `tests/test_tree.py`. She built the reproducible tree experiment in `src/tree_experiments.py`, prepared its figures and metrics, and drafted `report/tree_section.tex` and the tree slides in `presentation/tree_slides.tex`. These changes are in [PR #1](https://github.com/fatimekazimli/ML_Final_Projectcode/pull/1), commits `ee768ec`, `77f63b6`, and `60a7769`. In [PR #2](https://github.com/fatimekazimli/ML_Final_Projectcode/pull/2) (commit `590b4da`) she switched the tree experiment to the shared data pipeline, updated her section to the final numbers, and committed the final real-data run of `python -m src.run_all`. AI assistance (OpenAI Codex) was used and is disclosed in the report.

**Kazimli, Fatime —** Fatime Kazimli built the data pipeline in `src/data_prep.py` (deduplication of repeated scrapes, AZN conversion, parsing of area/floor/land, fixed outlier bounds, allow-list features so target-derived columns are excluded, stratified split, train-only imputation and encoding), the NumPy metrics in `src/evaluate.py` with tests against scikit-learn, the EDA figures in `src/eda.py`, and the one-command pipeline `src/run_all.py` that writes every table and number used in the report. She assembled `report/report.tex` and wrote the data section. The final pipeline reuses the cleaning rules Ilaha verified on the real CSV; AI assistance (Claude) was used for parts of this code and text and is disclosed in the report.

## Rough effort split

| Member | Approx. % |
|---|---|
| Allahverdi, Yunis | 34 |
| Gadirli, Ilaha | 33 |
| Kazimli, Fatime | 33 |

Each member owned one module end to end (data and integration, decision tree, SVM and benchmark) plus their report section and slides, so the split is roughly equal.