# SHAP Stability Audit

Auditing whether SHAP-based fairness explanations stay reliable when a
protected attribute is reformatted by ordinary, non-adversarial data
engineering choices.

**The question is not "is this model biased?"** It is: can the SHAP
explanation used to answer that question be trusted in the first place?

## The finding

All nine reformatting variants clear a global stability threshold of
Spearman rho >= 0.90. But the `coarse_binary` Age variant — a median
split into two bands — drops Age's own SHAP rank from 7th to last while
its attributed importance falls 78.6%. The global metric still reports
the explanation as stable.

A single aggregate correlation can pass comfortably at the same moment
the one attribute the audit exists to examine collapses.

## Structure

```
src/config.py          paths and constants
src/data_loader.py     CSV loading with target validation
src/preprocessing.py   validation and stratified split
src/train_model.py     Random Forest with configurable seed
src/reformat.py        the nine reformatting variants
src/audit.py           SHAP computation, canonicalisation, comparison
src/eda.py             exploratory figures
src/evaluate.py        confusion matrix and ROC curve
main.py                runs the full audit end to end
validation/            robustness experiments
```

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Outputs: `data/processed/audit_results_matrix.csv`, before/after SHAP
charts in `reports/figures/shap_audit/`, EDA figures in
`reports/figures/eda/`.

## Validation experiments

```bash
python -m validation.test_seed_stability        # five random seeds
python -m validation.test_generalisation        # UCI Adult Census Income
python -m validation.test_nonprotected_control  # non-protected features
```

Seed stability: Age lands last under `coarse_binary` at all five seeds,
rho ranging 0.915–0.964, never breaching the threshold. Baseline rank
varies between 7th and 8th by seed, so the finding is stated as "7th or
8th to last".

External validation: the pattern replicates directionally on Adult, but
age ranks 1st there rather than 7th and loses only 18% — suggesting the
blind spot bites hardest on mid-ranked features, which have no redundant
signal to fall back on.

Control: the same transformation applied to five non-protected features
cut their magnitude 9–53%, and all six still cleared rho >= 0.90. The
blind spot is general, not specific to protected attributes.

## Data

Recruitment dataset: Rabie El Kharoua, *Predicting Hiring Decisions in
Recruitment Data*, Kaggle, CC BY 4.0. 1,500 rows, binary hiring outcome.

External validation uses UCI Adult Census Income (Becker and Kohavi,
1996), downloaded at runtime.

## Notes on design

The model is **retrained** for each variant rather than reused. Reusing
the baseline model and only perturbing its input would test how a fixed
model responds to out-of-distribution data — a different question. The
scenario of interest is a team encoding a column differently upstream,
training on that encoding, and reading the resulting SHAP output.

Bin edges come from `pd.qcut` quantiles rather than fixed values, so the
reformatting adapts to any distribution. Column names are arguments, not
assumptions, which is what lets the same code audit `Age` here and `age`
in the Adult dataset.