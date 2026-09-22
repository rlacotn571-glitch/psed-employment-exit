# Predicting employment exit among wage workers with disabilities

Analysis code for:

> Kim CS, et al. *Predicting employment exit among wage workers with disabilities: an explainable machine learning study of a nationally representative panel in Korea.* Submitted to BMC Public Health, 2026.

A LightGBM, random-forest and penalised-logistic model predict one-year exit from wage employment in the Panel Survey of Employment for the Disabled (PSED-II, waves 1–8, 2016–2023), evaluated once on held-out waves 5–7, explained with SHAP, and externally validated in the Korean Disability and Life Dynamics Panel (KoDLD, waves 1–6, 2018–2023). Reporting follows TRIPOD+AI.

## What is and is not here

| Included | Not included (licence) |
|---|---|
| Every script that produced a number, table or figure in the manuscript | PSED microdata — apply at the Employment Development Institute, KEAD (https://edi.kead.or.kr) |
| Fixed hyperparameters (`config/fixed_params.json`) and the predictor list (`config/predictors_selected.csv`) | KoDLD microdata — data-use request to the Korea Disabled People's Development Institute (KODDI) |
| KoDLD → PSED variable mapping (`config/kodld_mapping_*.json`) | Intermediate person-wave files (`analysis_v3.csv`, `kodld_long.csv`), predictions and SHAP matrices, which are derived from the microdata |
| All result tables as Excel (`results/`) and the per-analysis JSON outputs | |

With the raw data in place the pipeline reproduces the manuscript end to end; the reported AUROCs were reproduced from the fixed parameters (`scripts/refit_predict.py`) before figures were drawn.

## Layout

```
scripts/     00 → 05 pipeline, sensitivity and ablation analyses, manuscript tables
figures/     figstyle.py (shared journal style) + one script per figure
config/      fixed hyperparameters, predictor list, KoDLD variable mapping
results/     result tables (xlsx) and per-analysis JSON produced by the scripts
```

## Environment

Python ≥ 3.11.

```bash
pip install -r requirements.txt
```

Scripts write to `$PSED_RESULTS` (default `./results`); set `PSED_WORK` to the folder holding the prepared analysis file if it is not the current directory.

## Reproduction

```bash
# 0–1  Variable map and person-wave analysis file from PSED raw files (SPSS) placed in ./raw
python scripts/00_build_variable_map.py --raw ./raw --out variable_map.xlsx
python scripts/01_build_panel.py --map variable_map.xlsx --raw ./raw --out analysis_v3.csv
#      (01b_build_from_long.py is the equivalent for the long-format release)

# 2  Tuning inside waves 1–4 (person-grouped 5-fold CV), one evaluation on waves 5–7,
#    calibration, DCA, top-decile capture, SHAP (overall / by disability type / by wave)
python scripts/02_model.py --data analysis_v3.csv --outdir results --primary lightgbm

# 3  Sensitivity analyses (outcome C, COVID waves excluded, severity, weighting schemes,
#    multiple imputation, age < 60, predictor-set ablation, sex-stratified re-fits)
python scripts/03_sensitivity.py --data analysis_v3.csv --params results/fixed_params.json --outdir results
python scripts/sens_lbfgs.py; python scripts/sens2.py; python scripts/sens_sex.py; python scripts/ablation.py

# 4–5  External validation in KoDLD (KODDI "variable download", long type, cp949 CSV)
python scripts/05_kodld_prepare.py --csv <KoDLD_long.csv> --outdir kodld_ready
python scripts/04_external_validation.py --psed analysis_v3.csv --kodld kodld_ready/kodld_long.csv \
    --mapping kodld_ready/kodld_mapping_with_tenure.json --params config/fixed_params.json --outdir ext_A
python scripts/04_external_validation.py ... --mapping kodld_ready/kodld_mapping_all_waves.json --outdir ext_B
python scripts/04_external_validation.py ... --max-age 64 --outdir ext_A_age64

# Tables and figures
python scripts/refit_predict.py          # predictions_val.csv, shap_val.csv from fixed parameters
python scripts/tables.py                 # Tables 1–3
python figures/fig1_flow.py; python figures/fig23.py; python figures/fig4_S1.py
python figures/fig5_external.py results; python figures/figS2.py
```

## Design rules enforced in code

| Rule | Where |
|---|---|
| Risk set = wage workers at index wave t with status observed at t+1 (adjacent waves only) | `01_build_panel.py` `apply_risk_set()`, `next_is_adjacent` |
| Leakage variables (job search, intention to leave, dismissal notice, sampling stratum) removed | `01_build_panel.py` `drop_leakage()` |
| Person-grouped cross-validation; validation waves inspected once | `02_model.py` `tune()` |
| Unweighted fitting, survey-weighted evaluation; person-clustered bootstrap CIs | `02_model.py`, `03_sensitivity.py` |
| No class-imbalance resampling (probabilities kept on the observed scale) | `02_model.py` |
| Recalibration-in-the-large and slope in the external cohort | `04_external_validation.py` |

## Figures

`figures/figstyle.py` sets the journal style used by every figure: 180 mm double-column width, 11-pt sans-serif base text, 600-dpi PNG/TIFF plus vector PDF with editable text, and a colour-blind-safe palette (navy #1F4E99 LightGBM, vermillion #C8451E logistic regression, teal #1B8A6B random forest).

## Citation

If you use this code, cite the paper above and this repository (Zenodo DOI to be added on release).

## Licence

MIT — see `LICENSE`. The licence covers the code only; the survey data remain under the terms of their providers.
