"""Additional file 1 (TRIPOD+AI checklist) and Additional file 2 (predictor list)."""
import os
import pandas as pd
from pathlib import Path
OUTD = Path(os.environ.get("PSED_RESULTS", "results"))

# ---------------- Additional file 2: predictors ----------------------------
eng = {
 "a0710": ("Primary disability type (15 statutory categories)", "Disability", "categorical"),
 "aarea": ("Region of residence (3 groups, derived)", "Demographics", "categorical"),
 "area": ("Region of residence (province)", "Demographics", "categorical"),
 "gender": ("Sex", "Demographics", "binary"),
 "birthy": ("Year of birth", "Demographics", "numeric"),
 "grade02": ("Severe vs mild disability (statutory definition, derived)", "Disability", "binary"),
 "type04": ("Disability type, 4-group (derived)", "Disability", "categorical"),
 "type06": ("Disability type, 6-group (derived)", "Disability", "categorical"),
 "emptype02": ("Employment status, 5-group (regular/temporary/daily wage worker; derived)", "Employment history", "categorical"),
 "emppme": ("Non-standard (non-regular) worker indicator (derived)", "Employment conditions", "binary"),
 "jobtype": ("Type of main job (derived)", "Job characteristics", "categorical"),
 "jobctotal": ("Number of jobs currently held (derived)", "Employment history", "numeric"),
 "aca0403": ("Tenure in current job, months (derived)", "Employment history", "numeric"),
 "ca1101": ("Job classification", "Job characteristics", "categorical"),
 "ca1201": ("Number of employees, whole firm", "Job characteristics", "categorical"),
 "ca1203": ("Number of employees, workplace", "Job characteristics", "categorical"),
 "ca1205": ("Number of employees with disabilities, workplace", "Job characteristics", "numeric"),
 "ca1301": ("Job designed for people with disabilities", "Job characteristics", "binary"),
 "ca1302": ("Seasonal / periodic job", "Job characteristics", "binary"),
 "ca1401": ("Contract period specified (fixed-term)", "Employment conditions", "binary"),
 "ca1501": ("Daily-hire work", "Employment conditions", "binary"),
 "ca1701": ("Full-time vs part-time", "Employment conditions", "binary"),
 "ca1801": ("Dispatch/agency work: who pays wages", "Employment conditions", "categorical"),
 "ca2201": ("Usual place of work", "Employment conditions", "categorical"),
 "ca2202": ("Special-type (dependent self-employed) worker", "Employment conditions", "binary"),
 "ca2301": ("Self-assessed non-regular status", "Employment conditions", "binary"),
 "ca2501": ("Working days per week fixed", "Working hours", "binary"),
 "ca2502": ("Working days per week", "Working hours", "numeric"),
 "ca2503": ("Daily working hours fixed", "Working hours", "binary"),
 "ca2504": ("Daily working hours", "Working hours", "numeric"),
 "ca2505": ("Frequency of overtime", "Working hours", "ordinal"),
 "ca2506": ("Actual days worked last month", "Working hours", "numeric"),
 "ca2507": ("Actual hours worked last week", "Working hours", "numeric"),
 "ca2508": ("Absence last month due to disability/health", "Health", "binary"),
 "ca2510": ("Shift work arrangement", "Working hours", "categorical"),
 "ca2601": ("Pay basis (monthly/daily/hourly/piece)", "Wages and benefits", "categorical"),
 "ca2603": ("Monthly income, KRW 10,000", "Wages and benefits", "numeric"),
 "ca2701": ("Unpaid wages experienced", "Wages and benefits", "binary"),
 "ca2801": ("National pension, workplace enrolment", "Wages and benefits", "binary"),
 "ca2802": ("Health insurance, workplace enrolment", "Wages and benefits", "binary"),
 "ca2803": ("Employment insurance enrolment", "Wages and benefits", "binary"),
 "ca2804": ("Industrial accident insurance enrolment", "Wages and benefits", "binary"),
 "ca2901": ("Benefit: retirement pay/pension", "Wages and benefits", "binary"),
 "ca2902": ("Benefit: bonus", "Wages and benefits", "binary"),
 "ca2903": ("Benefit: overtime pay", "Wages and benefits", "binary"),
 "ca2904": ("Benefit: paid holidays/leave", "Wages and benefits", "binary"),
 "ca2905": ("Benefit: sick leave", "Wages and benefits", "binary"),
 "ca2906": ("Benefit: maternity/parental leave", "Wages and benefits", "binary"),
 "ca2907": ("Benefit: education and training", "Wages and benefits", "binary"),
 "ca2908": ("Benefit: accident insurance", "Wages and benefits", "binary"),
 "ca4001": ("Disability known at workplace", "Workplace discrimination", "binary"),
 "ca4101": ("Discrimination at work: sex", "Workplace discrimination", "binary"),
 "ca4102": ("Discrimination at work: age", "Workplace discrimination", "binary"),
 "ca4103": ("Discrimination at work: disability", "Workplace discrimination", "binary"),
 "ca4104": ("Discrimination at work: education", "Workplace discrimination", "binary"),
 "ca4105": ("Discrimination at work: region of origin", "Workplace discrimination", "binary"),
 "ca4106": ("Discrimination at work: employment type", "Workplace discrimination", "binary"),
 "ca4107": ("Discrimination at work: career", "Workplace discrimination", "binary"),
 "ca4108": ("Discrimination at work: rank/track", "Workplace discrimination", "binary"),
 "ae0401": ("Holds a vocational certificate (cumulative)", "Human capital", "binary"),
 "ae1508": ("Number of people who could help find a job (derived)", "Human capital", "numeric"),
 "g0101": ("Self-rated health", "Health", "ordinal"),
 "g0102": ("Health compared with last year", "Health", "ordinal"),
 "g0201": ("Chronic disease other than disability", "Health", "binary"),
 "g0301": ("Exercise days per week", "Health", "numeric"),
 "g0401": ("Sleep hours per day", "Health", "numeric"),
 "g0501": ("Frequency of skipping meals", "Health", "ordinal"),
 "g0502": ("Regular meals", "Health", "binary"),
 "g0601": ("Current smoking", "Health", "binary"),
 "g0701": ("Current drinking", "Health", "binary"),
 "g0801": ("Daily-life stress", "Health", "ordinal"),
 "g0802": ("Disability-related stress", "Health", "ordinal"),
 "g0803": ("Depression in past year", "Health", "binary"),
 "g0804": ("Happiness (0–10)", "Health", "numeric"),
 "g0901": ("Need for help from others in daily life", "Health", "ordinal"),
 "aca0403_delta": ("Change in tenure from previous wave (derived)", "Derived", "numeric"),
 "ca2603_delta": ("Change in monthly income from previous wave (derived)", "Derived", "numeric"),
 "ca2507_delta": ("Change in weekly hours from previous wave (derived)", "Derived", "numeric"),
 "age": ("Age = survey year − birth year (derived)", "Derived", "numeric"),
 "wage_lag1": ("Wage worker at previous wave (derived)", "Derived", "binary"),
 "wage_lag2": ("Wage worker two waves earlier (derived)", "Derived", "binary"),
 "wage_hist2": ("Number of the two preceding waves in wage work (derived)", "Derived", "numeric"),
 "n_prior_waves": ("Number of prior waves observed (derived)", "Derived", "numeric"),
 "job_changed_last": ("Main-job identifier changed since previous wave (derived)", "Derived", "binary"),
}
for i in range(1, 11):
    eng[f"f06{i:02d}"] = (f"Rosenberg self-esteem item {i}", "Psychological scales", "ordinal (4-point)")
for i in range(1, 13):
    eng[f"f07{i:02d}"] = (f"Disability acceptance item {i}", "Psychological scales", "ordinal (5-point)")

r = pd.read_excel(OUTD / "analysis_v3.report.xlsx", sheet_name="변수사전")
r = r[~r["변수"].isin(["pid", "wave", "year", "wt01", "wt02", "y"])]
rows = []
for v, lab in zip(r["변수"], r["라벨"]):
    e = eng.get(v, ("[translate]", "[assign]", ""))
    rows.append({"Variable": v, "Survey label (Korean)": lab if isinstance(lab, str) else "(derived)",
                 "Description (English)": e[0], "Domain": e[1], "Type": e[2]})
af2 = pd.DataFrame(rows)
missing = af2[af2["Description (English)"] == "[translate]"]
print("untranslated:", missing.Variable.tolist())
print(af2.Domain.value_counts())

# ---------------- Additional file 1: TRIPOD+AI checklist --------------------
ck = [
 ("Title", "1", "Identify the study as developing and/or evaluating a prediction model, the target population, and the outcome", "Title page"),
 ("Abstract", "2", "Structured summary of objectives, methods, results and conclusions", "Abstract"),
 ("Introduction", "3a", "Background, rationale and intended use of the model", "Background, paras 1–4"),
 ("Introduction", "3b", "Objectives, including whether the study describes development and/or evaluation", "Background, para 5"),
 ("Methods: Data", "4a", "Sources of data and study design; dates of data collection", "Methods: Study design and data source"),
 ("Methods: Data", "4b", "Key dates of data collection, follow-up and prediction horizon", "Methods: Study design; Temporal validation"),
 ("Methods: Participants", "5a", "Setting and eligibility criteria", "Methods: Analytic sample"),
 ("Methods: Participants", "5b", "Details of treatments/services received, if relevant", "Not applicable (observational panel); designated jobs described as predictor"),
 ("Methods: Data preparation", "6", "Data pre-processing, recoding of special codes, derived variables", "Methods: Predictors (sentinel recoding; derived variables)"),
 ("Methods: Outcome", "7a", "Outcome definition and timing", "Methods: Outcome"),
 ("Methods: Outcome", "7b", "Blinding of outcome assessment to predictors", "Not applicable (survey self-report at t+1)"),
 ("Methods: Predictors", "8a", "Definition and measurement of all predictors", "Methods: Predictors; Additional file 2"),
 ("Methods: Predictors", "8b", "Blinding of predictor assessment to outcome", "Not applicable (predictors measured before outcome)"),
 ("Methods: Sample size", "9", "How the sample size was arrived at", "Methods: Sample size"),
 ("Methods: Missing data", "10", "Handling of missing data", "Methods: Model development (imputation within pipeline); Sensitivity analyses (multiple imputation)"),
 ("Methods: Analytical methods", "11a", "Type of model, model building, hyperparameter tuning", "Methods: Model development"),
 ("Methods: Analytical methods", "11b", "Handling of class imbalance", "Methods: Model development (AUPRC-based tuning; no resampling)"),
 ("Methods: Analytical methods", "11c", "Internal/external validation approach", "Methods: Temporal validation"),
 ("Methods: Analytical methods", "11d", "Performance measures", "Methods: Performance measures"),
 ("Methods: Analytical methods", "11e", "Model updating, if any", "Not done; Discussion: Limitations (recalibration for PSED-III)"),
 ("Methods: Fairness", "12", "Assessment of performance across subgroups", "Methods: Sensitivity analyses (sex, age, severity); Results: Performance by sex, age and severity; Table 5; Fig. S2"),
 ("Methods: Model output", "13", "Form of model output (probability, risk group)", "Methods: Performance measures (top-k flagging); Results: Utility for targeting"),
 ("Methods: Data separation", "14", "Separation of development and validation data", "Methods: Temporal validation; Fig. 1"),
 ("Methods: Ethics", "15", "Ethical approval", "Declarations"),
 ("Open science", "16a", "Funding", "Declarations"),
 ("Open science", "16b", "Competing interests", "Declarations"),
 ("Open science", "16c", "Protocol / registration", "No protocol registered; pre-specification of outcome and primary model stated in Methods"),
 ("Open science", "16d", "Data sharing", "Declarations: Availability of data"),
 ("Open science", "16e", "Code sharing", "Declarations: Availability of data (repository)"),
 ("Patient/public involvement", "17", "Involvement of people with disabilities in the study", "Not involved; stated in Declarations [add sentence]"),
 ("Results: Participants", "18a", "Flow of participants; Fig.", "Results: Sample; Fig. 1"),
 ("Results: Participants", "18b", "Characteristics of participants; development vs validation", "Results: Sample; Table 1"),
 ("Results: Participants", "18c", "Distribution of predictors and outcome in each set", "Results: Sample; Table 1; Table S1"),
 ("Results: Model development", "19a", "Full model specification (hyperparameters, preprocessing)", "Methods: Model development; fixed_params.json in repository"),
 ("Results: Model development", "19b", "How to use the model for prediction", "Discussion: Implications; code repository"),
 ("Results: Model performance", "20a", "Performance measures with CIs, overall", "Results: Discrimination and calibration; Table 2; Fig. 2"),
 ("Results: Model performance", "20b", "Performance in subgroups / fairness", "Results: Performance by sex, age and severity; Subgroup structure; Tables 4–5; Fig. 4; Fig. S2"),
 ("Results: Model updating", "21", "Results of any updating", "Not applicable"),
 ("Discussion", "22", "Interpretation of main results and comparison with previous work", "Discussion: Principal findings; Comparison with previous work"),
 ("Discussion", "23", "Limitations", "Discussion: Strengths and limitations"),
 ("Discussion", "24a", "Usability: how and by whom the model could be used", "Discussion: Implications"),
 ("Discussion", "24b", "Usability: human interaction, oversight and risks of misuse", "Discussion: Weaker performance among women and among workers with severe disabilities (deployment caution, equity)"),
 ("Discussion", "25", "Next steps: external validation, updating", "Discussion: Limitations (KoDLD external validation planned)"),
 ("Explainability (AI-specific)", "26", "Explainability methods and interpretation caveats", "Methods: Explainability; Results: Basis of prediction; Fig. 3"),
]
af1 = pd.DataFrame(ck, columns=["Section", "Item", "Checklist item (abbreviated)", "Location in manuscript"])
note = ("Checklist based on the TRIPOD+AI statement (Collins et al., BMJ 2024;385:e078378). Item wording is abbreviated and item numbering "
        "should be reconciled with the official checklist (https://www.tripod-statement.org) before submission; the two bracketed entries "
        "([add sentence], [expand]) mark places where the manuscript still needs a sentence.")

with pd.ExcelWriter(OUTD / "additional_files.xlsx", engine="openpyxl") as xw:
    af1.to_excel(xw, sheet_name="AF1_TRIPOD_AI", index=False)
    pd.DataFrame({"note": [note]}).to_excel(xw, sheet_name="AF1_note", index=False)
    af2.to_excel(xw, sheet_name="AF2_predictors", index=False)
print("written", len(af1), len(af2))
