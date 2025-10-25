# Stalking Non-Reporting Project

This repository contains an end‑to‑end workflow to (1) cluster stalking victims who **did not report** their incidents to police based on *reasons for non‑reporting*, (2) predict factors that influence those cluster labels, and (3) estimate relationships between cluster labels and help‑seeking behaviors.

> **Datasets provided**
>
> * `Projec3_36841-StalkingVictim-2016-New-Shortened_NoReport (n=1024).sav`
> * `Project3_37950-StalkingVictim-2019-New-Shortened_NoReport (n=990).sav`
>
> Example columns observed (subset): `V1003, V1004, V1005, V1006, AGErange, VicFemale, WEAPON, JOB, NumOffnd, Duration, CRrecord, NO_RptP1 ... NO_RptP22 (subset present), PRB_work, PRB_people, LostCost, LostJob, LiveAlone, MARRIED, WhiteNH, BlackNH, OtherNH, RelPartn, RelStrng, RelNonStrng, RelUnkn, C_NgtEmtn, C_Loss_TM, B_NgtEmtn, B_Loss_TM, B_ProfHSB, B_InformHSB, Retaliation, Affection, Control, CrimeRelated`.

---

## Objectives & Steps

### Step 0 — Data preparation

* Read `.sav` files, harmonize variable names and coding across years.
* **Handle missing/invalid codes:** values such as `8, 9, 98, 99, 998, 999, 9998, 9999` → convert to `NaN` by default (configurable).
* Restrict to **stalking victims who did not report** (already satisfied by provided files, but will verify filters).
* Optional: merge 2016 and 2019 into a combined dataset with a `year` indicator, or analyze separately and compare.

### Step 1 — Unsupervised clustering (labels)

* **Target signals:** the non‑reporting reason items `NO_RptP1–NO_RptP22` (binary: 0=No, 1=Yes). We'll use only those columns to learn clusters.
* **Preprocessing:**

  * Drop rows with all‑missing across `NO_RptP*`.
  * Impute remaining missing items (default: median/mode per item; sensitivity analysis with complete‑case).
  * Optionally reduce dimensionality with **tetrachoric PCA** or **factor analysis for mixed data**; otherwise use sparse methods on binaries.
* **Algorithms (choose via validation):**

  * `KMeans`/`MiniBatchKMeans` on PCA scores,
  * `Agglomerative` (Ward linkage) on Hamming distance,
  * `Latent Class Analysis (LCA)` alternative if needed.
* **Model selection:** silhouette/Davies‑Bouldin, stability (bootstrap), and interpretability. Typical K in [2–8].
* **Outputs:**

  * Cluster label per victim (`cluster_label`).
  * Cluster profiles: prevalence of each `NO_RptP*`, descriptive names (e.g., *Too Minor/Private*, *Police Distrust*, *Identity Rejection*).

### Step 2 — Supervised prediction of labels

* **Goal:** identify which victim/offense characteristics predict membership in the Step‑1 clusters.
* **Predictors (18 dichotomous features):** *to be supplied* from the Project document. Examples from the data include: `PRB_work`, `PRB_people`, `LostCost`, `LostJob`, `LiveAlone`, `MARRIED`, race indicators, relationship to offender (`Rel*`), coping (`C_*`, `B_*`), motives (`Affection`, `Control`, `CrimeRelated`), etc.
* **Models:**

  * **Multinomial logistic regression** (baseline, interpretable).
  * **Tree‑based models** (Random Forest / XGBoost) for non‑linearities and interactions, with permutation importance and SHAP for interpretation.
* **Outputs:** feature effects (odds ratios / marginal effects), importance plots, cross‑validated performance.

### Step 3 — Association with help‑seeking behaviors

* **Outcomes:** `B_ProfHSB` (Professional) and `B_InformHSB` (Informal), both binary.
* **Analytic plan:**

  * Logistic regressions: outcome ~ `cluster_label` + controls (age range, sex, relationship, etc.).
  * Robustness: year fixed‑effects if combining datasets; clustered SEs by `V1005` if respondent id repeats.
  * Report adjusted odds ratios, 95% CIs, and predicted probabilities by cluster.

---

## Data dictionary & coding

* **Binary items:** assume `0=No`, `1=Yes`. We will verify against metadata and ensure consistency across years.
* **Non‑reporting reasons:** `NO_RptP1–NO_RptP22` (some years may omit certain items; we will harmonize and document).
* **Help‑seeking:** `B_ProfHSB`, `B_InformHSB` (0/1).
* **Identifiers:** `V1005` appears to be an anonymized id; we will de‑duplicate cautiously.
* **Missingness:** all special codes (e.g., 8/9/98/99/998/999) → `NaN` by default (configurable in `config.yml`).

---

## Deliverables

1. Cleaned, harmonized analysis dataset(s) with a documented data dictionary.
2. Cluster solution with labeled cluster names and profile tables/plots.
3. Supervised model results with coefficients/importance and validation metrics.
4. Help‑seeking association estimates with tables and effect visualizations.
5. A short, plain‑language interpretation memo.

---

## Repository structure

```
/README.md                    ← this file
/data/raw/                    ← original .sav files (git‑ignored)
/data/processed/              ← cleaned parquet/csv (git‑ignored)
/notebooks/                   ← exploratory notebooks
/src/
  ├─ config.py                ← paths & constants
  ├─ io.py                    ← load_sav, write parquet/csv
  ├─ preprocess.py            ← cleaning, recodes, harmonization
  ├─ cluster.py               ← clustering + model selection
  ├─ predict.py               ← supervised models (step 2)
  ├─ outcomes.py              ← help‑seeking models (step 3)
  └─ report.py                ← tables/plots export
/config.yml                   ← project settings (missing codes, year combine, K grid, etc.)
/environment.yml              ← conda env spec
```

---

## Environment setup

```bash
# recommended: conda
conda env create -f environment.yml
conda activate stalking-clusters

# dev install
pip install -e .
```

**Key packages:** `pandas`, `numpy`, `pyreadstat`, `scikit-learn`, `statsmodels`, `scipy`, `category_encoders`, `matplotlib`, `seaborn`, `pyjanitor`, optional: `prince` (MCA), `semopy`/`lavaan2py` if SEM is desired, `shap`.

---

## Reproducible pipeline (high level)

```python
# src/io.py (sketch)
from pyreadstat import read_sav
import pandas as pd

def load_sav(path: str) -> pd.DataFrame:
    df, _ = read_sav(path)
    return df
```

```python
# src/preprocess.py (sketch)
SPECIAL_MISS = {8, 9, 98, 99, 998, 999, 9998, 9999}

def to_nan(df, cols):
    return df.assign(**{c: df[c].where(~df[c].isin(SPECIAL_MISS)) for c in cols})
```

---

## What I need from you (to finalize specs)

1. **Authoritative list of the 18 predictor variables** for Step 2 (exact names).
2. **Clustering preference:** numeric labels only vs. interpretable, named clusters (I can propose names from item profiles).
3. **Missing data policy:** confirm default → convert special codes to `NaN`; impute binary items with mode; drop rows with all‑missing `NO_RptP*`.
4. **Combine years?** Choose between (a) combined analysis with year indicator, or (b) separate analyses with a comparison section.
5. **Any mandatory control variables** for Step 3 (e.g., `AGErange`, `VicFemale`, relationship to offender).

---

## Quality checks

* Year‑by‑year codebook consistency checks.
* Duplicate id detection on `V1005`.
* Cluster stability via bootstrap and alternative distance metrics.
* Sensitivity: complete‑case vs. imputed.

---

## License & data handling

* Raw data are sensitive; keep under `/data/raw/` and **do not** commit to version control.
* Outputs will be aggregated and anonymized.

---

## Quickstart (once files are in `/data/raw/`)

```python
# notebooks/00_quickstart.ipynb
import pandas as pd
from pyreadstat import read_sav

p16, _ = read_sav('data/raw/Projec3_36841-StalkingVictim-2016-New-Shortened_NoReport (n=1024).sav')
p19, _ = read_sav('data/raw/Project3_37950-StalkingVictim-2019-New-Shortened_NoReport (n=990).sav')

# inspect
for df in [p16, p19]:
    print(df.shape)
    print(df.head(3))
```

---

**Next action:** Share the list of the 18 predictors for Step 2 and your choices for the four items in *What I need from you*. After that, I’ll instantiate the code skeleton and first pass of data cleaning.
