
"""
stalking_supervised.py
Step 2 (Option 1 + Option 2): Supervised model suite and outcome models.

Reads: outputs/combined_with_clusters.csv  (created by stalking_clustering.py)

- Option 1: Predict cluster_label from the 18 binary predictors (LogReg L1/L2, Decision Tree, NB, ANN) with 5-fold CV & small grids.
- Option 2: Within each cluster, classify help-seeking outcomes (B_ProfHSB, B_InformHSB) from the same predictors.
- Also runs global outcome models: Logistic regressions (OR plots) of outcomes ~ cluster + controls.
"""
from __future__ import annotations
import os, json, warnings
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTDIR = "outputs"
FIGDIR = os.path.join(OUTDIR, "outcome_figs")

# 18 predictors per spec
PREDICTORS: List[str] = [
    "WEAPON","NumOffnd","Duration","CRrecord",
    "B_Loss_TM","LostCost","LostJob",
    "RelPartn","RelStrng","RelNonStrng","RelUnkn",
    "Retaliation","Affection","Control","CrimeRelated",
    "B_NgtEmtn","C_NgtEmtn","C_Loss_TM"
]

CONTROL_CANDIDATES = ["AGErange", "VicFemale", "WhiteNH", "BlackNH", "OtherNH", "MARRIED", "year"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def coerce_predictors_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "NumOffnd" in out.columns:
        out["NumOffnd"] = (pd.to_numeric(out["NumOffnd"], errors="coerce") > 1).astype(float)
    if "Duration" in out.columns:
        out["Duration"] = (pd.to_numeric(out["Duration"], errors="coerce") >= 3).astype(float)
    if "B_Loss_TM" in out.columns:
        out["B_Loss_TM"] = (pd.to_numeric(out["B_Loss_TM"], errors="coerce") > 0).astype(float)
    for c in PREDICTORS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            uniq = set(pd.unique(out[c].dropna()))
            if not uniq.issubset({0.0, 1.0}):
                out[c] = out[c].apply(lambda x: np.nan if pd.isna(x) else (1.0 if x == 1 or x == 1.0 else 0.0))
    return out

def most_frequent_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else 0.0
            out[c] = out[c].fillna(fill)
    return out

def run_model_suite(df: pd.DataFrame, label_col: str, predictors: List[str], random_state: int = 42) -> Dict[str, Any]:
    X = df[predictors].astype(float).values
    y = df[label_col].astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    models = []

    models.append(("LogReg_L2", LogisticRegression(multi_class="multinomial", solver="saga", penalty="l2", max_iter=1000, random_state=random_state),
                   {"C":[0.1, 1.0, 3.0, 10.0]}))
    models.append(("LogReg_L1", LogisticRegression(multi_class="multinomial", solver="saga", penalty="l1", max_iter=1000, random_state=random_state),
                   {"C":[0.1, 1.0, 3.0, 10.0]}))
    models.append(("DecisionTree", DecisionTreeClassifier(random_state=random_state),
                   {"max_depth":[None, 3, 5, 10], "min_samples_split":[2, 10, 25]}))
    models.append(("GaussianNB", GaussianNB(), {"var_smoothing":[1e-9, 1e-8, 1e-7]}))
    models.append(("BernoulliNB", BernoulliNB(), {"alpha":[0.1, 0.5, 1.0, 2.0]}))
    models.append(("MLP", MLPClassifier(max_iter=400, random_state=random_state),
                   {"hidden_layer_sizes":[(50,), (100,), (50,50)], "alpha":[1e-4, 1e-3, 1e-2]}))

    best_models = {}
    for name, est, grid in models:
        pipe = make_pipeline(StandardScaler(with_mean=False), est)
        # GridSearchCV requires params named with estimator step lowercased
        step_name = est.__class__.__name__.lower()
        grid_prefixed = {f"{step_name}__{k}": v for k, v in grid.items()}
        gs = GridSearchCV(pipe, param_grid=grid_prefixed, cv=cv, scoring="f1_macro", refit=True)
        gs.fit(X, y)
        results.append({"model": name, "best_score": float(gs.best_score_), "best_params": gs.best_params_})
        best_models[name] = gs.best_estimator_

    res_df = pd.DataFrame(results).sort_values("best_score", ascending=False)
    res_df.to_csv(os.path.join(OUTDIR, "model_suite_cv_results.csv"), index=False)

    # bar plot
    fig = plt.figure()
    x = np.arange(len(res_df))
    plt.bar(x, res_df["best_score"].values)
    plt.xticks(x, res_df["model"].values, rotation=45, ha="right")
    plt.ylabel("Mean CV F1 (macro)")
    plt.title("Step 2 model suite: CV performance")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "model_suite_cv_scores.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUTDIR, "model_suite_best_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return {"results": res_df, "best_models": best_models}

def fit_outcome_logit(df: pd.DataFrame, outcome: str, cluster_col: str, controls: List[str]) -> Dict[str, Any]:
    cols = [outcome, cluster_col] + controls
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for outcome model: {missing}")
    # treat AGErange as categorical if present
    terms = [f"C({cluster_col})"]
    for c in controls:
        if c == "AGErange":
            terms.append("C(AGErange)")
        else:
            terms.append(c)
    formula = f"{outcome} ~ " + " + ".join(terms)
    model = smf.logit(formula, data=df).fit(disp=False)
    params = model.params
    conf = model.conf_int()
    or_table = pd.DataFrame({
        "term": params.index,
        "OR": np.exp(params.values),
        "CI_low": np.exp(conf[0].values),
        "CI_high": np.exp(conf[1].values),
        "p_value": model.pvalues.values,
    })
    return {"formula": formula, "model": model, "or_table": or_table}

def plot_outcome_or(or_table: pd.DataFrame, outcome: str, outpath: str):
    subset = or_table[or_table["term"].str.contains(r"C\(cluster_label\)\[T\.", regex=True)].copy()
    if subset.empty:
        return
    fig = plt.figure()
    x = np.arange(len(subset))
    errs = np.vstack([
        subset["OR"].values - subset["CI_low"].values,
        subset["CI_high"].values - subset["OR"].values
    ])
    plt.errorbar(x, subset["OR"].values, yerr=errs, fmt='o')
    plt.xticks(x, subset["term"].values, rotation=45, ha="right")
    plt.axhline(1.0, linestyle="--")
    plt.ylabel("Odds ratio")
    plt.title(f"Outcome ORs by cluster: {outcome}")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def clusterwise_binary_classifier(df: pd.DataFrame, cluster_label: int, outcome: str, predictors: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
    sub = df[df["cluster_label"] == cluster_label].dropna(subset=predictors + [outcome]).copy()
    if sub.empty or sub[outcome].nunique() < 2:
        return {"warning": "insufficient data or only one class present"}, None
    X = sub[predictors].astype(float)
    y = sub[outcome].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=500, random_state=42))
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    rep = classification_report(y_te, y_pred, output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    return {"report": rep, "cm": cm}, sub.index.values

def main():
    ensure_dir(OUTDIR); ensure_dir(FIGDIR)
    combined_path = os.path.join(OUTDIR, "combined_with_clusters.csv")
    if not os.path.exists(combined_path):
        raise SystemExit("Run stalking_clustering.py first to create outputs/combined_with_clusters.csv")

    df = pd.read_csv(combined_path)
    # Prepare predictors (binarize & impute)
    df = coerce_predictors_to_binary(df)
    present_preds = [c for c in PREDICTORS if c in df.columns]
    df = most_frequent_impute(df, present_preds)

    # ---------- Option 1: predict cluster_label ----------
    df_pred = df.dropna(subset=present_preds + ["cluster_label"]).copy()
    if df_pred["cluster_label"].nunique() >= 2:
        suite = run_model_suite(df_pred, "cluster_label", present_preds, random_state=42)
        print("Option 1 model suite done. Top models:")
        print(suite["results"].head())
    else:
        print("Skipping Option 1: not enough cluster variation after filtering.")

    # ---------- Global outcomes (OR plots) ----------
    outcomes = [c for c in ["B_ProfHSB","B_InformHSB"] if c in df.columns]
    controls = [c for c in CONTROL_CANDIDATES if c in df.columns]
    for outcome in outcomes:
        try:
            sub = df.dropna(subset=[outcome, "cluster_label"]).copy()
            res = fit_outcome_logit(sub, outcome, "cluster_label", controls=controls)
            res["or_table"].to_csv(os.path.join(OUTDIR, f"outcome_OR_{outcome}.csv"), index=False)
            plot_outcome_or(res["or_table"], outcome, os.path.join(FIGDIR, f"outcome_OR_{outcome}.png"))
            print(f"Outcome model for {outcome} completed.")
        except Exception as e:
            print(f"Outcome model for {outcome} failed:", e)

    # ---------- Option 2: cluster-wise help-seeking classifiers ----------
    for outcome in outcomes:
        for cl in sorted(df["cluster_label"].dropna().unique()):
            try:
                rep_cm, idx = clusterwise_binary_classifier(df, int(cl), outcome, present_preds)
                if "warning" in rep_cm:
                    print(f"[Cluster {cl}] {outcome} skipped:", rep_cm["warning"])
                    continue
                with open(os.path.join(OUTDIR, f"clusterwise_{outcome}_report_cluster{cl}.json"), "w") as f:
                    json.dump(rep_cm["report"], f, indent=2)
                pd.DataFrame(rep_cm["cm"]).to_csv(os.path.join(OUTDIR, f"clusterwise_{outcome}_cmatrix_cluster{cl}.csv"), index=False, header=False)
                # optional: simple confusion matrix plot
                fig = plt.figure()
                plt.imshow(rep_cm["cm"], interpolation='nearest')
                plt.colorbar()
                ticks = np.arange(rep_cm["cm"].shape[0])
                plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.title(f"Confusion: {outcome} (cluster {cl})")
                plt.tight_layout()
                fig.savefig(os.path.join(FIGDIR, f"clusterwise_{outcome}_cmatrix_cluster{cl}.png"), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[Cluster {cl}] {outcome} classification failed:", e)

    print("All done. See outputs/ and outputs/figs/.")

if __name__ == "__main__":
    main()
