
"""
stalking_clustering.py
Preprocessing + KMeans (K=4) on NO_RptP* items, figures, and labeled dataset export.

- Hardcoded SAV paths:
    P2016 = "Projec3_36841-StalkingVictim-2016-New-Shortened_NoReport (n=1024).sav"
    P2019 = "Project3_37950-StalkingVictim-2019-New-Shortened_NoReport (n=990).sav"

- Outputs:
    outputs/combined_with_clusters.csv
    outputs/cluster_profiles.csv
    outputs/cluster_selection.csv
    outputs/labels.csv
    outputs/figs/*  (cluster visuals)
"""
from __future__ import annotations
import os, re, warnings
from typing import List, Iterable, Dict, Any

import numpy as np
import pandas as pd
from pyreadstat import read_sav

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# Config (hardcoded)
# ---------------------------
P2016 = "Projec3_36841-StalkingVictim-2016-New-Shortened_NoReport (n=1024).sav"
P2019 = "Project3_37950-StalkingVictim-2019-New-Shortened_NoReport (n=990).sav"
OUTDIR = "outputs"
FIGDIR = os.path.join(OUTDIR, "figs")
ID_COL = "V1005"
NO_PREFIX = "NO_RptP"
SPECIAL_MISSING = {8, 9, 98, 99, 998, 999, 9998, 9999}
K_CLUSTERS = 4
LOW_VAR_MIN = 0.05
LOW_VAR_MAX = 0.95

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def infer_year_from_name(path: str) -> int | None:
    m = re.search(r"(20\d{2})", os.path.basename(path))
    return int(m.group(1)) if m else None

def convert_special_to_nan(df: pd.DataFrame, special: Iterable[int]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace(list(special), np.nan)
    return df

def detect_no_rpt_columns(df: pd.DataFrame, prefix: str = "NO_RptP") -> List[str]:
    ordered = []
    for i in range(1, 28):
        nm = f"{prefix}{i}"
        if nm in df.columns:
            ordered.append(nm)
    if ordered:
        return ordered
    return [c for c in df.columns if c.startswith(prefix)]

def enforce_binary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            uniq = set(pd.unique(df[c].dropna()))
            if not uniq.issubset({0.0, 1.0}):
                df[c] = df[c].apply(lambda x: np.nan if pd.isna(x) else (1.0 if x == 1 or x == 1.0 else 0.0))
    return df

def drop_all_missing(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    return df.loc[~df[subset].isna().all(axis=1)].copy()

def mode_impute(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else 0.0
            df[c] = df[c].fillna(fill)
    return df

def remove_low_variance_reasons(df: pd.DataFrame, no_cols: List[str], lo=LOW_VAR_MIN, hi=LOW_VAR_MAX) -> List[str]:
    keep = []
    for c in no_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        p = s.mean(skipna=True)
        if p is not None and (p >= lo) and (p <= hi):
            keep.append(c)
    return keep if keep else no_cols

def compute_2d_embedding_for_plot(X_imp: np.ndarray) -> np.ndarray:
    # Try MCA
    try:
        import prince
        Z2 = prince.MCA(n_components=2, random_state=42).fit_transform(pd.DataFrame(X_imp))
        return Z2.values
    except Exception:
        pass
    # Try UMAP
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.2).fit_transform(X_imp)
    except Exception:
        pass
    # Fallback: PCA
    return PCA(n_components=2, random_state=42).fit_transform(X_imp)

# ---------------------------
# Plots
# ---------------------------
def plot_cluster_profile_bars(profiles: pd.DataFrame, no_cols: List[str], outdir: str):
    ensure_dir(outdir)
    cols = [c for c in no_cols if c in profiles.columns] or list(profiles.columns)
    for cl in profiles.index:
        fig = plt.figure()
        vals = profiles.loc[cl, cols].values
        x = np.arange(len(cols))
        plt.bar(x, vals)
        plt.xticks(x, cols, rotation=90)
        plt.ylabel("Share endorsing reason")
        plt.title(f"Cluster {cl} profile")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"cluster_profile_cluster{cl}.png"), dpi=150)
        plt.close(fig)

def plot_cluster_heatmap(profiles: pd.DataFrame, outpath: str):
    fig = plt.figure()
    mat = profiles.values.T
    plt.imshow(mat, aspect='auto')
    plt.colorbar()
    plt.yticks(range(len(profiles.columns)), profiles.columns)
    plt.xticks(range(len(profiles.index)), [str(i) for i in profiles.index])
    plt.xlabel("Cluster")
    plt.ylabel("NO_RptP items")
    plt.title("Reasons by Cluster (mean 0–1)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_embedding_with_ellipses(Z2: np.ndarray, labels: np.ndarray, outpath: str):
    fig = plt.figure()
    ax = plt.gca()
    for cl in sorted(set(labels)):
        idx = labels == cl
        ax.scatter(Z2[idx, 0], Z2[idx, 1], label=f"Cluster {cl}", s=12)
        pts = Z2[idx]
        if pts.shape[0] >= 3:
            mu = pts.mean(axis=0)
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2*2*np.sqrt(max(vals[0],1e-9)), 2*2*np.sqrt(max(vals[1],1e-9))
            ell = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, linestyle="--")
            ax.add_patch(ell)
            ax.text(mu[0], mu[1], f"C{cl}", fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.set_title("2D embedding by cluster with 2σ ellipses")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_pca_scatter(Z: np.ndarray, labels: np.ndarray, outpath: str):
    if Z.shape[1] < 2:
        return
    fig = plt.figure()
    for cl in sorted(set(labels)):
        idx = labels == cl
        plt.scatter(Z[idx, 0], Z[idx, 1], label=f"Cluster {cl}", s=12)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA scatter by cluster")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_silhouette(Z: np.ndarray, labels: np.ndarray, outpath: str):
    if len(set(labels)) < 2:
        return
    fig = plt.figure()
    svals = silhouette_samples(Z, labels)
    y_lower = 10
    for cl in sorted(set(labels)):
        sv = svals[labels == cl]
        sv.sort()
        y_upper = y_lower + len(sv)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, sv)
        plt.text(-0.05, (y_lower + y_upper) / 2, str(cl))
        y_lower = y_upper + 10
    plt.axvline(np.mean(svals), linestyle="--")
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster")
    plt.title("Silhouette plot (k=4)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_elbow(k_list, inertias, outpath: str):
    if not k_list or not inertias:
        return
    fig = plt.figure()
    plt.plot(k_list, inertias, marker='o')
    plt.xlabel("k"); plt.ylabel("Inertia (WSS)"); plt.title("Elbow plot (k=4 fixed)")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(fig)

def plot_cluster_sizes(labels: np.ndarray, outpath: str):
    fig = plt.figure()
    uniq, cnts = np.unique(labels, return_counts=True)
    plt.bar(uniq, cnts)
    plt.xlabel("Cluster"); plt.ylabel("Count")
    plt.title("Cluster sizes (k=4)")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(fig)

def plot_within_cluster_dist(Z: np.ndarray, labels: np.ndarray, outpath: str):
    fig = plt.figure()
    dists = []
    for cl in sorted(set(labels)):
        pts = Z[labels == cl]
        ctr = pts.mean(axis=0)
        dist = np.linalg.norm(pts - ctr, axis=1)
        dists.append(dist)
    plt.boxplot(dists, labels=[f"{c}" for c in sorted(set(labels))])
    plt.xlabel("Cluster"); plt.ylabel("Distance to centroid")
    plt.title("Within-cluster dispersion")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(fig)

# ---------------------------
# Clustering
# ---------------------------
def kmeans_cluster(df: pd.DataFrame, no_cols: List[str], random_state=42, n_components=10) -> Dict[str, Any]:
    imp = SimpleImputer(strategy="most_frequent")
    X = df[no_cols].astype(float).values
    X_imp = imp.fit_transform(X)

    # PCA for modeling
    Z = X_imp
    pca = None
    if X_imp.shape[1] > 1:
        nc = max(1, min(n_components, X_imp.shape[1]-1))
        pca = PCA(n_components=nc, random_state=random_state)
        Z = pca.fit_transform(X_imp)

    km = KMeans(n_clusters=4, n_init=10, random_state=random_state, max_iter=300)
    labels = km.fit_predict(Z)
    try:
        sil = silhouette_score(Z, labels) if len(set(labels)) > 1 else np.nan
    except Exception:
        sil = np.nan

    profiles = pd.DataFrame(X_imp, columns=no_cols).groupby(labels).mean().sort_index()
    profiles.index.name = "cluster"

    inertias = [km.inertia_]
    k_tried = [4]

    # 2D embedding for clearer plotting
    Z_plot = compute_2d_embedding_for_plot(X_imp)

    return {
        "labels": labels,
        "profiles": profiles,
        "k": 4,
        "silhouette": sil,
        "Z": Z,
        "Z_plot": Z_plot,
        "k_tried": k_tried,
        "inertias": inertias,
        "X_imp": X_imp
    }

# ---------------------------
# Main
# ---------------------------
def main():
    ensure_dir(OUTDIR); ensure_dir(FIGDIR)

    df16, _ = read_sav(P2016)
    df19, _ = read_sav(P2019)
    df16 = df16.copy(); df16["year"] = infer_year_from_name(P2016) or 2016
    df19 = df19.copy(); df19["year"] = infer_year_from_name(P2019) or 2019
    df = pd.concat([df16, df19], axis=0, ignore_index=True)

    df = convert_special_to_nan(df, SPECIAL_MISSING)
    no_cols = detect_no_rpt_columns(df, prefix=NO_PREFIX)
    if len(no_cols) < 2:
        raise SystemExit(f"Not enough {NO_PREFIX}* columns. Found: {no_cols}")
    df = enforce_binary(df, no_cols)
    df = drop_all_missing(df, subset=no_cols)
    df = mode_impute(df, no_cols)
    filtered_no_cols = remove_low_variance_reasons(df, no_cols, lo=LOW_VAR_MIN, hi=LOW_VAR_MAX)
    if len(filtered_no_cols) < 2:
        filtered_no_cols = no_cols

    # Step 1: KMeans
    clus = kmeans_cluster(df, filtered_no_cols, random_state=42, n_components=10)
    df["cluster_label"] = clus["labels"]
    clus["profiles"].to_csv(os.path.join(OUTDIR, "cluster_profiles.csv"))
    pd.DataFrame({"k":[clus["k"]], "silhouette":[clus["silhouette"]]}).to_csv(os.path.join(OUTDIR, "cluster_selection.csv"), index=False)

    # Visuals
    plot_cluster_profile_bars(clus["profiles"], filtered_no_cols, FIGDIR)
    plot_cluster_heatmap(clus["profiles"], os.path.join(FIGDIR, "cluster_heatmap.png"))
    plot_pca_scatter(clus["Z"], clus["labels"], os.path.join(FIGDIR, "pca_scatter.png"))
    plot_embedding_with_ellipses(clus["Z_plot"], clus["labels"], os.path.join(FIGDIR, "embedding_scatter_ellipses.png"))
    plot_silhouette(clus["Z"], clus["labels"], os.path.join(FIGDIR, "silhouette_k4.png"))
    plot_elbow(clus["k_tried"], clus["inertias"], os.path.join(FIGDIR, "elbow_inertia.png"))
    plot_cluster_sizes(clus["labels"], os.path.join(FIGDIR, "cluster_sizes.png"))
    plot_within_cluster_dist(clus["Z"], clus["labels"], os.path.join(FIGDIR, "within_cluster_dispersion.png"))

    # Labels export & combined export
    labels_cols = ["year","cluster_label"]
    if ID_COL in df.columns:
        labels_cols = [ID_COL] + labels_cols
    df[labels_cols].to_csv(os.path.join(OUTDIR, "labels.csv"), index=False)
    df.to_csv(os.path.join(OUTDIR, "combined_with_clusters.csv"), index=False)
    print("Saved outputs to:", OUTDIR)

if __name__ == "__main__":
    main()
