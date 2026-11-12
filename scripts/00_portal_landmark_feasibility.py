"""Feasibility check: can portal tracts be located as discrete landmarks at bin50?

A lobular (central -> portal) coordinate needs both central-vein and portal landmarks.
Central veins are already defined (GLUL consensus 9/9 domains, earlier analysis).
Here we score portal-tract / periportal markers, smooth them spatially, and check
whether they form discrete foci that are spatially separated from the GLUL domains.

Reads earlier outputs read-only; writes only into analysis_beyond_paper/.
"""
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parents[1]
# Input data (bin50 AnnData objects); default is a GSE315416 folder next to this repository.
SRC = Path(os.environ.get("BOVINE_LIVER_DATA", OUT.parent / "GSE315416"))

# Portal tract = bile duct epithelium + portal mesenchyme/vessel wall; periportal hepatocytes.
PORTAL_STRUCT = ["KRT19", "KRT7", "EPCAM", "SOX9", "SPP1", "CFTR", "COL1A1", "COL1A2",
                 "COL3A1", "ACTA2", "MYH11", "TAGLN", "DCN", "LUM", "VIM"]
PERIPORTAL_HEP = ["CPS1", "ASS1", "ASL", "PCK1", "HAL", "SDS", "ARG1", "HAMP", "CYP2F1", "ALDOB"]
PERICENTRAL = ["GLUL", "CYP2E1", "CYP2A13", "OAT", "LECT2", "RHBG", "AXIN2", "CYP1A2"]


def grid_smooth(values, ix, iy, shape, sigma):
    img = np.zeros(shape, dtype=float)
    cnt = np.zeros(shape, dtype=float)
    np.add.at(img, (iy, ix), values)
    np.add.at(cnt, (iy, ix), 1.0)
    num = ndimage.gaussian_filter(img, sigma)
    den = ndimage.gaussian_filter(cnt, sigma)
    return (num / np.maximum(den, 1e-9))[iy, ix]


def main():
    a = ad.read_h5ad(SRC / "processed_data/GSM10044212_bin50_QC_filtered.h5ad")
    glul = ad.read_h5ad(SRC / "processed_data/GSM10044212_bin50_GLUL_consensus_components.h5ad",
                        backed="r")
    xy = a.obsm["spatial"].astype(float)
    step = np.median(np.diff(np.unique(xy[:, 0])))
    ix = ((xy[:, 0] - xy[:, 0].min()) / step).round().astype(int)
    iy = ((xy[:, 1] - xy[:, 1].min()) / step).round().astype(int)
    shape = (iy.max() + 1, ix.max() + 1)

    counts = a.X.tocsr()
    libsize = np.asarray(counts.sum(1)).ravel()
    cpm = counts.multiply(1e4 / libsize[:, None]).tocsr()

    rows = []
    for group, genes in [("portal_struct", PORTAL_STRUCT), ("periportal_hep", PERIPORTAL_HEP),
                         ("pericentral", PERICENTRAL)]:
        for g in genes:
            if g not in a.var_names:
                rows.append((group, g, False, np.nan, np.nan))
                continue
            col = counts[:, a.var_names.get_loc(g)].toarray().ravel()
            rows.append((group, g, True, (col > 0).mean() * 100, col.sum()))
    detect = pd.DataFrame(rows, columns=["group", "gene", "present", "pct_bins_detected", "total_UMI"])
    detect.to_csv(OUT / "results/00_landmark_gene_detection.csv", index=False)
    print(detect.to_string(index=False))

    def score(genes):
        genes = [g for g in genes if g in a.var_names]
        m = np.log1p(cpm[:, [a.var_names.get_loc(g) for g in genes]].toarray())
        z = (m - m.mean(0)) / (m.std(0) + 1e-9)
        return z.mean(1)

    sigma = 3  # bins; bin50 ~ 25 um so sigma ~ 75 um
    s_portal = grid_smooth(score(PORTAL_STRUCT), ix, iy, shape, sigma)
    s_pp = grid_smooth(score(PERIPORTAL_HEP), ix, iy, shape, sigma)
    s_pc = grid_smooth(score(PERICENTRAL), ix, iy, shape, sigma)

    # GLUL 9/9 bins as central-vein landmarks
    gobs = glul.obs
    freq_col = [c for c in gobs.columns if "consensus" in c.lower() and "freq" in c.lower()]
    freq = gobs[freq_col[0]].reindex(a.obs_names).to_numpy() if freq_col else None
    cv = freq == 9 if freq is not None else np.zeros(a.n_obs, bool)

    # Portal candidates: top 1% of smoothed structural score
    thr = np.quantile(s_portal, 0.99)
    pt = s_portal >= thr
    lab, n = ndimage.label(np.isin(np.arange(shape[0] * shape[1]).reshape(shape),
                                   np.ravel_multi_index((iy[pt], ix[pt]), shape)))
    sizes = np.bincount(lab.ravel())[1:]

    d_cv_tree = cKDTree(np.c_[ix[cv], iy[cv]]) if cv.any() else None
    if d_cv_tree is not None:
        d_pt_to_cv = d_cv_tree.query(np.c_[ix[pt], iy[pt]])[0]
        d_rand = d_cv_tree.query(np.c_[ix, iy][np.random.default_rng(0).choice(a.n_obs, pt.sum())])[0]
    else:
        d_pt_to_cv = d_rand = np.array([np.nan])

    corr = pd.DataFrame({"portal_struct": s_portal, "periportal_hep": s_pp, "pericentral": s_pc}).corr("spearman")
    summary = pd.Series({
        "n_bins": a.n_obs,
        "grid_step_coord_units": step,
        "n_GLUL_9of9_bins": int(cv.sum()),
        "portal_top1pct_bins": int(pt.sum()),
        "portal_candidate_foci": int(n),
        "portal_foci_ge5_bins": int((sizes >= 5).sum()),
        "median_dist_portal_to_CV_bins": float(np.median(d_pt_to_cv)),
        "median_dist_random_to_CV_bins": float(np.median(d_rand)),
        "spearman_portal_vs_periportal": corr.loc["portal_struct", "periportal_hep"],
        "spearman_portal_vs_pericentral": corr.loc["portal_struct", "pericentral"],
        "spearman_periportal_vs_pericentral": corr.loc["periportal_hep", "pericentral"],
    })
    summary.to_csv(OUT / "results/00_portal_feasibility_summary.csv", header=["value"])
    print(summary.to_string())

    fig, axs = plt.subplots(1, 4, figsize=(22, 6.5))
    for ax, v, t in zip(axs[:3], [s_portal, s_pp, s_pc],
                        ["Portal-structure score (smoothed)", "Periportal hepatocyte score",
                         "Pericentral score"]):
        lo, hi = np.quantile(v, [0.02, 0.995])
        sc = ax.scatter(ix, iy, c=v, s=0.3, cmap="magma", vmin=lo, vmax=hi, rasterized=True)
        ax.set_title(t); ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
        plt.colorbar(sc, ax=ax, fraction=0.03)
    ax = axs[3]
    ax.scatter(ix, iy, c="#e5e5e5", s=0.2, rasterized=True)
    ax.scatter(ix[pt], iy[pt], c="#1f77b4", s=0.8, label="portal top 1%")
    ax.scatter(ix[cv], iy[cv], c="#d62728", s=0.8, label="GLUL 9/9 (central)")
    ax.legend(markerscale=8, loc="upper right"); ax.set_title("Landmarks")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "figures/00_portal_landmark_feasibility.png", dpi=150)


if __name__ == "__main__":
    main()
