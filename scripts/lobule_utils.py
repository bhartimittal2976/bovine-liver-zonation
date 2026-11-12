"""Shared helpers: bin grid, landmarks, lobular coordinate, batched quasi-Poisson GLM."""
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import ndimage, stats

OUT = Path(__file__).resolve().parents[1]
# Input data (bin50 AnnData objects); default is a GSE315416 folder next to this repository.
SRC = Path(os.environ.get("BOVINE_LIVER_DATA", OUT.parent / "GSE315416"))

BIN_UM = 25.0  # bin50 = 50 DNB x 0.5 um

BILIARY = ["KRT19", "KRT7", "EPCAM", "SOX9", "SPP1", "CFTR"]
MESENCHYMAL = ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "MYH11", "TAGLN", "DCN", "LUM", "VIM"]
PORTAL_STRUCT = BILIARY + MESENCHYMAL
# Genes used to place landmarks; their own zonation is circular and is flagged downstream.
LANDMARK_GENES = set(PORTAL_STRUCT) | {"GLUL"}


def load_bins():
    a = ad.read_h5ad(SRC / "processed_data/GSM10044212_bin50_QC_filtered.h5ad")
    g = ad.read_h5ad(SRC / "processed_data/GSM10044212_bin50_GLUL_consensus_components.h5ad",
                     backed="r").obs
    a.obs["GLUL_consensus_frequency"] = g["GLUL_consensus_frequency"].reindex(a.obs_names).to_numpy()
    xy = a.obsm["spatial"].astype(float)
    a.obs["ix"] = ((xy[:, 0] - xy[:, 0].min()) / 50).round().astype(int)
    a.obs["iy"] = ((xy[:, 1] - xy[:, 1].min()) / 50).round().astype(int)
    a.X = a.X.tocsr()
    a.obs["lib"] = np.asarray(a.X.sum(1)).ravel()
    return a


def grid_shape(a):
    return a.obs["iy"].max() + 1, a.obs["ix"].max() + 1


def smooth_on_grid(values, ix, iy, shape, sigma):
    img = np.zeros(shape)
    cnt = np.zeros(shape)
    np.add.at(img, (iy, ix), values)
    np.add.at(cnt, (iy, ix), 1.0)
    return (ndimage.gaussian_filter(img, sigma) / np.maximum(ndimage.gaussian_filter(cnt, sigma), 1e-9))[iy, ix]


def zscore_logcpm(a, genes):
    genes = [g for g in genes if g in a.var_names]
    m = a.X[:, [a.var_names.get_loc(g) for g in genes]].toarray() * 1e4 / a.obs["lib"].to_numpy()[:, None]
    m = np.log1p(m)
    return ((m - m.mean(0)) / (m.std(0) + 1e-9)).mean(1)


def oe_ratio(a, genes, mask):
    """Observed/expected UMI for a gene set in mask, expected from library size share."""
    genes = [g for g in genes if g in a.var_names]
    c = np.asarray(a.X[:, [a.var_names.get_loc(g) for g in genes]].sum(1)).ravel()
    lib = a.obs["lib"].to_numpy()
    return c[mask].sum() / (c.sum() * lib[mask].sum() / lib.sum())


def build_landmarks(a, portal_score, cv_min_consensus=7, portal_quantile=0.985,
                    min_focus_bins=5, min_biliary_oe=2.0):
    """Return grid masks (cv, pt) and a per-focus QC table for portal candidates."""
    ix, iy = a.obs["ix"].to_numpy(), a.obs["iy"].to_numpy()
    shape = grid_shape(a)
    cv = np.zeros(shape, bool)
    sel = a.obs["GLUL_consensus_frequency"].to_numpy() >= cv_min_consensus
    cv[iy[sel], ix[sel]] = True

    cand = np.zeros(shape, bool)
    sel = portal_score >= np.quantile(portal_score, portal_quantile)
    cand[iy[sel], ix[sel]] = True
    cand &= ~ndimage.binary_dilation(cv, iterations=2)
    lab, n = ndimage.label(cand)
    bin_lab = lab[iy, ix]

    rows = []
    pt = np.zeros(shape, bool)
    dist_cv = ndimage.distance_transform_edt(~cv) * BIN_UM
    for k in range(1, n + 1):
        m = bin_lab == k
        if m.sum() < min_focus_bins:
            continue
        r = dict(focus=k, n_bins=int(m.sum()),
                 biliary_oe=oe_ratio(a, BILIARY, m),
                 mesenchymal_oe=oe_ratio(a, MESENCHYMAL, m),
                 GLUL_oe=oe_ratio(a, ["GLUL"], m),
                 dist_to_CV_um=float(dist_cv[lab == k].min()))
        r["kept_as_portal"] = r["biliary_oe"] >= min_biliary_oe
        rows.append(r)
        if r["kept_as_portal"]:
            pt |= lab == k
    return cv, pt, pd.DataFrame(rows)


def lobular_coordinate(a, cv, pt, max_span_um=1500.0, buffer_um=50.0):
    """eta = d_CV / (d_CV + d_PT): 0 at central vein, 1 at portal tract."""
    ix, iy = a.obs["ix"].to_numpy(), a.obs["iy"].to_numpy()
    d_cv = ndimage.distance_transform_edt(~cv)[iy, ix] * BIN_UM
    d_pt = ndimage.distance_transform_edt(~pt)[iy, ix] * BIN_UM
    span = d_cv + d_pt
    eta = np.where(span > 0, d_cv / np.maximum(span, 1e-9), np.nan)
    # Parenchyma only: landmark bins and a buffer around them are excluded from profiling,
    # so vessel wall / bile duct signal does not leak into the outermost layers.
    valid = (d_cv > buffer_um) & (d_pt > buffer_um) & (span <= max_span_um)
    return pd.DataFrame({"d_CV_um": d_cv, "d_PT_um": d_pt, "span_um": span,
                         "eta": eta, "valid": valid}, index=a.obs_names)


def pseudobulk(a, keep, layer, block, min_bins=20):
    """Sum counts per (block, layer) cell. Returns counts (cells x genes), meta."""
    key = pd.Series(block[keep].astype(str), dtype=str) + "|" + pd.Series(layer[keep].astype(str), dtype=str)
    codes, uniq = pd.factorize(key)
    n_cells = len(uniq)
    ind = np.zeros((n_cells, keep.sum()))
    # sparse aggregation via indicator matrix
    import scipy.sparse as sp
    ind = sp.csr_matrix((np.ones(keep.sum()), (codes, np.arange(keep.sum()))), shape=(n_cells, keep.sum()))
    counts = np.asarray((ind @ a.X[keep]).todense())
    meta = pd.DataFrame([u.split("|") for u in uniq], columns=["block", "layer"])
    meta["layer"] = meta["layer"].astype(int)
    meta["n_bins"] = np.bincount(codes)
    meta["lib"] = counts.sum(1)
    ok = meta["n_bins"].to_numpy() >= min_bins
    return counts[ok], meta[ok].reset_index(drop=True)


def ortho_poly(x, degree=3):
    """Orthogonal polynomial basis (like R poly()) with a function to evaluate new x."""
    x = np.asarray(x, float)
    mu = x.mean()
    V = np.vander(x - mu, degree + 1, increasing=True)
    Q, R = np.linalg.qr(V)
    basis = Q[:, 1:]

    def predict(xn):
        Vn = np.vander(np.asarray(xn, float) - mu, degree + 1, increasing=True)
        return (Vn @ np.linalg.inv(R))[:, 1:]
    return basis, predict


def batched_poisson_glm(Y, X, offset, n_iter=30, ridge=1e-6):
    """Poisson IRLS for many genes sharing one design. Y: n x G. Returns beta (G x p), mu (n x G)."""
    n, G = Y.shape
    p = X.shape[1]
    mu = Y + 0.5
    eta = np.log(mu)
    beta = np.zeros((G, p))
    for _ in range(n_iter):
        z = eta - offset[:, None] + (Y - mu) / mu
        XtWX = np.einsum("ni,ng,nj->gij", X, mu, X) + ridge * np.eye(p)
        XtWz = np.einsum("ni,ng->gi", X, mu * z)
        new = np.linalg.solve(XtWX, XtWz[..., None])[..., 0]
        if np.max(np.abs(new - beta)) < 1e-7:
            beta = new
            break
        beta = new
        eta = np.clip(X @ beta.T + offset[:, None], -50, 50)
        mu = np.exp(eta)
    mu = np.exp(np.clip(X @ beta.T + offset[:, None], -50, 50))
    return beta, mu


def poisson_deviance(Y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(Y > 0, Y * np.log(Y / mu), 0.0)
    return 2 * (t - (Y - mu)).sum(0)


def quasi_poisson_trend_test(Y, X_null, X_full, offset):
    """F-test for the extra columns of X_full; dispersion from the full model (Pearson)."""
    _, mu0 = batched_poisson_glm(Y, X_null, offset)
    beta1, mu1 = batched_poisson_glm(Y, X_full, offset)
    df_extra = X_full.shape[1] - X_null.shape[1]
    df_res = Y.shape[0] - X_full.shape[1]
    phi = np.maximum(((Y - mu1) ** 2 / mu1).sum(0) / df_res, 1.0)
    F = (poisson_deviance(Y, mu0) - poisson_deviance(Y, mu1)) / df_extra / phi
    F = np.maximum(F, 0)
    p = stats.f.sf(F, df_extra, df_res)
    return beta1, F, p, phi


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out
