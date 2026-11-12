"""Analysis 1: two-landmark lobular coordinate and depth-aware genome-wide zonation.

eta = d_CV / (d_CV + d_PT), with 0 at the central vein (GLUL consensus domains) and 1 at the
portal tract (smoothed biliary + portal-mesenchyme foci that are biliary-enriched).

Zonation model, per gene, on pseudobulk of (spatial block x eta layer) cells:
    log E[y] = block + poly3(eta) + log(library size)      (quasi-Poisson)
The spatial-block fixed effect absorbs regional depth/RNA-quality differences across the section,
so a gene must vary along eta *within* blocks to be called zonated. Robust calls also need the same
direction in the top and bottom halves, which have different sequencing depth.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from lobule_utils import (BIN_UM, LANDMARK_GENES, OUT, PORTAL_STRUCT, bh, build_landmarks,
                          grid_shape, load_bins, lobular_coordinate, ortho_poly, pseudobulk,
                          quasi_poisson_trend_test, batched_poisson_glm, smooth_on_grid,
                          zscore_logcpm)

N_LAYERS = 8
BLOCK_UM = 2000.0
MAIN = dict(cv_min_consensus=7, portal_quantile=0.985, max_span_um=1500.0)
VARIANTS = [
    dict(cv_min_consensus=9, portal_quantile=0.99, max_span_um=1500.0),
    dict(cv_min_consensus=9, portal_quantile=0.985, max_span_um=1500.0),
    dict(cv_min_consensus=7, portal_quantile=0.99, max_span_um=1500.0),
    dict(cv_min_consensus=7, portal_quantile=0.98, max_span_um=1500.0),
    dict(cv_min_consensus=7, portal_quantile=0.985, max_span_um=1000.0),
    dict(cv_min_consensus=7, portal_quantile=0.985, max_span_um=2000.0),
]
# Independent of landmark genes. Expected direction from rodent/human zonation literature.
VALIDATION = {
    "CYP2E1": "PC", "CYP2A13": "PC", "CYP1A2": "PC", "CYP3A76": "PC", "LECT2": "PC",
    "RHBG": "PC", "OAT": "PC", "AXIN2": "PC",
    "CPS1": "PP", "ASS1": "PP", "ASL": "PP", "ARG1": "PP", "PCK1": "PP", "HAL": "PP",
    "HAMP": "PP", "ALDOB": "PP", "SDS": "PP",
}
MIN_TOTAL_UMI = 200
MIN_DETECT_FRAC = 0.005
FDR_MAX = 0.05
MIN_ABS_LOG2FC = 0.5


def layer_of(eta):
    return np.minimum((eta * N_LAYERS).astype(int), N_LAYERS - 1)


def design(meta, layer_mid):
    blocks = pd.get_dummies(meta["block"], drop_first=True).to_numpy(float)
    X_null = np.c_[np.ones(len(meta)), blocks]
    basis, predict = ortho_poly(layer_mid[meta["layer"].to_numpy()], 3)
    return X_null, np.c_[X_null, basis], predict, X_null.shape[1]


def fit_zonation(a, coord, genes_idx, half=None, layers=None):
    """Return dict with F, p, log2FC (last vs first layer used), COM, profile (G x layers, log)."""
    keep = coord["valid"].to_numpy()
    if half is not None:
        keep = keep & half
    layer = layer_of(np.nan_to_num(coord["eta"].to_numpy()))
    if layers is not None:
        keep = keep & np.isin(layer, layers)
    bx = (a.obs["ix"].to_numpy() * BIN_UM // BLOCK_UM).astype(int)
    by = (a.obs["iy"].to_numpy() * BIN_UM // BLOCK_UM).astype(int)
    block = bx * 100 + by
    counts, meta = pseudobulk(a, keep, layer, block)
    Y = counts[:, genes_idx]
    mids = (np.arange(N_LAYERS) + 0.5) / N_LAYERS
    X_null, X_full, predict, k0 = design(meta, mids)
    offset = np.log(meta["lib"].to_numpy(float))
    beta, F, p, phi = quasi_poisson_trend_test(Y, X_null, X_full, offset)
    used = np.sort(meta["layer"].unique())
    prof = predict(mids[used]) @ beta[:, k0:].T  # layers x G, natural log, block-adjusted
    mids = mids[used]
    prof = prof - prof.mean(0)
    w = np.exp(prof)
    return dict(F=F, p=p, phi=phi, log2FC=(prof[-1] - prof[0]) / np.log(2),
                COM=(mids[:, None] * w).sum(0) / w.sum(0), profile=prof.T, meta=meta)


def main():
    a = load_bins()
    ix, iy = a.obs["ix"].to_numpy(), a.obs["iy"].to_numpy()
    shape = grid_shape(a)
    portal_score = smooth_on_grid(zscore_logcpm(a, PORTAL_STRUCT), ix, iy, shape, sigma=3)

    # ---- landmarks + coordinate (main) ----
    cv, pt, focus_qc = build_landmarks(a, portal_score, MAIN["cv_min_consensus"], MAIN["portal_quantile"])
    focus_qc.to_csv(OUT / "results/01_portal_focus_qc.csv", index=False)
    coord = lobular_coordinate(a, cv, pt, MAIN["max_span_um"])
    coord["layer"] = np.where(coord["valid"], layer_of(np.nan_to_num(coord["eta"])), -1)
    coord["is_CV_landmark"] = cv[iy, ix]
    coord["is_PT_landmark"] = pt[iy, ix]
    coord["x_um"], coord["y_um"] = ix * BIN_UM, iy * BIN_UM
    coord["lib"] = a.obs["lib"].to_numpy()
    coord.to_csv(OUT / "data/01_bin50_lobular_coordinate.csv.gz")
    from scipy import ndimage
    print(f"CV components {ndimage.label(cv)[1]}, portal foci kept {int(focus_qc.kept_as_portal.sum())}"
          f"/{len(focus_qc)}, valid parenchymal bins {coord.valid.sum()} ({coord.valid.mean():.1%})")

    # ---- gene filter ----
    valid = coord["valid"].to_numpy()
    Xv = a.X[valid]
    tot = np.asarray(Xv.sum(0)).ravel()
    det = np.asarray((Xv > 0).mean(0)).ravel()
    genes_idx = np.where((tot >= MIN_TOTAL_UMI) & (det >= MIN_DETECT_FRAC))[0]
    genes = a.var_names[genes_idx]
    print(f"genes tested: {len(genes)}")

    # ---- main fit + split halves (top/bottom differ in depth) ----
    main_fit = fit_zonation(a, coord, genes_idx)
    top = iy < np.median(iy)
    fit_top = fit_zonation(a, coord, genes_idx, half=top)
    fit_bot = fit_zonation(a, coord, genes_idx, half=~top)
    # Interior only (eta 0.125-0.875): a gradient across hepatocyte parenchyma, not a step
    # at the vessel / portal boundary.
    fit_int = fit_zonation(a, coord, genes_idx, layers=np.arange(1, N_LAYERS - 1))
    meta = main_fit["meta"]
    print(f"pseudobulk cells: {len(meta)} across {meta.block.nunique()} blocks")
    meta.to_csv(OUT / "results/01_pseudobulk_design.csv", index=False)

    res = pd.DataFrame({
        "gene": genes, "total_UMI_parenchyma": tot[genes_idx], "pct_bins_detected": det[genes_idx] * 100,
        "F": main_fit["F"], "p": main_fit["p"], "FDR": bh(main_fit["p"]), "dispersion": main_fit["phi"],
        "log2FC_PP_vs_PC": main_fit["log2FC"], "COM_eta": main_fit["COM"],
        "log2FC_top_half": fit_top["log2FC"], "log2FC_bottom_half": fit_bot["log2FC"],
        "FDR_top_half": bh(fit_top["p"]), "FDR_bottom_half": bh(fit_bot["p"]),
        "log2FC_interior": fit_int["log2FC"], "FDR_interior": bh(fit_int["p"]),
        "landmark_gene": [g in LANDMARK_GENES for g in genes],
    })
    for k in range(N_LAYERS):
        res[f"profile_L{k + 1}"] = main_fit["profile"][:, k]
    # Enrichment inside landmark bins vs parenchyma (library-size normalised). Genes enriched in
    # the vessel / portal structures themselves are likely non-hepatocyte signal.
    lib = a.obs["lib"].to_numpy()
    for name, m in [("PT", coord.is_PT_landmark.to_numpy()), ("CV", coord.is_CV_landmark.to_numpy())]:
        rate_l = np.asarray(a.X[m][:, genes_idx].sum(0)).ravel() / lib[m].sum()
        rate_p = tot[genes_idx] / lib[valid].sum()
        res[f"log2_enrich_in_{name}_landmark"] = np.log2((rate_l + 1e-7) / (rate_p + 1e-7))
    same_sign = np.sign(res.log2FC_top_half) == np.sign(res.log2FC_bottom_half)
    res["zonated"] = (res.FDR < FDR_MAX) & (res.log2FC_PP_vs_PC.abs() >= MIN_ABS_LOG2FC)
    res["robust"] = res.zonated & same_sign & (np.sign(res.log2FC_top_half) == np.sign(res.log2FC_PP_vs_PC))
    res["direction"] = np.where(~res.robust, "", np.where(res.log2FC_PP_vs_PC > 0, "periportal", "pericentral"))
    interior_ok = (res.FDR_interior < FDR_MAX) & (np.sign(res.log2FC_interior) == np.sign(res.log2FC_PP_vs_PC))
    res["pattern"] = np.where(~res.robust, "", np.where(interior_ok, "parenchymal_gradient", "boundary_restricted"))
    res["portal_structure_enriched"] = res.log2_enrich_in_PT_landmark >= 1.0  # GLUL domains are hepatocytes, so CV enrichment is not a flag
    res["parenchymal_zonated"] = (res.pattern == "parenchymal_gradient") & ~res.portal_structure_enriched & ~res.landmark_gene

    # ---- landmark sensitivity ----
    sens_rows = []
    for v in VARIANTS:
        cv_v, pt_v, _ = build_landmarks(a, portal_score, v["cv_min_consensus"], v["portal_quantile"])
        c_v = lobular_coordinate(a, cv_v, pt_v, v["max_span_um"])
        f_v = fit_zonation(a, c_v, genes_idx)
        tag = f"CV>={v['cv_min_consensus']}_q{v['portal_quantile']}_span{int(v['max_span_um'])}"
        res[f"log2FC_{tag}"] = f_v["log2FC"]
        r = res.robust.to_numpy()
        sens_rows.append(dict(variant=tag, valid_bins=int(c_v.valid.sum()),
                              spearman_all=stats.spearmanr(res.log2FC_PP_vs_PC, f_v["log2FC"])[0],
                              spearman_robust=stats.spearmanr(res.log2FC_PP_vs_PC[r], f_v["log2FC"][r])[0],
                              sign_agreement_robust=float(np.mean(np.sign(f_v["log2FC"][r]) ==
                                                                  np.sign(res.log2FC_PP_vs_PC[r])))))
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "results/01_zonation_landmark_sensitivity.csv", index=False)
    lfc_cols = [c for c in res.columns if c.startswith("log2FC_CV")]
    res["sign_stable_all_variants"] = (np.sign(res[lfc_cols]).eq(np.sign(res.log2FC_PP_vs_PC), axis=0)).all(1)
    res = res.sort_values(["robust", "FDR"], ascending=[False, True])
    res.to_csv(OUT / "results/01_zonation_genomewide.csv", index=False)

    # ---- validation against independent markers ----
    val = res.set_index("gene").reindex(list(VALIDATION))
    val["expected"] = pd.Series(VALIDATION)
    val["observed"] = np.where(val.log2FC_PP_vs_PC > 0, "PP", "PC")
    val.loc[val.log2FC_PP_vs_PC.isna(), "observed"] = "not_tested"
    val["agrees"] = (val.expected == val.observed).where(val.observed != "not_tested")
    val[["expected", "observed", "agrees", "log2FC_PP_vs_PC", "FDR", "COM_eta", "robust",
         "log2FC_top_half", "log2FC_bottom_half"]].to_csv(OUT / "results/01_axis_validation_markers.csv")

    # ---- summary ----
    rob = res[res.robust & ~res.landmark_gene]
    summ = pd.Series({
        "genes_tested": len(res),
        "zonated_FDR_and_effect": int(res.zonated.sum()),
        "robust_zonated_non_landmark": len(rob),
        "robust_periportal": int((rob.direction == "periportal").sum()),
        "robust_pericentral": int((rob.direction == "pericentral").sum()),
        "robust_sign_stable_all_variants": int(rob.sign_stable_all_variants.sum()),
        "robust_parenchymal_gradient_PP": int(((rob.pattern == "parenchymal_gradient") & (rob.direction == "periportal")).sum()),
        "robust_parenchymal_gradient_PC": int(((rob.pattern == "parenchymal_gradient") & (rob.direction == "pericentral")).sum()),
        "robust_boundary_restricted_PP": int(((rob.pattern == "boundary_restricted") & (rob.direction == "periportal")).sum()),
        "robust_boundary_restricted_PC": int(((rob.pattern == "boundary_restricted") & (rob.direction == "pericentral")).sum()),
        "interior_FDR05_any_effect": int((res.FDR_interior < FDR_MAX).sum()),
        "parenchymal_zonated_PP": int((rob.parenchymal_zonated & (rob.direction == "periportal")).sum()),
        "parenchymal_zonated_PC": int((rob.parenchymal_zonated & (rob.direction == "pericentral")).sum()),
        "zonated_same_sign_both_halves_pct": float(100 * np.mean(np.sign(res.log2FC_top_half[res.zonated]) ==
                                                                np.sign(res.log2FC_bottom_half[res.zonated]))),
        "validation_markers_agree": f"{int(val.agrees.fillna(False).sum())}/{int(val.agrees.notna().sum())} tested",
        "median_dispersion": float(res.dispersion.median()),
    })
    summ.to_csv(OUT / "results/01_zonation_summary.csv", header=["value"])
    print(summ.to_string())
    print(val[["expected", "observed", "log2FC_PP_vs_PC", "FDR", "log2FC_interior", "FDR_interior", "robust"]].to_string())
    print(sens.to_string(index=False))
    for pat, sub in [("parenchymal_zonated", rob[rob.parenchymal_zonated]),
                     ("portal_structure_enriched", rob[rob.portal_structure_enriched])]:
        print(pat, "PC:", sub[sub.direction == "pericentral"].nsmallest(25, "log2FC_PP_vs_PC").gene.tolist())
        print(pat, "PP:", sub[sub.direction == "periportal"].nlargest(25, "log2FC_PP_vs_PC").gene.tolist())

    # ---- bin-level marker profiles (model-free check) ----
    lib = a.obs["lib"].to_numpy()
    prof_rows = []
    lay10 = np.minimum((np.nan_to_num(coord.eta.to_numpy()) * 10).astype(int), 9)
    for g in VALIDATION:
        if g not in a.var_names:
            continue
        cpm = a.X[:, a.var_names.get_loc(g)].toarray().ravel() / lib * 1e4
        for h, hm in [("top", top), ("bottom", ~top), ("all", np.ones_like(top))]:
            m = valid & hm
            for k in range(10):
                s = m & (lay10 == k)
                prof_rows.append((g, h, (k + 0.5) / 10, cpm[s].mean(), s.sum()))
    bprof = pd.DataFrame(prof_rows, columns=["gene", "half", "eta_mid", "mean_CP10k", "n_bins"])
    bprof.to_csv(OUT / "results/01_validation_marker_bin_profiles.csv", index=False)

    make_figures(coord, focus_qc, res, val, bprof, sens)


def make_figures(coord, focus_qc, res, val, bprof, sens):
    PC_COL, PP_COL = "#c0392b", "#2471a3"
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    ax = fig.add_subplot(gs[0, 0])
    x, y = coord.x_um / 1000, coord.y_um / 1000
    ax.scatter(x, y, c="#e6e6e6", s=0.15, rasterized=True)
    ax.scatter(x[coord.is_CV_landmark], y[coord.is_CV_landmark], c=PC_COL, s=0.8, label="Central vein (GLUL)")
    ax.scatter(x[coord.is_PT_landmark], y[coord.is_PT_landmark], c=PP_COL, s=0.8, label="Portal tract (biliary+)")
    ax.legend(markerscale=10, loc="lower left", fontsize=9, frameon=True)
    ax.set_title("A  Landmarks"); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("mm"); ax.set_ylabel("mm")

    ax = fig.add_subplot(gs[0, 1])
    v = coord.valid
    ax.scatter(x[~v], y[~v], c="#ececec", s=0.15, rasterized=True)
    sc = ax.scatter(x[v], y[v], c=coord.eta[v], cmap="coolwarm_r", s=0.3, vmin=0, vmax=1, rasterized=True)
    ax.set_title(f"B  Lobular coordinate η (0 = CV, 1 = PT); {v.mean():.0%} of bins")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xlabel("mm")
    plt.colorbar(sc, ax=ax, fraction=0.035, label="η")

    ax = fig.add_subplot(gs[0, 2])
    allp = bprof[bprof.half == "all"]
    for g, sub in allp.groupby("gene", sort=False):
        if val.loc[g, "observed"] == "not_tested":
            continue
        exp = val.loc[g, "expected"]
        rel = sub.mean_CP10k / sub.mean_CP10k.mean()
        ax.plot(sub.eta_mid, np.log2(rel), color=PC_COL if exp == "PC" else PP_COL, alpha=0.75, lw=1.3)
        ax.text(sub.eta_mid.iloc[-1] + 0.01, np.log2(rel.iloc[-1]), g, fontsize=7,
                color=PC_COL if exp == "PC" else PP_COL, va="center")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("η (central → portal)"); ax.set_ylabel("log2(expression / mean), raw bins")
    ax.set_title(f"C  Independent markers along η\n(red: expected PC, blue: expected PP; "
                 f"{int(val.agrees.fillna(False).sum())}/{int(val.agrees.notna().sum())} tested agree)")

    ax = fig.add_subplot(gs[1, 0])
    nl = res[~res.landmark_gene]
    ax.scatter(nl.log2FC_PP_vs_PC, -np.log10(nl.FDR.clip(1e-300)), s=3, c="#bbbbbb", rasterized=True)
    r = nl[nl.robust]
    grad = r.pattern == "parenchymal_gradient"
    ax.scatter(r.log2FC_PP_vs_PC[~grad], -np.log10(r.FDR[~grad].clip(1e-300)), s=7, facecolors="none",
               edgecolors=np.where(r.log2FC_PP_vs_PC[~grad] > 0, PP_COL, PC_COL), lw=0.6,
               label="boundary-restricted", rasterized=True)
    ax.scatter(r.log2FC_PP_vs_PC[grad], -np.log10(r.FDR[grad].clip(1e-300)), s=6,
               c=np.where(r.log2FC_PP_vs_PC[grad] > 0, PP_COL, PC_COL), label="parenchymal gradient",
               rasterized=True)
    ax.legend(fontsize=8, loc="upper left")
    rg = r[grad]
    for _, row in pd.concat([rg.nsmallest(12, "log2FC_PP_vs_PC"), rg.nlargest(12, "log2FC_PP_vs_PC")]).iterrows():
        ax.annotate(row.gene, (row.log2FC_PP_vs_PC, -np.log10(max(row.FDR, 1e-300))), fontsize=7)
    ax.set_xlabel("log2 FC  portal layer / central layer (block-adjusted)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title(f"D  Genome-wide zonation: {int((r.direction == 'pericentral').sum())} PC, "
                 f"{int((r.direction == 'periportal').sum())} PP robust genes")

    ax = fig.add_subplot(gs[1, 1])
    prof_cols = [c for c in res.columns if c.startswith("profile_L")]
    rg = r[r.parenchymal_zonated]
    top = pd.concat([rg.nsmallest(30, "log2FC_PP_vs_PC"), rg.nlargest(30, "log2FC_PP_vs_PC")]).sort_values("COM_eta")
    M = top[prof_cols].to_numpy()
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.gene, fontsize=6)
    ax.set_xticks(range(len(prof_cols))); ax.set_xticklabels([f"{(k + .5) / 8:.2f}" for k in range(8)], fontsize=7)
    ax.set_xlabel("η layer"); ax.set_title("E  Top 30 PC / PP parenchymal-zonated genes (z-scored)")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(res.log2FC_top_half, res.log2FC_bottom_half, s=2, c="#bbbbbb", rasterized=True)
    ax.scatter(r.log2FC_top_half, r.log2FC_bottom_half, s=4,
               c=np.where(r.log2FC_PP_vs_PC > 0, PP_COL, PC_COL), rasterized=True)
    rho = stats.spearmanr(res.log2FC_top_half, res.log2FC_bottom_half)[0]
    z = res[res.zonated]
    rho_z = stats.spearmanr(z.log2FC_top_half, z.log2FC_bottom_half)[0]
    same_z = np.mean(np.sign(z.log2FC_top_half) == np.sign(z.log2FC_bottom_half))
    lim = np.nanpercentile(np.abs(np.r_[res.log2FC_top_half, res.log2FC_bottom_half]), 99.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
    ax.set_xlabel("log2FC, top half (lower depth)"); ax.set_ylabel("log2FC, bottom half (higher depth)")
    txt = "\n".join(f"{s.variant}: ρ={s.spearman_all:.2f}, sign={s.sign_agreement_robust:.0%}" for s in sens.itertuples())
    ax.set_title(f"F  Split-half replication: zonated genes ρ={rho_z:.2f}, same sign {same_z:.0%}\n"
                 f"(all {len(res)} tested genes ρ={rho:.2f}; coloured = robust)")
    ax.text(0.02, 0.98, "Landmark-variant concordance:\n" + txt, transform=ax.transAxes, fontsize=6.5,
            va="top", family="monospace", bbox=dict(fc="white", ec="#cccccc"))

    fig.tight_layout()
    fig.savefig(OUT / "figures/01_lobular_axis_zonation.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "figures/01_lobular_axis_zonation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
