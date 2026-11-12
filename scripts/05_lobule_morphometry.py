"""Analysis 4b: lobule segmentation and morphometry (CV-PT distances, lobule size in um).

Two landmark sets:
  marker    : the analysis 01 landmarks (GLUL consensus >= 7/9 domains = central veins; biliary-positive
              portal foci = portal tracts). Conservative, but incomplete: GLUL is detected in ~4% of bins.
  signature : local extrema of a smoothed zonation signature (mean z of 01's parenchymal-zonated PC genes
              minus PP genes; GLUL and portal-structure genes excluded). Minima-free: PC peaks = CV
              candidates, PP peaks = PT candidates. The genes were chosen with the marker landmarks, so in
              regions that already have markers the two sets agree by construction. The point is coverage
              of regions that have none.

Morphometry (1 bin = 25 um; coordinates are a 2D section, so sizes are apparent sizes):
  * CV / PT component size and shape (marker set)
  * nearest-neighbour centroid distances CV-CV, PT-PT, CV->PT, PT->CV
  * CV-centred Voronoi lobules inside tissue: area, equivalent diameter, neighbours, circularity,
    portal tracts per lobule; cells touching the tissue edge or reaching > 1.5 mm from their CV dropped
  * zonation scaling: does the pericentral zone have a fixed width in um, or a fixed fraction of the
    CV-PT span? Half-max of a pericentral score vs d_CV (um) and vs eta, by span tertile, with a
    spatial-block bootstrap.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import regionprops

sys.path.insert(0, str(Path(__file__).parent))
from lobule_utils import BIN_UM, LANDMARK_GENES, OUT, grid_shape, load_bins

SIG_SMOOTH = 2           # bins (50 um) for the signature map
DETREND = 60             # bins (1.5 mm) large-scale background removed from the signature
PEAK_MIN_DIST = 8        # bins (200 um) between peaks of the same kind
PEAK_QUANTILE = 0.90     # peak must exceed this quantile of the signature (or of its negative)
EDGE_BINS = 4            # ignore peaks within 100 um of the tissue edge
MAX_LOBULE_REACH_UM = 1500
N_BOOT = 300
RNG = np.random.default_rng(11)


MERGE_BINS = 2            # marker fragments within ~100 um are one landmark


def components(mask):
    grouped, _ = ndimage.label(ndimage.binary_dilation(mask, iterations=MERGE_BINS))
    lab = np.where(mask, grouped, 0)
    props = regionprops(lab)
    rows = [dict(label=p.label, n_bins=p.area, area_um2=p.area * BIN_UM ** 2,
                 eq_diam_um=p.equivalent_diameter_area * BIN_UM,
                 major_um=p.axis_major_length * BIN_UM, minor_um=p.axis_minor_length * BIN_UM,
                 eccentricity=p.eccentricity, cy=p.centroid[0], cx=p.centroid[1]) for p in props]
    return lab, pd.DataFrame(rows)


def signature_landmarks(a, res, tissue, shape):
    ix, iy = a.obs["ix"].to_numpy(), a.obs["iy"].to_numpy()
    z = res[res.parenchymal_zonated & ~res.gene.isin(LANDMARK_GENES)]
    pc = z[z.direction == "pericentral"].gene.tolist()
    pp = z[z.direction == "periportal"].gene.tolist()
    lib = a.obs["lib"].to_numpy()

    def zmean(genes):
        m = np.log1p(a.X[:, [a.var_names.get_loc(g) for g in genes]].toarray() * 1e4 / lib[:, None])
        return ((m - m.mean(0)) / (m.std(0) + 1e-9)).mean(1)
    score = zmean(pc) - zmean(pp)                      # high = pericentral
    img = np.zeros(shape); cnt = np.zeros(shape)
    np.add.at(img, (iy, ix), score); np.add.at(cnt, (iy, ix), 1)

    def sm(s):
        return ndimage.gaussian_filter(img, s) / np.maximum(ndimage.gaussian_filter(cnt, s), 1e-9)
    sig = np.where(tissue, sm(SIG_SMOOTH) - sm(DETREND), 0.0)
    inner = ndimage.binary_erosion(tissue, iterations=EDGE_BINS)
    peaks = {}
    for kind, s in [("CV", sig), ("PT", -sig)]:
        thr = np.quantile(s[tissue], PEAK_QUANTILE)
        pk = peak_local_max(np.where(inner, s, s.min()), min_distance=PEAK_MIN_DIST, threshold_abs=thr,
                            exclude_border=False)
        peaks[kind] = pd.DataFrame(pk, columns=["cy", "cx"])
    # PP genes only (independent of the pericentral score used in the scaling test)
    img_pp = np.zeros(shape); np.add.at(img_pp, (iy, ix), zmean(pp))
    sm_pp = lambda s_: ndimage.gaussian_filter(img_pp, s_) / np.maximum(ndimage.gaussian_filter(cnt, s_), 1e-9)
    sig_pp = np.where(tissue, sm_pp(SIG_SMOOTH) - sm_pp(DETREND), 0.0)
    thr = np.quantile(sig_pp[tissue], PEAK_QUANTILE)
    pk = peak_local_max(np.where(inner, sig_pp, sig_pp.min()), min_distance=PEAK_MIN_DIST, threshold_abs=thr,
                        exclude_border=False)
    peaks["PT_pp_only"] = pd.DataFrame(pk, columns=["cy", "cx"])
    return sig, peaks, len(pc), len(pp)


def nn_distances(A, B=None):
    """Nearest-neighbour centroid distance (um) from each point of A to B (or to the rest of A)."""
    A = np.asarray(A, float)
    if B is None:
        d, _ = cKDTree(A).query(A, k=2)
        return d[:, 1] * BIN_UM
    d, _ = cKDTree(np.asarray(B, float)).query(A)
    return d * BIN_UM


def voronoi_lobules(cv_pts, pt_pts, tissue):
    """CV-centred Voronoi cells clipped to tissue."""
    yy, xx = np.nonzero(tissue)
    d, owner = cKDTree(np.asarray(cv_pts, float)).query(np.c_[yy, xx])
    lab = np.zeros(tissue.shape, int)
    lab[yy, xx] = owner + 1
    reach = np.zeros(len(cv_pts))
    np.maximum.at(reach, owner, d * BIN_UM)
    edge = ~ndimage.binary_erosion(tissue, iterations=1) & tissue
    touches_edge = np.zeros(len(cv_pts), bool)
    touches_edge[np.unique(lab[edge]) - 1] = True
    pt_owner = cKDTree(np.asarray(cv_pts, float)).query(np.asarray(pt_pts, float))[1] if len(pt_pts) else []
    rows = []
    for k in range(len(cv_pts)):
        m = lab == k + 1
        if not m.any():
            continue
        area = m.sum()
        perim = (m & ~ndimage.binary_erosion(m)).sum()
        neigh = set(np.unique(lab[ndimage.binary_dilation(m) & ~m])) - {0, k + 1}
        rows.append(dict(lobule=k + 1, area_um2=area * BIN_UM ** 2,
                         eq_diam_um=2 * np.sqrt(area / np.pi) * BIN_UM,
                         max_reach_um=reach[k], n_neighbours=len(neigh),
                         circularity=4 * np.pi * area / max(perim, 1) ** 2,
                         n_PT_inside=int(np.sum(np.asarray(pt_owner) == k)),
                         touches_edge=touches_edge[k]))
    df = pd.DataFrame(rows)
    df["complete"] = ~df.touches_edge & (df.max_reach_um <= MAX_LOBULE_REACH_UM)
    return lab, df


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), median=np.median(x), q25=np.percentile(x, 25), q75=np.percentile(x, 75),
                mean=x.mean()) if len(x) else dict(n=0)


def half_max(xs, ys):
    """Half-max position of a decreasing profile, read from a monotone (isotonic) fit so that
    noise in sparsely sampled bins cannot produce an early spurious crossing."""
    from sklearn.isotonic import IsotonicRegression
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys)
    if ok.sum() < 3:
        return np.nan
    xs, ys = xs[ok], IsotonicRegression(increasing=False).fit_transform(xs[ok], ys[ok])
    if ys[0] - ys[-1] <= 0:
        return np.nan
    target = ys[0] - (ys[0] - ys[-1]) / 2
    for i in range(1, len(ys)):
        if ys[i] <= target:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    return np.nan


def hybrid_coordinate(a, coord, cvm, ptm, pt_peaks, shape):
    """Same CVs; portal tracts = marker PT mask plus PP-signature peaks (disc of 2 bins)."""
    ix, iy = a.obs["ix"].to_numpy(), a.obs["iy"].to_numpy()
    pt = ptm.copy()
    disc = np.zeros(shape, bool)
    disc[pt_peaks.cy.to_numpy(), pt_peaks.cx.to_numpy()] = True
    pt |= ndimage.binary_dilation(disc, iterations=2) & ~ndimage.binary_dilation(cvm, iterations=4)
    d_cv = ndimage.distance_transform_edt(~cvm)[iy, ix] * BIN_UM
    d_pt = ndimage.distance_transform_edt(~pt)[iy, ix] * BIN_UM
    span = d_cv + d_pt
    out = coord.copy()
    out["d_CV_um"], out["d_PT_um"], out["span_um"] = d_cv, d_pt, span
    out["eta"] = d_cv / np.maximum(span, 1e-9)
    out["valid"] = (d_cv > 50) & (d_pt > 50) & (span <= 1500)
    return out


def zonation_scaling(a, coord, res):
    """Pericentral score vs d_CV (um) and eta, by CV-PT span tertile."""
    ok = coord.valid.to_numpy()
    z = res[res.parenchymal_zonated & ~res.gene.isin(LANDMARK_GENES) & (res.direction == "pericentral")]
    genes = z.gene.tolist()
    lib = a.obs["lib"].to_numpy()
    m = np.log1p(a.X[:, [a.var_names.get_loc(g) for g in genes]].toarray() * 1e4 / lib[:, None])
    score = ((m - m[ok].mean(0)) / (m[ok].std(0) + 1e-9)).mean(1)
    span = coord.span_um.to_numpy()
    terc = pd.qcut(span[ok], 3, labels=["short", "mid", "long"])
    df = pd.DataFrame({"score": score[ok], "d_cv": coord.d_CV_um.to_numpy()[ok], "eta": coord.eta.to_numpy()[ok],
                       "span": span[ok], "tertile": terc,
                       "block": (coord.x_um.to_numpy()[ok] // 2000).astype(int) * 100 +
                                (coord.y_um.to_numpy()[ok] // 2000).astype(int)})
    um_edges = np.arange(50, 1050, 75)
    eta_edges = np.linspace(0, 1, 11)

    def profiles(d):
        out = {}
        for t, sub in d.groupby("tertile", observed=True):
            pu = sub.groupby(pd.cut(sub.d_cv, um_edges), observed=False).score.mean()
            pe = sub.groupby(pd.cut(sub.eta, eta_edges), observed=False).score.mean()
            xu = (um_edges[:-1] + um_edges[1:]) / 2
            xe = (eta_edges[:-1] + eta_edges[1:]) / 2
            # restrict the um profile to distances that exist in this tertile (d_cv < median span)
            lim = sub.span.median()
            keep = xu < lim
            out[t] = dict(xu=xu[keep], pu=pu.to_numpy()[keep], xe=xe, pe=pe.to_numpy(),
                          hm_um=half_max(xu[keep], pu.to_numpy()[keep]), hm_eta=half_max(xe, pe.to_numpy()),
                          span_median=lim, n=len(sub))
        return out
    est = profiles(df)
    blocks = df.block.unique()
    boots = []
    for _ in range(N_BOOT):
        pick = RNG.choice(blocks, len(blocks), replace=True)
        d = pd.concat([df[df.block == b] for b in pick])
        p = profiles(d)
        boots.append({t: (p[t]["hm_um"], p[t]["hm_eta"]) for t in p})
    rows = []
    for t in ["short", "mid", "long"]:
        hu = np.array([b[t][0] for b in boots if t in b], float)
        he = np.array([b[t][1] for b in boots if t in b], float)
        rows.append(dict(tertile=t, n_bins=est[t]["n"], span_median_um=est[t]["span_median"],
                         halfmax_dCV_um=est[t]["hm_um"], halfmax_um_lo=np.nanpercentile(hu, 2.5),
                         halfmax_um_hi=np.nanpercentile(hu, 97.5),
                         halfmax_eta=est[t]["hm_eta"], halfmax_eta_lo=np.nanpercentile(he, 2.5),
                         halfmax_eta_hi=np.nanpercentile(he, 97.5)))
    tab = pd.DataFrame(rows)
    # Which description is more invariant across tertiles? CV of half-max positions per bootstrap.
    cv_um, cv_eta = [], []
    for b in boots:
        u = np.array([b[t][0] for t in ["short", "mid", "long"]], float)
        e = np.array([b[t][1] for t in ["short", "mid", "long"]], float)
        if np.all(np.isfinite(u)) and np.all(np.isfinite(e)):
            cv_um.append(u.std() / u.mean()); cv_eta.append(e.std() / e.mean())
    cv_um, cv_eta = np.array(cv_um), np.array(cv_eta)
    test = dict(genes=len(genes), cv_um_median=np.median(cv_um), cv_eta_median=np.median(cv_eta),
                frac_boot_eta_more_invariant=np.mean(cv_eta < cv_um),
                halfmax_um_long_minus_short=tab.halfmax_dCV_um.iloc[2] - tab.halfmax_dCV_um.iloc[0],
                halfmax_um_ratio_long_short=tab.halfmax_dCV_um.iloc[2] / tab.halfmax_dCV_um.iloc[0],
                span_ratio_long_short=tab.span_median_um.iloc[2] / tab.span_median_um.iloc[0])
    return tab, test, est


def main():
    a = load_bins()
    coord = pd.read_csv(OUT / "data/01_bin50_lobular_coordinate.csv.gz", index_col=0)
    res = pd.read_csv(OUT / "results/01_zonation_genomewide.csv")
    ix, iy = a.obs["ix"].to_numpy(), a.obs["iy"].to_numpy()
    shape = grid_shape(a)
    tissue = np.zeros(shape, bool); tissue[iy, ix] = True
    tissue = ndimage.binary_closing(tissue, iterations=2) & ndimage.binary_fill_holes(tissue)
    tissue_area_mm2 = tissue.sum() * BIN_UM ** 2 / 1e6

    # ---------------- marker landmarks
    cvm = np.zeros(shape, bool); cvm[iy[coord.is_CV_landmark], ix[coord.is_CV_landmark]] = True
    ptm = np.zeros(shape, bool); ptm[iy[coord.is_PT_landmark], ix[coord.is_PT_landmark]] = True
    _, cv_c = components(cvm)
    _, pt_c = components(ptm)
    cv_c["kind"], pt_c["kind"] = "CV (GLUL+ domain)", "PT (biliary focus)"
    comp = pd.concat([cv_c, pt_c])
    comp.to_csv(OUT / "results/05_marker_landmark_components.csv", index=False)

    # ---------------- signature landmarks
    sig, peaks, n_pc, n_pp = signature_landmarks(a, res, tissue, shape)
    # recall of marker CVs/PTs by signature peaks (within 150 um)
    rec = {}
    for kind, comp_df in [("CV", cv_c), ("PT", pt_c)]:
        d = nn_distances(comp_df[["cy", "cx"]], peaks[kind][["cy", "cx"]])
        rec[kind] = float(np.mean(d <= 150))
    # how many signature peaks are new (> 150 um from any marker landmark of that kind)
    new = {k: float(np.mean(nn_distances(peaks[k][["cy", "cx"]], (cv_c if k == "CV" else pt_c)[["cy", "cx"]]) > 150))
           for k in peaks}

    sets = {
        "marker": (cv_c[["cy", "cx"]].to_numpy(), pt_c[["cy", "cx"]].to_numpy()),
        "signature": (peaks["CV"][["cy", "cx"]].to_numpy(), peaks["PT"][["cy", "cx"]].to_numpy()),
    }
    dist_rows, lob_tabs, labs = [], {}, {}
    for name, (cvp, ptp) in sets.items():
        for metric, vals in [("CV-CV nearest", nn_distances(cvp)), ("PT-PT nearest", nn_distances(ptp)),
                             ("CV->nearest PT", nn_distances(cvp, ptp)), ("PT->nearest CV", nn_distances(ptp, cvp))]:
            dist_rows.append(dict(landmarks=name, metric=metric, **describe(vals)))
        lab, lob = voronoi_lobules(cvp, ptp, tissue)
        lob["landmarks"] = name
        lob_tabs[name], labs[name] = lob, lab
        c = lob[lob.complete]
        for metric, col in [("lobule area (um2)", "area_um2"), ("lobule eq. diameter (um)", "eq_diam_um"),
                            ("lobule neighbours", "n_neighbours"), ("lobule circularity", "circularity"),
                            ("PT per lobule", "n_PT_inside")]:
            dist_rows.append(dict(landmarks=name, metric=metric, **describe(c[col])))
        dist_rows.append(dict(landmarks=name, metric="CV density (per mm2)", n=len(cvp), median=len(cvp) / tissue_area_mm2))
        dist_rows.append(dict(landmarks=name, metric="PT density (per mm2)", n=len(ptp), median=len(ptp) / tissue_area_mm2))
        dist_rows.append(dict(landmarks=name, metric="PT:CV ratio", n=len(ptp), median=len(ptp) / len(cvp)))
    morph = pd.DataFrame(dist_rows)
    morph.to_csv(OUT / "results/05_morphometry_summary.csv", index=False)
    pd.concat(lob_tabs.values()).to_csv(OUT / "results/05_voronoi_lobules.csv", index=False)

    # ---------------- zonation scaling
    scal, test, prof = zonation_scaling(a, coord, res)
    hyb = hybrid_coordinate(a, coord, cvm, ptm, peaks["PT_pp_only"], shape)
    scal_h, test_h, prof_h = zonation_scaling(a, hyb, res)
    scal["portal_landmarks"], scal_h["portal_landmarks"] = "marker", "marker + PP-signature peaks"
    pd.concat([scal, scal_h]).to_csv(OUT / "results/05_zonation_scaling_by_span.csv", index=False)
    pd.DataFrame({"marker": test, "marker + PP-signature peaks": test_h,
                  }).to_csv(OUT / "results/05_zonation_scaling_test.csv")
    print("hybrid PT landmarks: valid bins", int(hyb.valid.sum()), "median span", round(float(hyb.span_um[hyb.valid].median())))

    qc = pd.Series({"tissue_area_mm2": tissue_area_mm2, "signature_PC_genes": n_pc, "signature_PP_genes": n_pp,
                    "signature_CV_peaks": len(peaks["CV"]), "signature_PT_peaks": len(peaks["PT"]),
                    "marker_CV_recalled_by_signature": rec["CV"], "marker_PT_recalled_by_signature": rec["PT"],
                    "signature_CV_peaks_new": new["CV"], "signature_PT_peaks_new": new["PT"]})
    qc.to_csv(OUT / "results/05_signature_landmark_qc.csv", header=["value"])
    pd.set_option("display.width", 200)
    print(qc.to_string()); print(comp.groupby("kind")[["n_bins", "eq_diam_um", "major_um", "minor_um", "eccentricity"]]
                                 .median().round(1).to_string())
    print(morph.round(2).to_string(index=False))
    print(pd.concat([scal, scal_h]).round(3).to_string(index=False))
    print(pd.DataFrame({"marker": test, "hybrid": test_h}).round(3).to_string())
    make_figure(coord, tissue, cvm, ptm, sig, peaks, labs, lob_tabs, morph, comp, scal, test, prof, sets,
                scal_h, test_h)


def make_figure(coord, tissue, cvm, ptm, sig, peaks, labs, lob_tabs, morph, comp, scal, test, prof, sets,
                scal_h, test_h):
    PC_COL, PP_COL = "#c0392b", "#2471a3"
    ext = [0, tissue.shape[1] * BIN_UM / 1000, tissue.shape[0] * BIN_UM / 1000, 0]
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.where(tissue, sig, np.nan), cmap="RdBu_r", extent=ext,
                   vmin=-np.nanpercentile(np.abs(sig[tissue]), 98), vmax=np.nanpercentile(np.abs(sig[tissue]), 98))
    for k, c, mk in [("CV", PC_COL, "o"), ("PT", PP_COL, "s")]:
        p = peaks[k]
        ax.scatter(p.cx * BIN_UM / 1000, p.cy * BIN_UM / 1000, s=14, facecolors="none", edgecolors="k", marker=mk, lw=0.7)
    ax.contour(cvm, levels=[0.5], colors=PC_COL, linewidths=0.8, extent=[0, ext[1], 0, ext[2]], origin="upper")
    ax.contour(ptm, levels=[0.5], colors=PP_COL, linewidths=0.8, extent=[0, ext[1], 0, ext[2]], origin="upper")
    ax.set_title("A  Zonation signature (red = pericentral)\ncoloured outlines: marker CV / PT; ○ □ signature peaks", fontsize=10)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    plt.colorbar(im, ax=ax, fraction=0.04)

    for j, name in enumerate(["marker", "signature"]):
        ax = fig.add_subplot(gs[0, 1 + j])
        lab = labs[name]
        lob = lob_tabs[name].set_index("lobule")
        rnd = np.random.default_rng(3).permutation(lab.max() + 1)
        show = np.where(lab > 0, rnd[lab] % 20, np.nan).astype(float)
        incomplete = np.isin(lab, lob.index[~lob.complete])
        show[incomplete] = np.nan
        ax.imshow(np.where(tissue, 0.0, np.nan), cmap="Greys", vmin=0, vmax=4, extent=ext)
        ax.imshow(show, cmap="tab20", extent=ext, alpha=0.75, interpolation="nearest")
        cvp, ptp = sets[name]
        ax.scatter(cvp[:, 1] * BIN_UM / 1000, cvp[:, 0] * BIN_UM / 1000, s=10, c=PC_COL)
        ax.scatter(ptp[:, 1] * BIN_UM / 1000, ptp[:, 0] * BIN_UM / 1000, s=10, c=PP_COL, marker="s")
        c = lob[lob.complete]
        ax.set_title(f"{'BC'[j]}  Voronoi lobules, {name} landmarks\n"
                     f"{len(c)} complete, median eq. diameter {c.eq_diam_um.median():.0f} µm", fontsize=10)
        ax.set_xlabel("mm")

    ax = fig.add_subplot(gs[0, 3])
    m = morph[morph.metric.isin(["CV-CV nearest", "PT-PT nearest", "CV->nearest PT", "PT->nearest CV"])]
    x = np.arange(4)
    for j, (name, off, col) in enumerate([("marker", -0.18, "#555555"), ("signature", 0.18, "#e67e22")]):
        s = m[m.landmarks == name].set_index("metric").loc[["CV-CV nearest", "PT-PT nearest", "CV->nearest PT", "PT->nearest CV"]]
        ax.bar(x + off, s["median"], 0.36, color=col, label=f"{name} landmarks")
        ax.errorbar(x + off, s["median"], yerr=[s["median"] - s.q25, s.q75 - s["median"]], fmt="none", c="k", capsize=3)
        for xi, (_, r) in zip(x + off, s.iterrows()):
            ax.text(xi, r.q75 + 25, f"{r['median']:.0f}\nn={int(r.n)}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(["CV–CV", "PT–PT", "CV→PT", "PT→CV"])
    ax.set_ylabel("nearest-neighbour centroid distance (µm), median ± IQR")
    ax.legend(fontsize=8)
    ax.set_title("D  Landmark spacing", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    for name, col in [("marker", "#555555"), ("signature", "#e67e22")]:
        c = lob_tabs[name][lob_tabs[name].complete]
        ax.hist(c.eq_diam_um, bins=np.arange(0, 2600, 150), alpha=0.55, color=col, label=f"{name} (n={len(c)})")
    ax.set_xlabel("lobule equivalent diameter (µm)"); ax.set_ylabel("lobules")
    ax.legend(fontsize=8)
    ax.set_title("E  Lobule size (CV-centred Voronoi, complete cells)", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    for k, col in [("CV (GLUL+ domain)", PC_COL), ("PT (biliary focus)", PP_COL)]:
        c = comp[comp.kind == k]
        ax.scatter(c.major_um, c.minor_um, s=np.clip(c.n_bins, 5, 200), c=col, alpha=0.6, label=f"{k} (n={len(c)})")
    lim = comp.major_um.max() * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=0.8)
    ax.set_xlabel("major axis (µm)"); ax.set_ylabel("minor axis (µm)")
    ax.legend(fontsize=8)
    ax.set_title("F  Marker landmark size and elongation\n(GLUL+ domain = pericentral hepatocyte cuff, not lumen)", fontsize=10)

    cols = {"short": "#1b9e77", "mid": "#d95f02", "long": "#7570b3"}
    for j, (key_x, key_y, xl, hm, ttl) in enumerate([
            ("xu", "pu", "distance from CV (µm)", "halfmax_dCV_um", "G  Pericentral score vs absolute distance"),
            ("xe", "pe", "η (fraction of CV–PT span)", "halfmax_eta", "H  Pericentral score vs relative position η")]):
        ax = fig.add_subplot(gs[1, 2 + j])
        for t in ["short", "mid", "long"]:
            p = prof[t]
            r = scal.set_index("tertile").loc[t]
            ax.plot(p[key_x], p[key_y], "o-", c=cols[t], ms=3,
                    label=f"{t} span (median {r.span_median_um:.0f} µm): half-max {r[hm]:.2f}" if j else
                          f"{t} span (median {r.span_median_um:.0f} µm): half-max {r[hm]:.0f} µm")
            ax.axvline(r[hm], c=cols[t], ls=":", lw=1)
        ax.set_xlabel(xl); ax.set_ylabel("pericentral signature (mean z)")
        ax.legend(fontsize=7.5)
        ax.set_title(ttl, fontsize=10)
    fig.text(0.74, 0.0, "Spread of half-max position across span tertiles (coefficient of variation, block-bootstrap median)\n"
             f"marker PTs: µm {test['cv_um_median']:.2f} vs η {test['cv_eta_median']:.2f} "
             f"(η more invariant in {test['frac_boot_eta_more_invariant']:.0%} of resamples);  "
             f"marker + PP-signature PTs: µm {test_h['cv_um_median']:.2f} vs η {test_h['cv_eta_median']:.2f} "
             f"(η more invariant in {test_h['frac_boot_eta_more_invariant']:.0%}); "
             f"half-max µm by tertile: {', '.join(f'{v:.0f}' for v in scal_h.halfmax_dCV_um)}",
             ha="center", fontsize=9)
    fig.savefig(OUT / "figures/05_lobule_morphometry.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figures/05_lobule_morphometry.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
