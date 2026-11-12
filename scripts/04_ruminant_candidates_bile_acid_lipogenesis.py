"""Analysis 4: are bile-acid synthesis and de novo lipogenesis more pericentral in cow?

Gene-by-gene and pooled, with per-animal controls (pig n=5, human non-steatotic donors M4-M8 n=5,
mouse n=6; all Visium, Yakubovsky 2026) and cow = independent Visium cows (n=2) + this study (n=1).

Portal bias = log2(mean last 2 layers / mean first 2 layers); negative = pericentral.
  raw    : Visium animals only (one platform, one pipeline), cows vs controls
  scaled : portal bias / |portal bias of pooled CYP2E1+CYP1A2| in the same animal, which removes
           platform / resolution dynamic range so this study's Stereo-seq can be included
This study gets 95% CIs from a spatial-block bootstrap (within-section variability only).
Animal QC (genes not under test): pooled CYP2E1+CYP1A2 bias <= -0.3 and HAL periportal when detected.
COW-C1 (CYP2E1 -0.14, GLUL +0.19) and BOAR-P2 (HAL -1.94) fail, so their lobule axes are not trusted.
Tests on scaled values, one-sided (cow more pericentral):
  cow_p  : exact permutation, cows = this study + QC-passing Visium cow(s) vs QC-passing controls
  ours_p : rank of this study among controls, (1 + #controls <= ours) / (n + 1)
A gene is used in an animal only if its max layer expression >= MIN_EXPR (UMI fraction).
"""
import importlib
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lobule_utils import BIN_UM, OUT, bh, load_bins, pseudobulk, batched_poisson_glm

z01 = importlib.import_module("01_lobular_axis_zonation")
z03 = importlib.import_module("03_cross_species_zonation")
CSV = OUT / "data/reference/csv"

PANELS = {
    "Bile-acid synthesis enzymes": ["CYP7A1", "CYP8B1", "CYP27A1", "CYP7B1", "HSD3B7", "AKR1D1", "SLC27A5",
                                    "ACOX2", "AMACR", "HSD17B4", "SCP2"],
    "Bile-acid regulation / transport": ["NR0B2", "NR1H4", "SLC10A1", "ABCB11"],
    "De novo lipogenesis enzymes": ["ACLY", "ACACA", "FASN", "SCD", "ELOVL6", "ME1", "ACSS2"],
    "Lipogenic regulators": ["SREBF1", "MLXIPL", "THRSP", "INSIG1"],
}
SCALE_GENES = ["CYP2E1", "CYP1A2"]
MIN_EXPR = 2e-5
MIN_EXPR_STRICT = 5e-5
N_BOOT = 500
RNG = np.random.default_rng(7)
SPECIES_COL = {"cow (this study)": "#000000", "cow Visium": "#8e44ad", "pig": "#e67e22",
               "human": "#16a085", "mouse": "#7f8c8d"}


# ------------------------------------------------------------------ reference animals
def reference_animals(cow_to_human, mouse_to_human):
    """Return {animal: (species, DataFrame[human symbol -> layers array])} for Visium animals."""
    specs = [("cow Visium", "nonhuman_COW-*.csv", "Gene", cow_to_human),
             ("pig", "nonhuman_PIG-*.csv", "Gene", str.upper),
             ("pig", "nonhuman_BOAR-*.csv", "Gene", str.upper),
             ("mouse", "nonhuman_MUS-*.csv", "Gene", mouse_to_human)]
    animals = {}
    for species, pat, col, fn in specs:
        for f in sorted(CSV.glob(pat)):
            animals[f.stem.replace("nonhuman_", "")] = (species, _layers(f, col, fn, "Mean lobule layer"))
    for s in ["M4", "M5", "M6", "M7", "M8"]:  # non-steatotic live donors (combined n=5 table)
        animals[f"HUMAN-{s}"] = ("human", _layers(CSV / f"human_sample_{s}.csv", None, lambda x: x, "Mean_Zone_"))
    return animals


def _layers(path, col, to_human, prefix):
    d = pd.read_csv(path)
    lab = d.iloc[:, 0] if col is None else d[col]
    L = [c for c in d.columns if c.startswith(prefix)]
    X = d[L].to_numpy(float)
    out = pd.DataFrame(X, columns=range(X.shape[1]))
    out["human"] = lab.map(to_human).to_numpy()
    out["mx"] = X.max(1)
    return out.dropna(subset=["human"]).sort_values("mx", ascending=False).drop_duplicates("human").set_index("human")


def animal_qc(tab):
    lay = [c for c in tab.columns if isinstance(c, int)]
    ok = tab[tab.mx >= MIN_EXPR]
    sc = [g for g in SCALE_GENES if g in ok.index]
    cyp = pb_from_layers(ok.loc[sc, lay].to_numpy(float).sum(0))[0] if sc else np.nan
    hal = pb_from_layers(ok.loc["HAL", lay].to_numpy(float))[0] if "HAL" in ok.index else np.nan
    return dict(cyp_bias=cyp, hal_bias=hal, qc_pass=bool(cyp <= -0.3 and not (np.isfinite(hal) and hal <= 0)))


def pb_from_layers(X):
    X = np.atleast_2d(X)
    return np.log2((X[:, -2:].mean(1) + 1e-6) / (X[:, :2].mean(1) + 1e-6))


def animal_values(tab, min_expr):
    """Per-gene, per-panel pooled and CYP-scale portal bias for one reference animal."""
    lay = [c for c in tab.columns if isinstance(c, int)]
    ok = tab[tab.mx >= min_expr]
    vals = {g: pb_from_layers(ok.loc[g, lay].to_numpy(float))[0] for g in itertools.chain(*PANELS.values())
            if g in ok.index}
    for p, genes in PANELS.items():
        gs = [g for g in genes if g in ok.index]
        if gs:
            vals[f"[pooled] {p}"] = pb_from_layers(ok.loc[gs, lay].to_numpy(float).sum(0))[0]
    sc = [g for g in SCALE_GENES if g in ok.index]
    vals["_scale"] = abs(pb_from_layers(ok.loc[sc, lay].to_numpy(float).sum(0))[0]) if sc else np.nan
    return vals


# ------------------------------------------------------------------ this study + block bootstrap
def this_study(cow_to_human):
    res = pd.read_csv(OUT / "results/01_zonation_genomewide.csv")
    coord = pd.read_csv(OUT / "data/01_bin50_lobular_coordinate.csv.gz", index_col=0)
    a = load_bins()
    valid = coord["valid"].to_numpy()
    layer = np.where(valid, coord["layer"].to_numpy(), 0)
    bx = (a.obs["ix"].to_numpy() * BIN_UM // z01.BLOCK_UM).astype(int)
    by = (a.obs["iy"].to_numpy() * BIN_UM // z01.BLOCK_UM).astype(int)
    counts, meta = pseudobulk(a, valid, layer, bx * 100 + by)
    tested = set(res.gene)
    h2cow = {}
    for g in res.gene:
        h = cow_to_human(g)
        if h and h not in h2cow:
            h2cow[h] = g
    cols, names = [], []
    for g in itertools.chain(*PANELS.values()):
        if g in h2cow:
            cols.append(counts[:, a.var_names.get_loc(h2cow[g])]); names.append(g)
    for p, genes in PANELS.items():
        gs = [h2cow[g] for g in genes if g in h2cow]
        if gs:
            cols.append(counts[:, [a.var_names.get_loc(x) for x in gs]].sum(1)); names.append(f"[pooled] {p}")
    sc = [h2cow[g] for g in SCALE_GENES if g in h2cow]
    cols.append(counts[:, [a.var_names.get_loc(x) for x in sc]].sum(1)); names.append("_scale")
    Y = np.column_stack(cols)
    lib = meta["lib"].to_numpy(float)
    mids = (np.arange(z01.N_LAYERS) + 0.5) / z01.N_LAYERS

    def fit(Yb, mb, libb):
        _, X_full, predict, k0 = z01.design(mb, mids)
        beta, _ = batched_poisson_glm(Yb, X_full, np.log(libb))
        prof = predict(mids) @ beta[:, k0:].T
        return (prof[6:].mean(0) - prof[:2].mean(0)) / np.log(2)

    est = fit(Y, meta, lib)
    blocks = meta["block"].unique()
    boots = []
    for _ in range(N_BOOT):
        pick = RNG.choice(blocks, size=len(blocks), replace=True)
        idx, newb = [], []
        for i, b in enumerate(pick):
            r = np.where(meta["block"].to_numpy() == b)[0]
            idx.extend(r); newb.extend([f"{b}_{i}"] * len(r))
        mb = meta.iloc[idx].reset_index(drop=True).copy()
        mb["block"] = newb
        boots.append(fit(Y[idx], mb, lib[idx]))
    B = np.array(boots)
    out = pd.DataFrame({"est": est, "lo": np.percentile(B, 2.5, 0), "hi": np.percentile(B, 97.5, 0)}, index=names)
    scale_boot = np.abs(B[:, names.index("_scale")])
    sc_est = abs(est[names.index("_scale")])
    out["scaled_est"] = out.est / sc_est
    out["scaled_lo"] = np.percentile(B / scale_boot[:, None], 2.5, 0)
    out["scaled_hi"] = np.percentile(B / scale_boot[:, None], 97.5, 0)
    out["p_boot_pericentral"] = (B >= 0).mean(0)          # fraction of resamples not pericentral
    out["tested_in_01"] = [n.startswith("[") or n == "_scale" or h2cow.get(n) in tested for n in names]
    return out


def ours_gene_profiles(cow_to_human):
    """Pseudobulk counts for panel genes + scale genes in this study (for leave-one-out pooling)."""
    res = pd.read_csv(OUT / "results/01_zonation_genomewide.csv")
    coord = pd.read_csv(OUT / "data/01_bin50_lobular_coordinate.csv.gz", index_col=0)
    a = load_bins()
    valid = coord["valid"].to_numpy()
    layer = np.where(valid, coord["layer"].to_numpy(), 0)
    bx = (a.obs["ix"].to_numpy() * BIN_UM // z01.BLOCK_UM).astype(int)
    by = (a.obs["iy"].to_numpy() * BIN_UM // z01.BLOCK_UM).astype(int)
    counts, meta = pseudobulk(a, valid, layer, bx * 100 + by)
    tested = set(res.gene)
    cols = {}
    for g in res.gene:
        h = cow_to_human(g)
        if h and h not in cols and g in tested:
            cols[h] = counts[:, a.var_names.get_loc(g)]
    return dict(cols=cols, meta=meta)


def ours_pooled_subset(og, keep, panel):
    mids = (np.arange(z01.N_LAYERS) + 0.5) / z01.N_LAYERS
    gs = [g for g in keep if g in og["cols"]]
    sc = [g for g in SCALE_GENES if g in og["cols"]]
    if not gs:
        return np.nan
    Y = np.column_stack([sum(og["cols"][g] for g in gs), sum(og["cols"][g] for g in sc)])
    _, X_full, predict, k0 = z01.design(og["meta"], mids)
    beta, _ = batched_poisson_glm(Y, X_full, np.log(og["meta"]["lib"].to_numpy(float)))
    prof = predict(mids) @ beta[:, k0:].T
    pbv = (prof[6:].mean(0) - prof[:2].mean(0)) / np.log(2)
    return pbv[0] / abs(pbv[1])


# ------------------------------------------------------------------ permutation test
def perm_test(cow, ctrl):
    """One-sided exact permutation: is mean(cow) < mean(ctrl)? Returns diff, p, n_perm."""
    cow, ctrl = np.asarray(cow, float), np.asarray(ctrl, float)
    allv = np.r_[cow, ctrl]
    k = len(cow)
    obs = cow.mean() - ctrl.mean()
    combs = list(itertools.combinations(range(len(allv)), k))
    null = np.array([allv[list(c)].mean() - np.delete(allv, list(c)).mean() for c in combs])
    return obs, float(np.mean(null <= obs + 1e-12)), len(combs)


def main():
    cow_to_human, mouse_to_human, _ = z03.load_orthologs()
    animals = reference_animals(cow_to_human, mouse_to_human)
    ours = this_study(cow_to_human)
    ours.to_csv(OUT / "results/04_this_study_bootstrap.csv")

    qc = pd.DataFrame({a: {"species": sp, **animal_qc(t)} for a, (sp, t) in animals.items()}).T
    qc.to_csv(OUT / "results/04_animal_qc.csv")
    print(qc.to_string())
    animals = {a: v for a, v in animals.items() if qc.loc[a, "qc_pass"]}

    features = list(itertools.chain(*PANELS.values())) + [f"[pooled] {p}" for p in PANELS]
    rows = []
    for tier, thr in [("main", MIN_EXPR), ("strict", MIN_EXPR_STRICT)]:
        vals = {a: animal_values(t, thr) for a, (s, t) in animals.items()}
        species = {a: s for a, (s, t) in animals.items()}
        for f in features:
            raw = {a: v.get(f, np.nan) for a, v in vals.items()}
            scl = {a: raw[a] / vals[a]["_scale"] for a in raw}
            ours_ok = f in ours.index and ours.loc[f, "tested_in_01"]
            cow_vis = [scl[a] for a in scl if species[a] == "cow Visium" and np.isfinite(scl[a])]
            ctrl = [scl[a] for a in scl if species[a] != "cow Visium" and np.isfinite(scl[a])]
            cows = ([ours.loc[f, "scaled_est"]] if ours_ok else []) + cow_vis
            r = dict(tier=tier, feature=f,
                     panel=next((p for p, g in PANELS.items() if f in g or f.endswith(p)), ""),
                     this_study=ours.loc[f, "est"] if ours_ok else np.nan,
                     this_study_scaled=ours.loc[f, "scaled_est"] if ours_ok else np.nan,
                     this_study_scaled_lo=ours.loc[f, "scaled_lo"] if ours_ok else np.nan,
                     this_study_scaled_hi=ours.loc[f, "scaled_hi"] if ours_ok else np.nan,
                     this_study_p_boot=ours.loc[f, "p_boot_pericentral"] if ours_ok else np.nan,
                     cow_visium_scaled=np.mean(cow_vis) if cow_vis else np.nan,
                     n_cows=len(cows), n_controls=len(ctrl))
            for sp in ["pig", "human", "mouse"]:
                v = [scl[a] for a in scl if species[a] == sp and np.isfinite(scl[a])]
                r[f"{sp}_mean_scaled"] = np.mean(v) if v else np.nan
                r[f"{sp}_n"] = len(v)
            if len(cows) >= 2 and len(ctrl) >= 6:
                r["cow_diff"], r["cow_p"], r["cow_nperm"] = perm_test(cows, ctrl)
            if ours_ok and len(ctrl) >= 6:
                r["ours_p"] = (1 + sum(c <= r["this_study_scaled"] for c in ctrl)) / (len(ctrl) + 1)
                r["ours_ci_below_all_controls"] = r["this_study_scaled_hi"] < min(ctrl)
            if cow_vis and len(ctrl) >= 6:
                r["cow_visium_rank_p"] = (1 + sum(c <= np.mean(cow_vis) for c in ctrl)) / (len(ctrl) + 1)
            means = [r[f"{sp}_mean_scaled"] for sp in ["pig", "human", "mouse"] if np.isfinite(r[f"{sp}_mean_scaled"])]
            r["cow_mean_scaled"] = np.mean(cows) if cows else np.nan
            r["cow_below_all_control_species"] = bool(means) and np.isfinite(r["cow_mean_scaled"]) and r["cow_mean_scaled"] < min(means)
            r["cow_sources_agree_PC"] = (len(cows) >= 2) and all(c < 0 for c in cows)
            rows.append(r)
    res = pd.DataFrame(rows)
    for tier in res.tier.unique():
        for col in ["cow_p", "ours_p"]:
            m = (res.tier == tier) & res[col].notna() & ~res.feature.str.startswith("[")
            res.loc[m, col.replace("_p", "_FDR_genes")] = bh(res.loc[m, col])
    res.to_csv(OUT / "results/04_gene_tests.csv", index=False)

    # leave-one-gene-out for the pooled enzyme panels (scaled, main tier)
    vals = {a: animal_values(t, MIN_EXPR) for a, (s, t) in animals.items()}
    species = {a: s for a, (s, t) in animals.items()}
    loo_rows = []
    ours_genes = ours_gene_profiles(cow_to_human)
    for p in ["Bile-acid synthesis enzymes", "De novo lipogenesis enzymes"]:
        for drop in [None] + PANELS[p]:
            keep = [g for g in PANELS[p] if g != drop]
            v = {}
            for a, (s, t) in animals.items():
                lay = [c for c in t.columns if isinstance(c, int)]
                ok = t[(t.mx >= MIN_EXPR) & t.index.isin(keep)]
                v[a] = pb_from_layers(ok[lay].to_numpy(float).sum(0))[0] / vals[a]["_scale"] if len(ok) else np.nan
            cow_vis = [v[a] for a in v if species[a] == "cow Visium" and np.isfinite(v[a])]
            ctrl = [v[a] for a in v if species[a] != "cow Visium" and np.isfinite(v[a])]
            ours_v = ours_pooled_subset(ours_genes, keep, p)
            cows = ([ours_v] if np.isfinite(ours_v) else []) + cow_vis
            d, pv, _ = perm_test(cows, ctrl) if len(cows) >= 2 else (np.nan, np.nan, 0)
            loo_rows.append(dict(panel=p, dropped=drop or "(none)", this_study_scaled=ours_v,
                                 cow_visium_scaled=np.mean(cow_vis) if cow_vis else np.nan,
                                 controls_scaled=np.mean(ctrl), diff=d, p=pv))
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(OUT / "results/04_leave_one_gene_out.csv", index=False)

    show = ["feature", "this_study_scaled", "this_study_scaled_lo", "this_study_scaled_hi", "cow_visium_scaled",
            "pig_mean_scaled", "human_mean_scaled", "mouse_mean_scaled", "n_controls", "cow_p", "ours_p",
            "ours_ci_below_all_controls", "cow_below_all_control_species", "cow_sources_agree_PC"]
    pd.set_option("display.width", 250)
    print(res[res.tier == "main"][show].round(3).to_string(index=False))
    print(loo.round(3).to_string(index=False))
    s = res[(res.tier == "strict") & res.feature.str.startswith("[")][["feature", "cow_p", "ours_p", "cow_mean_scaled",
                                                                         "pig_mean_scaled", "human_mean_scaled", "mouse_mean_scaled"]]
    print("strict tier pooled:\n", s.round(3).to_string(index=False))
    make_figure(res[res.tier == "main"], ours, animals, loo, qc)


def make_figure(res, ours, animals, loo, qc):
    vals = {a: animal_values(t, MIN_EXPR) for a, (s, t) in animals.items()}
    species = {a: s for a, (s, t) in animals.items()}
    panels = ["Bile-acid synthesis enzymes", "Bile-acid regulation / transport",
              "De novo lipogenesis enzymes", "Lipogenic regulators"]
    fig = plt.figure(figsize=(22, 17))
    gsp = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.3)
    jitter = {"cow Visium": -0.2, "pig": -0.07, "human": 0.07, "mouse": 0.2}
    for k, p in enumerate(panels):
        ax = fig.add_subplot(gsp[k // 2, (k % 2) * 2:(k % 2) * 2 + 2])
        feats = PANELS[p] + [f"[pooled] {p}"]
        for i, f in enumerate(feats):
            for a, v in vals.items():
                if f in v and np.isfinite(v[f]) and np.isfinite(v["_scale"]):
                    s = species[a]
                    ax.scatter(i + jitter[s] + RNG.uniform(-0.03, 0.03), v[f] / v["_scale"], s=26,
                               color=SPECIES_COL[s], alpha=0.85, edgecolor="none", zorder=3)
            if f in ours.index and ours.loc[f, "tested_in_01"]:
                o = ours.loc[f]
                ax.errorbar(i, o.scaled_est, yerr=[[o.scaled_est - o.scaled_lo], [o.scaled_hi - o.scaled_est]],
                            fmt="*", ms=13, color="k", capsize=3, zorder=4)
            r = res[res.feature == f]
            if len(r):
                rr = r.iloc[0]
                txt = []
                if np.isfinite(rr.get("cow_p", np.nan)):
                    txt.append(f"cows p={rr.cow_p:.3f}")
                if np.isfinite(rr.get("ours_p", np.nan)):
                    txt.append(f"ours p={rr.ours_p:.2f}")
                ax.text(i, 2.4, "\n".join(txt), ha="center", fontsize=6.3, va="top")
        ax.axhline(0, c="k", lw=0.6)
        ax.axvspan(len(feats) - 1.5, len(feats) - 0.5, color="#f2f2f2", zorder=0)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels([f.replace("[pooled] ", "POOLED\n") if f.startswith("[") else f for f in feats], fontsize=8)
        ax.set_ylim(-2.6, 2.5)
        ax.set_ylabel("portal bias / |CYP2E1+CYP1A2 bias|\n(<0 = pericentral)")
        ax.set_title(f"{'ABCD'[k]}  {p}", fontsize=11, loc="left")
    handles = [plt.Line2D([], [], marker="*", ls="", color="k", ms=12, label="cow, this study (95% block-bootstrap CI)")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in SPECIES_COL.items() if s != "cow (this study)"]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.94))

    for j, p in enumerate(["Bile-acid synthesis enzymes", "De novo lipogenesis enzymes"]):
        ax = fig.add_subplot(gsp[2, j * 2:j * 2 + 2])
        l = loo[loo.panel == p]
        x = np.arange(len(l))
        ax.bar(x - 0.27, l.this_study_scaled, 0.27, color="k", label="cow, this study")
        ax.bar(x, l.cow_visium_scaled, 0.27, color=SPECIES_COL["cow Visium"], label="cow Visium (C2)")
        ax.bar(x + 0.27, l.controls_scaled, 0.27, color="#bdc3c7", label="controls mean (pig, human, mouse)")
        lo = np.nanmin(l[["this_study_scaled", "cow_visium_scaled", "controls_scaled"]].to_numpy())
        for xi, pv in zip(x, l.p):
            if np.isfinite(pv):
                ax.text(xi, lo - 0.08, f"{pv:.3f}", ha="center", va="top", fontsize=7)
        ax.set_ylim(lo - 0.3, None)
        ax.set_xticks(x); ax.set_xticklabels(["none" if d == "(none)" else f"−{d}" for d in l.dropped], fontsize=8, rotation=30)
        ax.axhline(0, c="k", lw=0.6)
        ax.set_ylabel("pooled scaled portal bias")
        ax.legend(fontsize=8)
        ax.set_title(f"{'EF'[j]}  {p}: leave-one-gene-out (numbers = cows vs controls permutation p)", fontsize=10, loc="left")

    dropped = ", ".join(qc.index[~qc.qc_pass.astype(bool)])
    fig.suptitle("Candidate ruminant features: bile-acid synthesis and de novo lipogenesis zonation, cow vs controls\n"
                 f"(values scaled by each animal's own CYP2E1+CYP1A2 gradient; animals failing axis QC excluded: {dropped})",
                 fontsize=13, y=0.99)
    fig.savefig(OUT / "figures/04_bile_acid_lipogenesis_cow_vs_controls.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figures/04_bile_acid_lipogenesis_cow_vs_controls.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
