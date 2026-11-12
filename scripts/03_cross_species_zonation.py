"""Analysis 3: cross-species comparison of liver zonation (cow vs pig, human, mouse).

References (layer 1 = central vein in all; orientation checked with GLUL/CYP2E1 vs HAL/SDS):
  * Yakubovsky et al. 2026 Nature (PMC13216088), Supp. Tables 2 and 7, Visium, same pipeline:
      human live healthy donors (n=5, 8 layers, combined), cow (n=2), pig (3 domestic + 2 wild boar),
      mouse (n=6), each 6 layers
  * Halpern et al. 2017 Nature, Table S3: mouse hepatocyte scRNA-seq, 9 layers
  * Brosch et al. 2018 Nat Commun: human laser-capture (CV / IZ / PV), significant genes only

Common metric: portal bias = log2(mean of last 2 layers / mean of first 2 layers), pseudocount 1e-6
(expression as a UMI fraction). For this study's Stereo-seq data the same contrast is taken from
the block-adjusted 8-layer model profiles of analysis 01 (L7-8 vs L1-2).
Genes are matched on human symbols through Ensembl one-to-one cow-human(-mouse) orthologs.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from lobule_utils import OUT

REF = OUT / "data/reference"
EPS = 1e-6
MIN_MAX_EXPR = 5e-5      # Yakubovsky 2026 expression threshold
ZONATED = 0.5            # |portal bias| for a direction call
FLAT = 0.25              # |portal bias| below this counts as flat
z02 = __import__("importlib").import_module("02_ruminant_metabolism_zonation")


# ---------------------------------------------------------------- orthologs
def load_orthologs():
    o_all = pd.read_csv(REF / "orthologs/cow_human_mouse_orthologs.tsv", sep="\t")
    o_all.columns = ["ens", "cow_name", "human", "human_type", "mouse", "mouse_type"]
    o = o_all[o_all.human_type.eq("ortholog_one2one")]
    # one2many genes whose cattle symbol equals the human symbol (e.g. GLUL): accept by name
    same = o_all[o_all.cow_name.notna() & o_all.cow_name.eq(o_all.human) & ~o_all.human_type.eq("ortholog_one2one")]
    same_name = set(same.cow_name)
    x = pd.read_csv(REF / "orthologs/cow_ensembl_ncbi_xref.tsv", sep="\t")
    x.columns = ["ens", "cow_name", "ncbi_symbol", "ncbi_id"]
    ens2h = dict(zip(o.ens, o.human))
    name2h = dict(zip(o.cow_name.dropna(), o.human))
    ncbi2h = {s: ens2h[e] for s, e in zip(x.ncbi_symbol, x.ens) if isinstance(s, str) and e in ens2h}
    m = o[o.mouse_type.eq("ortholog_one2one")]
    mouse2h = dict(zip(m.mouse, m.human))

    def cow_to_human(label):
        if label in ens2h:
            return ens2h[label]
        if label in ncbi2h:
            return ncbi2h[label]
        if label in name2h:
            return name2h[label]
        return label if label in same_name else None

    def mouse_to_human(label):
        for part in str(label).split(";"):
            if part in mouse2h:
                return mouse2h[part]
            if part.upper() in same_name:
                return part.upper()
        return None
    return cow_to_human, mouse_to_human, set(o.human)


def portal_bias(layers):
    layers = np.asarray(layers, float)
    return np.log2((layers[:, -2:].mean(1) + EPS) / (layers[:, :2].mean(1) + EPS))


def collapse(df):
    """One row per human symbol: keep the most highly expressed source gene."""
    df = df.dropna(subset=["human"]).sort_values("max_expr", ascending=False)
    return df.drop_duplicates("human").set_index("human")


# ---------------------------------------------------------------- datasets
def load_layered(path, label_col, mean_prefix, to_human):
    d = pd.read_csv(path)
    L = [c for c in d.columns if c.startswith(mean_prefix)]
    L = L[: len(L) // 2] if len(L) > 9 else L          # Halpern lists means then s.e.m.
    X = d[L].to_numpy(float)
    out = pd.DataFrame({"human": d[label_col].map(to_human), "pb": portal_bias(X), "max_expr": X.max(1),
                        "layers": list(X)})
    return collapse(out)


def load_all(cow_to_human, mouse_to_human):
    ds = {}
    # this study
    r = pd.read_csv(OUT / "results/01_zonation_genomewide.csv")
    prof = r[[f"profile_L{k}" for k in range(1, 9)]].to_numpy()
    ours = pd.DataFrame({"human": r.gene.map(cow_to_human),
                         "pb": (prof[:, 6:].mean(1) - prof[:, :2].mean(1)) / np.log(2),
                         "max_expr": r.total_UMI_parenchyma.to_numpy(float), "robust": r.robust.to_numpy(),
                         "parenchymal": r.parenchymal_zonated.to_numpy(), "gene_cow": r.gene, "FDR": r.FDR})
    ds["Cow Stereo-seq (this study)"] = collapse(ours)

    def multi(prefix, to_human, label):
        per, mx = [], []
        for f in sorted((REF / "csv").glob(f"nonhuman_{prefix}*.csv")):
            d = load_layered(f, "Gene", "Mean lobule layer", to_human)
            d = d[d.max_expr >= MIN_MAX_EXPR]
            per.append(d.pb.rename(f.stem.replace("nonhuman_", "")))
            mx.append(d.max_expr.rename(f.stem))
        P = pd.concat(per, axis=1)
        MX = pd.concat(mx, axis=1)
        keep = P.notna().sum(1) >= max(1, int(np.ceil(P.shape[1] / 2)))
        agg = pd.DataFrame({"pb": P[keep].mean(1), "n_animals": P[keep].notna().sum(1),
                            "max_expr": MX.reindex(P[keep].index).mean(1),
                            "sign_consistency": (np.sign(P[keep]).eq(np.sign(P[keep].mean(1)), axis=0)
                                                 .sum(1) / P[keep].notna().sum(1))})
        ds[label] = agg
        return P

    animals = {
        "Cow Visium (n=2)": multi("COW", cow_to_human, "Cow Visium (n=2)"),
        "Pig Visium (n=5)": multi("PIG", str.upper, "Pig Visium (n=5)"),
        "Mouse Visium (n=6)": multi("MUS", mouse_to_human, "Mouse Visium (n=6)"),
    }
    boar = multi("BOAR", str.upper, "_boar")
    animals["Pig Visium (n=5)"] = pd.concat([animals["Pig Visium (n=5)"], boar], axis=1)
    P = animals["Pig Visium (n=5)"]
    keep = P.notna().sum(1) >= 3
    pig_mx = pd.concat([ds["Pig Visium (n=5)"].max_expr, ds["_boar"].max_expr], axis=1).mean(1)
    ds["Pig Visium (n=5)"] = pd.DataFrame({"pb": P[keep].mean(1), "n_animals": P[keep].notna().sum(1),
                                           "max_expr": pig_mx.reindex(P[keep].index)})
    ds.pop("_boar")

    h = pd.read_csv(REF / "csv/human_LHD5_zonation.csv")
    X = h[[f"mean_zone_{i}" for i in range(1, 9)]].to_numpy(float)
    hd = collapse(pd.DataFrame({"human": h.gene_name, "pb": portal_bias(X), "max_expr": X.max(1),
                                "qValue": h.qValue}))
    ds["Human Visium (n=5)"] = hd[hd.max_expr >= MIN_MAX_EXPR]

    hal = load_layered(REF / "csv/mouse_halpern2017_S3.csv", "Gene Symbol", "Layer", mouse_to_human)
    ds["Mouse scRNA (Halpern 2017)"] = hal[hal.max_expr >= MIN_MAX_EXPR]
    return ds, animals


# ---------------------------------------------------------------- pathways
def pathway_bias(cow_to_human, mouse_to_human):
    """Pooled pathway portal bias in every dataset (sum of member genes per layer)."""
    paths = dict(z02.PATHWAYS)
    sets = pd.read_csv(OUT / "results/02_pathway_set_zonation.csv").set_index("pathway")
    rows = {}
    for p in sets.index:
        prof = sets.loc[p, [f"pooled_profile_L{k}" for k in range(1, 9)]].to_numpy(float)
        rows.setdefault(p, {})["Cow Stereo-seq (this study)"] = (prof[6:].mean() - prof[:2].mean()) / np.log(2)
    human_sets = {p: {cow_to_human(g) or g for g in sets.loc[p, "genes"].split(";")} for p in sets.index}

    def add(label, files, label_col, prefix, to_human, nlay=None):
        for p, hs in human_sets.items():
            vals = []
            for f in files:
                d = pd.read_csv(f)
                L = [c for c in d.columns if c.startswith(prefix)]
                L = L[: len(L) // 2] if len(L) > 9 else L
                hum = d[label_col].map(to_human)
                X = d.loc[hum.isin(hs), L].to_numpy(float)
                if len(X):
                    vals.append(portal_bias(X.sum(0, keepdims=True))[0])
            rows[p][label] = np.mean(vals) if vals else np.nan

    csv = REF / "csv"
    add("Cow Visium (n=2)", sorted(csv.glob("nonhuman_COW*.csv")), "Gene", "Mean lobule layer", cow_to_human)
    add("Pig Visium (n=5)", sorted(csv.glob("nonhuman_PIG*.csv")) + sorted(csv.glob("nonhuman_BOAR*.csv")),
        "Gene", "Mean lobule layer", str.upper)
    add("Human Visium (n=5)", [csv / "human_LHD5_zonation.csv"], "gene_name", "mean_zone_", lambda s: s)
    add("Mouse Visium (n=6)", sorted(csv.glob("nonhuman_MUS*.csv")), "Gene", "Mean lobule layer", mouse_to_human)
    add("Mouse scRNA (Halpern 2017)", [csv / "mouse_halpern2017_S3.csv"], "Gene Symbol", "Layer", mouse_to_human)
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- main
TIERS = [5e-5, 2e-4, 1e-3]   # minimum max-layer expression in every Visium / scRNA reference compared
MAIN_TIER = 2e-4
OTHERS = ["Pig Visium (n=5)", "Human Visium (n=5)", "Mouse Visium (n=6)", "Mouse scRNA (Halpern 2017)"]


def tier_mask(ds, names, t):
    """Genes present in all `names`, with reference expression >= t (this study: all tested genes)."""
    idx = None
    for k in names:
        d = ds[k]
        ok = d.index if k.startswith("Cow Stereo") else d.index[d.max_expr >= t]
        idx = ok if idx is None else idx.intersection(ok)
    return idx


def main():
    cow_to_human, mouse_to_human, _ = load_orthologs()
    ds, animals = load_all(cow_to_human, mouse_to_human)
    names = list(ds)
    ours = ds["Cow Stereo-seq (this study)"]
    print({k: len(v) for k, v in ds.items()})
    print(f"this study: {len(ours)} of 5524 tested genes mapped to human orthologs; GLUL mapped: {'GLUL' in ours.index}")

    # ---- reference reproducibility (noise ceilings)
    ceil_rows = []
    for label, P in animals.items():
        for t in TIERS:
            rhos = []
            for i in range(P.shape[1]):
                for j in range(i + 1, P.shape[1]):
                    m = P.iloc[:, [i, j]].dropna()
                    m = m[ds[label].max_expr.reindex(m.index) >= t]
                    if len(m) > 20:
                        rhos.append(stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])[0])
            ceil_rows.append(dict(dataset=label, tier=t, n_pairs=len(rhos), median_animal_pair_rho=np.median(rhos)))
    ceil = pd.DataFrame(ceil_rows)
    ceil.to_csv(OUT / "results/03_reference_animal_reproducibility.csv", index=False)
    print(ceil.round(2).to_string(index=False))

    # ---- pairwise correlations at expression tiers
    pb = pd.DataFrame({k: v.pb for k, v in ds.items()})
    corr_rows, mats = [], {}
    for t in TIERS:
        R = pd.DataFrame(np.nan, index=names, columns=names)
        N = pd.DataFrame(0, index=names, columns=names)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                idx = tier_mask(ds, [a, b], t)
                if len(idx) > 20:
                    r = stats.spearmanr(pb.loc[idx, a], pb.loc[idx, b])[0]
                    R.loc[a, b] = R.loc[b, a] = r
                    N.loc[a, b] = N.loc[b, a] = len(idx)
                    corr_rows.append(dict(tier=t, a=a, b=b, n=len(idx), spearman=r))
        mats[t] = (R, N)
    corr_long = pd.DataFrame(corr_rows)
    corr_long.to_csv(OUT / "results/03_portal_bias_spearman_by_tier.csv", index=False)
    print(corr_long[corr_long.a.str.startswith("Cow Stereo")].round(2).to_string(index=False))

    # ---- agreement on this study's robust genes (well-expressed in the reference)
    rob = ours[ours.robust]
    agree_rows = []
    for k in names[1:]:
        ref = ds[k][ds[k].max_expr >= MAIN_TIER]
        m = rob.join(ref.pb.rename("ref"), how="inner")
        z = m[m.ref.abs() >= ZONATED]
        agree_rows.append(dict(dataset=k, robust_genes_shared=len(m),
                               spearman=stats.spearmanr(m.pb, m.ref)[0] if len(m) > 5 else np.nan,
                               sign_agreement_all=np.mean(np.sign(m.pb) == np.sign(m.ref)) if len(m) else np.nan,
                               ref_zonated=len(z),
                               sign_agreement_ref_zonated=np.mean(np.sign(z.pb) == np.sign(z.ref)) if len(z) else np.nan,
                               median_abs_ratio_this_over_ref=float(np.median(z.pb.abs() / z.ref.abs())) if len(z) else np.nan))
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(OUT / "results/03_robust_gene_agreement.csv", index=False)
    print(agree.round(3).to_string(index=False))

    # ---- gene classes (others = pig, human, mouse; need >= 2 of them well expressed)
    def ref_pb(k):
        d = ds[k]
        return d.pb.where(d.max_expr >= MAIN_TIER)
    g = pd.DataFrame({"cow_stereo": ours.pb, "cow_stereo_FDR": ours.FDR, "cow_stereo_robust": ours.robust,
                      "cow_gene": ours.gene_cow, "cow_visium": ref_pb("Cow Visium (n=2)"),
                      "pig": ref_pb("Pig Visium (n=5)"), "human": ref_pb("Human Visium (n=5)"),
                      "mouse_visium": ref_pb("Mouse Visium (n=6)"), "mouse_scrna": ref_pb("Mouse scRNA (Halpern 2017)")})
    g = g[g.cow_stereo.notna()]
    oth = g[["pig", "human", "mouse_visium", "mouse_scrna"]]
    # mouse counts once: average its two datasets
    oth = pd.DataFrame({"pig": oth.pig, "human": oth.human, "mouse": oth[["mouse_visium", "mouse_scrna"]].mean(1)})
    n_oth = oth.notna().sum(1)
    cow_dir = np.sign(g.cow_stereo)
    zon_same = (np.sign(oth).eq(cow_dir, axis=0) & (oth.abs() >= ZONATED))
    not_same = (np.sign(oth).ne(cow_dir, axis=0) | (oth.abs() < FLAT)) & oth.notna()
    oth_dir = np.sign(oth.mean(1))
    oth_strong = ((np.sign(oth).eq(oth_dir, axis=0) & (oth.abs() >= 1.0)) | oth.isna()).all(1) & (n_oth >= 2)
    g["class"] = ""
    g.loc[g.cow_stereo_robust & (n_oth >= 2) & (zon_same.sum(1) == n_oth), "class"] = "conserved"
    g.loc[g.cow_stereo_robust & (n_oth >= 2) & (not_same.sum(1) == n_oth), "class"] = "cow-divergent"
    g.loc[~g.cow_stereo_robust & (g.cow_stereo.abs() < FLAT) & (g.cow_stereo_FDR > 0.2) & oth_strong,
          "class"] = "flat in cow, strongly zonated elsewhere"
    # Independent cows count only when they show a gradient themselves (|bias| >= FLAT).
    cv = g.cow_visium
    g["cow_visium_supports"] = np.select(
        [cv.isna(), (np.sign(cv) == cow_dir) & (cv.abs() >= FLAT), (np.sign(cv) != cow_dir) & (cv.abs() >= FLAT)],
        ["n/a", "yes", "no"], default="uninformative")
    flat = g["class"] == "flat in cow, strongly zonated elsewhere"
    g.loc[flat, "cow_visium_supports"] = np.select([cv[flat].isna(), cv[flat].abs() < FLAT], ["n/a", "yes"], "no")
    g["others_mean"] = oth.mean(1)
    g["n_other_species"] = n_oth
    g.rename_axis("human_symbol").sort_values(["class", "cow_stereo"]).to_csv(OUT / "results/03_cross_species_gene_table.csv")
    for c in ["conserved", "cow-divergent", "flat in cow, strongly zonated elsewhere"]:
        sub = g[g["class"] == c]
        key = "others_mean" if c.startswith("flat") else "cow_stereo"
        print(f"{c}: n={len(sub)}  (cow Visium supports: {(sub.cow_visium_supports == 'yes').sum()}, "
              f"contradicts: {(sub.cow_visium_supports == 'no').sum()})")
        print("   PC:", sub[sub[key] < 0].sort_values(key).index.tolist()[:40])
        print("   PP:", sub[sub[key] > 0].sort_values(key, ascending=False).index.tolist()[:40])

    # ---- pathways: raw and scaled by each dataset's own pericentral CYP gradient
    pw = pathway_bias(cow_to_human, mouse_to_human)
    scale = pw.loc["Xenobiotic CYPs"].abs()
    pw_norm = pw / scale
    pw.to_csv(OUT / "results/03_pathway_portal_bias_across_species.csv")
    pw_norm.to_csv(OUT / "results/03_pathway_portal_bias_scaled_by_CYP.csv")
    print(pw.round(2).to_string()); print(pw_norm.round(2).to_string())

    markers = ["GLUL", "CYP2E1", "CYP1A2", "CYP7A1", "TBX3", "RHBG", "SULT1E1", "HAMP", "GLS2", "HAL", "ALDOB",
               "ASS1", "ASL", "ARG1", "CPS1", "OTC", "PCK1", "G6PC1", "FBP1", "PC", "PCCA", "PCCB", "MMUT",
               "HMGCS2", "CPT1A", "ACADM", "SCD", "APOE"]
    mk = pd.DataFrame({k: v.pb.where(k.startswith("Cow Stereo") | (v.max_expr >= MIN_MAX_EXPR)) for k, v in ds.items()}).reindex(markers)
    mk.to_csv(OUT / "results/03_marker_portal_bias_across_species.csv")
    print(mk.round(2).to_string())

    make_figure(mats[MAIN_TIER], ceil, corr_long, agree, g, pw, pw_norm, mk)
    cls = g["class"].value_counts()
    summary = pd.Series({
        "genes_this_study_mapped": len(ours),
        **{f"rho_vs_{b.split(' (')[0].replace(' ', '_')}_tier{t:g}": r.spearman
           for t in TIERS for _, r in corr_long[(corr_long.tier == t) & (corr_long.a == names[0])].iterrows()
           for b in [r.b]},
        **{f"cow_visium_animal_rho_tier{t:g}": ceil[(ceil.dataset == 'Cow Visium (n=2)') & (ceil.tier == t)]
           .median_animal_pair_rho.iloc[0] for t in TIERS},
        **{f"n_{k}": int(v) for k, v in cls.items() if k},
    })
    summary.to_csv(OUT / "results/03_cross_species_summary.csv", header=["value"])


def make_figure(mat, ceil, corr_long, agree, g, pw, pw_norm, mk):
    PC_COL, PP_COL = "#c0392b", "#2471a3"
    short = {"Cow Stereo-seq (this study)": "Cow Stereo-seq\n(this study)", "Cow Visium (n=2)": "Cow Visium\n(n=2)",
             "Pig Visium (n=5)": "Pig Visium\n(n=5)", "Human Visium (n=5)": "Human Visium\n(n=5)",
             "Mouse Visium (n=6)": "Mouse Visium\n(n=6)", "Mouse scRNA (Halpern 2017)": "Mouse scRNA\n(Halpern 2017)"}
    R, N = mat
    fig = plt.figure(figsize=(25, 17))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], wspace=0.38, hspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(R.to_numpy(float), cmap="viridis", vmin=0, vmax=0.8)
    lab = [short[c] for c in R.columns]
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if i != j and np.isfinite(R.iat[i, j]):
                ax.text(j, i, f"{R.iat[i, j]:.2f}\nn={N.iat[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if R.iat[i, j] < 0.45 else "black")
    plt.colorbar(im, ax=ax, fraction=0.045, label="Spearman ρ of portal bias")
    ax.set_title(f"A  Zonation similarity, genes with reference expression ≥ {MAIN_TIER:g}")

    ax = fig.add_subplot(gs[0, 1])
    cl = corr_long[corr_long.a == "Cow Stereo-seq (this study)"]
    for b, sub in cl.groupby("b", sort=False):
        ax.plot(np.log10(sub.tier), sub.spearman, "o-", label=short[b].replace("\n", " "), lw=2)
    cv = ceil[ceil.dataset == "Cow Visium (n=2)"]
    ax.plot(np.log10(cv.tier), cv.median_animal_pair_rho, "k--", lw=1.5, label="cow Visium C1 vs C2 (noise ceiling)")
    mv = ceil[ceil.dataset == "Mouse Visium (n=6)"]
    ax.plot(np.log10(mv.tier), mv.median_animal_pair_rho, ":", c="grey", lw=1.5, label="mouse Visium animal pairs")
    ax.set_xticks(np.log10(TIERS)); ax.set_xticklabels([f"≥{t:g}" for t in TIERS])
    ax.set_xlabel("minimum reference expression (max layer, UMI fraction)")
    ax.set_ylabel("Spearman ρ with this study")
    ax.axhline(0, c="k", lw=.5); ax.legend(fontsize=7.5)
    ax.set_title("B  Similarity to this study rises with expression;\ncow Visium reference is noisy (dashed)")

    ax = fig.add_subplot(gs[0, 2])
    ag = agree.set_index("dataset")
    x = np.arange(len(ag))
    ax.bar(x - 0.2, ag.sign_agreement_all * 100, 0.4, color="#95a5a6", label="robust genes, reference expr ≥ 2e-4")
    ax.bar(x + 0.2, ag.sign_agreement_ref_zonated * 100, 0.4, color="#2c3e50", label="… and reference |bias| ≥ 0.5")
    for xi, (_, row) in zip(x, ag.iterrows()):
        ax.text(xi - 0.2, row.sign_agreement_all * 100 + 1, f"{row.robust_genes_shared}", ha="center", fontsize=7)
        ax.text(xi + 0.2, row.sign_agreement_ref_zonated * 100 + 1, f"{row.ref_zonated}", ha="center", fontsize=7)
    ax.axhline(50, c="k", ls=":", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([short[k] for k in ag.index], fontsize=8)
    ax.set_ylim(0, 110); ax.set_ylabel("% same direction as this study (n genes above bars)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("C  Direction agreement for this study's robust zonated genes")

    cols = list(pw.columns)
    order = pw_norm["Human Visium (n=5)"].sort_values().index
    for k, (M_df, ttl, vmax) in enumerate([(pw, "D  Pathway portal bias (raw log2 PP/PC)", 1.5),
                                           (pw_norm, "E  Same, scaled by each dataset's |xenobiotic CYP| bias\n"
                                                     "(removes platform / resolution dynamic range)", 1.0)]):
        ax = fig.add_subplot(gs[1, k])
        M = M_df.loc[order, cols].to_numpy(float)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.5)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([short[c] for c in cols], fontsize=7.5, rotation=45, ha="right")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
        plt.colorbar(im, ax=ax, fraction=0.045)
        ax.set_title(ttl)

    ax = fig.add_subplot(gs[1, 2])
    M = mk[cols].to_numpy(float)
    im = ax.imshow(np.clip(M, -3, 3), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(mk))); ax.set_yticklabels(mk.index, fontsize=7.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([short[c] for c in cols], fontsize=7.5, rotation=45, ha="right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}" if np.isfinite(M[i, j]) else "–", ha="center", va="center", fontsize=6,
                    color="#999999" if not np.isfinite(M[i, j]) else ("white" if abs(M[i, j]) > 1.8 else "black"))
    plt.colorbar(im, ax=ax, fraction=0.045, label="portal bias (clipped ±3)")
    ax.set_title("F  Marker and metabolic genes ('–' = below expression threshold / unmapped)")

    fig.savefig(OUT / "figures/03_cross_species_zonation.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figures/03_cross_species_zonation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
