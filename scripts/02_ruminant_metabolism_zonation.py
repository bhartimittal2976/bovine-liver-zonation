"""Analysis 2: zonation of ruminant-relevant metabolic pathways along the lobular axis (eta).

Cattle absorb little glucose and make most of it in the liver from rumen propionate. This asks
where along the central -> portal axis the propionate -> glucose route, the urea cycle,
ammonia scavenging, beta-oxidation / ketogenesis, lipid export, lipogenesis and bile acid synthesis
sit, and whether each matches the rodent pattern.

Uses the lobular coordinate and the genome-wide fit from 01 (run 01 first).
Set-level statistics:
  * pooled: sum of the set's counts, same block-adjusted quasi-Poisson model as 01
  * gene-mean: mean per-gene log2FC (PP vs PC), with an empirical p from 10,000
    expression-matched random gene sets (reported; underpowered for 3-6 gene sets)
Call: pooled FDR < 0.05 AND >= 2 member genes significant (FDR < 0.05) in that direction with
none significant in the other; "mixed" if members disagree; otherwise "weak" / "not zonated".
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lobule_utils import BIN_UM, OUT, bh, load_bins, pseudobulk, quasi_poisson_trend_test

z01 = importlib.import_module("01_lobular_axis_zonation")
N_LAYERS, BLOCK_UM = z01.N_LAYERS, z01.BLOCK_UM

# Rodent expectation from mouse single-cell / spatial zonation atlases
# (Halpern 2017 Nature; Ben-Moshe 2019 Nat Metab; Droin 2021 Nat Metab). "PP"/"PC"/"mixed".
PATHWAYS = {
    "SCFA activation (acyl-CoA synthetases)": ("mixed", ["ACSS1", "ACSS2", "ACSS3", "ACSM1", "ACSM3", "ACSM5", "ACSM2B"]),
    "Propionyl-CoA -> succinyl-CoA": ("unknown", ["PCCA", "PCCB", "MCEE", "MMUT", "MMAA", "MMAB"]),
    "Succinyl-CoA ligase / TCA entry": ("PP", ["SUCLG1", "SUCLG2", "SUCLA2"]),
    "Gluconeogenesis core": ("PP", ["PC", "PCK1", "PCK2", "FBP1", "G6PC1", "SLC37A4"]),
    "Glucogenic amino acid catabolism": ("PP", ["GLS2", "SDS", "TAT", "HAL", "AGXT", "PAH", "HPD", "HGD", "FAH", "GPT", "GPT2"]),
    "Lactate / glycerol entry": ("PP", ["LDHB", "SLC16A1", "GK", "GPD1"]),
    "Glycolysis": ("PC", ["GCK", "PKLR", "PFKL", "SLC2A2"]),
    "Urea cycle": ("PP", ["NAGS", "CPS1", "OTC", "ASS1", "ASL", "ARG1"]),
    "Glutamine synthesis / NH3 scavenging": ("PC", ["GLUL", "RHBG", "OAT", "SLC1A2"]),
    "Mitochondrial beta-oxidation": ("PP", ["CPT1A", "CPT2", "ACADM", "ACADL", "ACADVL", "HADHA", "HADHB", "PPARA"]),
    "Ketogenesis": ("PP", ["HMGCS2", "HMGCL", "BDH1", "ACAT1"]),
    "VLDL assembly / export": ("mixed", ["APOB", "MTTP", "APOC3", "APOE", "SAR1B"]),
    "De novo lipogenesis": ("PC", ["ACLY", "ACACA", "FASN", "SCD", "ELOVL6", "SREBF1"]),
    "Bile acid synthesis": ("PC", ["CYP7A1", "CYP8B1", "CYP27A1", "SLC27A5"]),
    "Xenobiotic CYPs": ("PC", ["CYP2E1", "CYP1A2", "CYP2A13", "CYP3A76", "CYP2C89"]),
    "Wnt / pericentral identity": ("PC", ["AXIN2", "TBX3", "LGR5", "SP5", "ZNRF3", "NOTUM", "RNF43", "NKD1", "LECT2"]),
    "Iron / hepcidin": ("PP", ["HAMP", "TF", "HPX"]),
}
OXPHOS_PREFIX = ("NDUFA", "NDUFB", "NDUFS", "NDUFV", "COX4", "COX5", "COX6", "COX7", "UQCR", "ATP5")
N_PERM = 10_000
RNG = np.random.default_rng(42)


def main():
    res = pd.read_csv(OUT / "results/01_zonation_genomewide.csv").set_index("gene")
    coord = pd.read_csv(OUT / "data/01_bin50_lobular_coordinate.csv.gz", index_col=0)
    a = load_bins()
    assert (a.obs_names == coord.index).all()

    paths = dict(PATHWAYS)
    oxphos = [g for g in res.index if g.startswith(OXPHOS_PREFIX)]
    paths["OXPHOS (complex I/III/IV/V)"] = ("PP", oxphos)

    # ---------- gene level ----------
    rows = []
    for pname, (rodent, genes) in paths.items():
        for g in genes:
            r = dict(pathway=pname, rodent_expected=rodent, gene=g, in_matrix=g in a.var_names, tested=g in res.index)
            if r["tested"]:
                rr = res.loc[g]
                r.update(log2FC_PP_vs_PC=rr.log2FC_PP_vs_PC, FDR=rr.FDR, log2FC_interior=rr.log2FC_interior,
                         FDR_interior=rr.FDR_interior, COM_eta=rr.COM_eta, robust=rr.robust,
                         log2FC_top_half=rr.log2FC_top_half, log2FC_bottom_half=rr.log2FC_bottom_half,
                         pct_bins_detected=rr.pct_bins_detected,
                         **{c: rr[c] for c in res.columns if c.startswith("log2FC_CV")})
                r["call"] = ("PP" if rr.log2FC_PP_vs_PC > 0 else "PC") if rr.FDR < 0.05 else "n.s."
            else:
                r["call"] = "not_tested"
            rows.append(r)
    genes_df = pd.DataFrame(rows)
    genes_df.to_csv(OUT / "results/02_pathway_gene_zonation.csv", index=False)

    # ---------- set level: pooled counts through the 01 model ----------
    valid = coord["valid"].to_numpy()
    layer = np.where(valid, coord["layer"].to_numpy(), 0)
    bx = (a.obs["ix"].to_numpy() * BIN_UM // BLOCK_UM).astype(int)
    by = (a.obs["iy"].to_numpy() * BIN_UM // BLOCK_UM).astype(int)
    counts, meta = pseudobulk(a, valid, layer, bx * 100 + by)
    mids = (np.arange(N_LAYERS) + 0.5) / N_LAYERS
    X_null, X_full, predict, k0 = z01.design(meta, mids)
    offset = np.log(meta["lib"].to_numpy(float))

    tested_paths = {p: [g for g in gs if g in res.index] for p, (_, gs) in paths.items()}
    tested_paths = {p: gs for p, gs in tested_paths.items() if len(gs) >= 2}
    Y = np.column_stack([counts[:, [a.var_names.get_loc(g) for g in gs]].sum(1) for gs in tested_paths.values()])
    beta, F, p, phi = quasi_poisson_trend_test(Y, X_null, X_full, offset)
    prof = predict(mids) @ beta[:, k0:].T
    prof = prof - prof.mean(0)

    # ---------- set level: gene-mean with expression-matched null ----------
    expr_bin = pd.qcut(np.log10(res.total_UMI_parenchyma), 10, labels=False)
    pool = {b: res.index[expr_bin == b].to_numpy() for b in range(10)}
    lfc = res.log2FC_PP_vs_PC

    set_rows = []
    for i, (pname, gs) in enumerate(tested_paths.items()):
        obs = lfc[gs].mean()
        bins = expr_bin[gs].to_numpy()
        null = np.zeros(N_PERM)
        for b in np.unique(bins):
            k = (bins == b).sum()
            draws = RNG.choice(pool[b], size=(N_PERM, k), replace=True)
            null += lfc.reindex(draws.ravel()).to_numpy().reshape(N_PERM, k).sum(1)
        null /= len(gs)
        p_emp = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (N_PERM + 1)
        w = np.exp(prof[:, i])
        set_rows.append(dict(
            pathway=pname, rodent_expected=paths[pname][0], n_genes_tested=len(gs),
            n_genes_listed=len(paths[pname][1]),
            pooled_log2FC_PP_vs_PC=(prof[-1, i] - prof[0, i]) / np.log(2), pooled_p=p[i],
            pooled_COM_eta=(mids * w).sum() / w.sum(),
            genemean_log2FC=obs, genemean_null_mean=null.mean(), genemean_p_emp=p_emp,
            n_PP_FDR05=int(((lfc[gs] > 0) & (res.FDR[gs] < 0.05)).sum()),
            n_PC_FDR05=int(((lfc[gs] < 0) & (res.FDR[gs] < 0.05)).sum()),
            genes=";".join(gs),
            **{f"pooled_profile_L{k + 1}": prof[k, i] for k in range(N_LAYERS)}))
    sets = pd.DataFrame(set_rows)
    sets["pooled_FDR"] = bh(sets.pooled_p)
    sets["genemean_FDR"] = bh(sets.genemean_p_emp)
    def call(r):
        if r.n_PP_FDR05 > 0 and r.n_PC_FDR05 > 0:
            return "mixed"
        d = "PP" if r.pooled_log2FC_PP_vs_PC > 0 else "PC"
        n_dir = r.n_PP_FDR05 if d == "PP" else r.n_PC_FDR05
        if r.pooled_FDR < 0.05 and n_dir >= 2:
            return d
        if r.pooled_FDR < 0.05 or (r.n_PP_FDR05 + r.n_PC_FDR05) > 0:
            return f"weak {d}" if n_dir > 0 else "not zonated"
        return "not zonated"
    sets["bovine_call"] = sets.apply(call, axis=1)
    sets["matches_rodent"] = np.where(sets.rodent_expected.isin(["PP", "PC"]),
                                      (sets.bovine_call == sets.rodent_expected).astype(str), "n/a")
    sets = sets.sort_values("pooled_COM_eta")
    sets.to_csv(OUT / "results/02_pathway_set_zonation.csv", index=False)
    show = ["pathway", "rodent_expected", "bovine_call", "n_genes_tested", "pooled_log2FC_PP_vs_PC",
            "pooled_FDR", "genemean_log2FC", "genemean_FDR", "n_PP_FDR05", "n_PC_FDR05", "pooled_COM_eta"]
    print(sets[show].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print()
    print(genes_df[genes_df.tested][["pathway", "gene", "log2FC_PP_vs_PC", "FDR", "log2FC_interior",
                                     "FDR_interior", "call"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("not tested:", genes_df[~genes_df.tested].groupby("pathway").gene.apply(list).to_dict())

    make_figure(sets, genes_df, res, coord, a)


def make_figure(sets, genes_df, res, coord, a):
    PC_COL, PP_COL, NS = "#c0392b", "#2471a3", "#9a9a9a"
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1.15, 1], wspace=0.45)

    # A: set-level forest plot
    ax = fig.add_subplot(gs[0, 0])
    s = sets.sort_values("pooled_log2FC_PP_vs_PC")
    y = np.arange(len(s))
    col = [PP_COL if c == "PP" else PC_COL if c == "PC" else "#9ec3e0" if c == "weak PP" else
           "#e8a49b" if c == "weak PC" else NS for c in s.bovine_call]
    ax.barh(y, s.pooled_log2FC_PP_vs_PC, color=col, alpha=0.85)
    ax.scatter(s.genemean_log2FC, y, marker="D", s=22, c="k", zorder=3, label="gene-mean log2FC")
    ax.scatter(s.genemean_null_mean, y, marker="|", s=80, c="#555555", zorder=3, label="expression-matched null")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}  [rodent {r}]" for p, r in zip(s.pathway, s.rodent_expected)], fontsize=8)
    ax.axvline(0, c="k", lw=0.6)
    for yi, (_, row) in zip(y, s.iterrows()):
        mark = {"True": "✓", "False": "✗", "n/a": ""}[row.matches_rodent]
        ax.annotate(mark, (1.01, yi), xycoords=("axes fraction", "data"), fontsize=10, va="center",
                    color="green" if mark == "✓" else "red")
    ax.set_xlabel("log2 FC portal / central (bars: pooled counts, block-adjusted)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("A  Pathway zonation in bovine liver\n(dark = called, light = weak, grey = none/mixed; ✓/✗ vs rodent)")

    # B: pooled pathway profiles along eta
    ax = fig.add_subplot(gs[0, 1])
    pcols = [c for c in sets.columns if c.startswith("pooled_profile_L")]
    mids = (np.arange(N_LAYERS) + 0.5) / N_LAYERS
    focus = ["Propionyl-CoA -> succinyl-CoA", "Gluconeogenesis core", "Urea cycle",
             "Glutamine synthesis / NH3 scavenging", "Mitochondrial beta-oxidation", "Ketogenesis",
             "De novo lipogenesis", "Bile acid synthesis", "OXPHOS (complex I/III/IV/V)"]
    cmap = plt.get_cmap("tab10")
    for i, pname in enumerate([f for f in focus if f in set(sets.pathway)]):
        row = sets.set_index("pathway").loc[pname]
        ax.plot(mids, np.array(row[pcols], float) / np.log(2), lw=2, color=cmap(i),
                ls="-" if row.bovine_call in ("PP", "PC") else ":", label=f"{pname} [{row.bovine_call}]")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("η (0 = central vein, 1 = portal tract)")
    ax.set_ylabel("log2 relative expression (block-adjusted)")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title("B  Pooled pathway profiles\n(dotted = weak / mixed / not zonated)")

    # C: gene-level heatmap within metabolic route
    ax = fig.add_subplot(gs[0, 2])
    route = ["SCFA activation (acyl-CoA synthetases)", "Propionyl-CoA -> succinyl-CoA",
             "Succinyl-CoA ligase / TCA entry", "Gluconeogenesis core", "Urea cycle",
             "Glutamine synthesis / NH3 scavenging", "Ketogenesis"]
    gd = genes_df[genes_df.tested & genes_df.pathway.isin(route)].copy()
    gd["order"] = gd.pathway.map({p: i for i, p in enumerate(route)})
    gd = gd.sort_values(["order", "log2FC_PP_vs_PC"])
    prof_cols = [c for c in res.columns if c.startswith("profile_L")]
    M = res.loc[gd.gene, prof_cols].to_numpy() / np.log(2)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_yticks(range(len(gd)))
    ax.set_yticklabels([f"{g}{' *' if f < 0.05 else ''}" for g, f in zip(gd.gene, gd.FDR)], fontsize=7)
    ax.set_xticks(range(N_LAYERS)); ax.set_xticklabels([f"{m:.2f}" for m in mids], fontsize=7)
    ax.set_xlabel("η layer (central → portal)")
    prev = None
    for i, p in enumerate(gd.pathway):
        if p != prev:
            ax.axhline(i - 0.5, c="k", lw=0.8)
            ax.text(N_LAYERS - 0.4, i - 0.3, p, fontsize=6.5, va="top", ha="left", clip_on=False)
            prev = p
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.25, label="log2 relative expression")
    ax.set_title("C  Propionate → glucose and\nN-handling genes (* FDR<0.05)")

    # D: every tested pathway gene, bovine gradient grouped by rodent expectation
    ax = fig.add_subplot(gs[1, :])
    gd = genes_df[genes_df.tested & ~genes_df.pathway.str.startswith("OXPHOS")].copy()
    order = sets[~sets.pathway.str.startswith("OXPHOS")].sort_values(
        ["rodent_expected", "pooled_log2FC_PP_vs_PC"]).pathway.tolist()
    xpos = {p: i for i, p in enumerate(order)}
    rng = np.random.default_rng(0)
    for _, r in gd.iterrows():
        xi = xpos[r.pathway] + rng.uniform(-0.18, 0.18)
        c = PP_COL if r.call == "PP" else PC_COL if r.call == "PC" else NS
        ax.scatter(xi, r.log2FC_PP_vs_PC, s=28 if r.call in ("PP", "PC") else 14, c=c, zorder=3)
        if r.call in ("PP", "PC") or r.gene in ("CPS1", "PC", "PCCA", "PCCB", "MMUT", "HMGCS2", "CPT1A", "G6PC1"):
            ax.annotate(r.gene, (xi, r.log2FC_PP_vs_PC), fontsize=6.5, xytext=(3, 0), textcoords="offset points")
    rodent = sets.set_index("pathway").rodent_expected
    for p, i in xpos.items():
        e = rodent[p]
        ax.axvspan(i - 0.45, i + 0.45, color=PP_COL if e == "PP" else PC_COL if e == "PC" else "#dddddd",
                   alpha=0.07, lw=0)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{p}\n[rodent {rodent[p]}]" for p in order], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("gene log2 FC portal / central")
    ax.set_title("D  Every tested pathway gene (coloured = FDR<0.05; background = rodent expectation). "
                 "Pericentral programmes keep rodent-like gradients; propionate entry, β-oxidation, "
                 "ketogenesis are flat and CPS1 is not periportal", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "figures/02_ruminant_metabolism_zonation.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "figures/02_ruminant_metabolism_zonation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
