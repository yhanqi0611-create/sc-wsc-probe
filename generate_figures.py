"""
Generate paper figures from existing evaluation CSVs.
Outputs: paper/figures/fig_parity_comet.pdf, paper/figures/fig_comet_by_label.pdf

Run: python generate_figures.py
Requires: pandas, numpy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES_CSV = os.path.join(ROOT, "translation_results_200.csv")
SEG_CSV = os.path.join(ROOT, "translation_segment_scores.csv")
OUT_DIR = os.path.join(ROOT, "paper", "figures")
GLM, DS = "GLM-4-Plus", "DeepSeek-V4-Flash"
B = 2000
SEED = 42


def bootstrap_comet_ci(scores: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(scores)
    means = [scores[rng.integers(0, n, n)].mean() for _ in range(B)]
    return float(scores.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fig_parity_comet(seg: pd.DataFrame) -> None:
    glm = seg[f"{GLM}_COMET"].to_numpy()
    ds = seg[f"{DS}_COMET"].to_numpy()
    g_mean, g_lo, g_hi = bootstrap_comet_ci(glm)
    d_mean, d_lo, d_hi = bootstrap_comet_ci(ds)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    x = [0, 1]
    means = [g_mean, d_mean]
    yerr = [
        [g_mean - g_lo, d_mean - d_lo],
        [g_hi - g_mean, d_hi - d_mean],
    ]
    colors = ["#3b82f6", "#16a34a"]
    ax.bar(x, means, width=0.55, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["GLM-4-Plus", "DeepSeek-V4-Flash"], fontsize=9)
    ax.set_ylabel("System COMET ($\\times 100$)", fontsize=9)
    ax.set_ylim(68, 73)
    ax.set_title("Corpus-level COMET with 95% bootstrap CI", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.text(0.5, 71.6, "Overlapping CIs (p=0.836)", ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_parity_comet.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    print(f"  GLM: {g_mean:.2f} [{g_lo:.2f}, {g_hi:.2f}]")
    print(f"  DS:  {d_mean:.2f} [{d_lo:.2f}, {d_hi:.2f}]")


def fig_comet_by_label(res: pd.DataFrame, seg: pd.DataFrame) -> None:
    m = res.merge(seg[["sichuanese", f"{GLM}_COMET", f"{DS}_COMET"]], on="sichuanese", how="left")
    lab = m["preference"].astype(str).str.strip()
    m["avg_comet"] = (m[f"{GLM}_COMET"] + m[f"{DS}_COMET"]) / 2
    tg = m.loc[lab == "tie-good", "avg_comet"]
    tb = m.loc[lab == "tie-bad", "avg_comet"]

    tg_mu, tb_mu = tg.mean(), tb.mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bins = np.linspace(50, 95, 16)
    ax.hist(tg, bins=bins, alpha=0.65, color="#16a34a",
            label=f"tie-good (n=28, mean={tg_mu:.1f})", edgecolor="white")
    ax.hist(tb, bins=bins, alpha=0.65, color="#dc2626",
            label=f"tie-bad (n=56, mean={tb_mu:.1f})", edgecolor="white")
    ax.axvline(tg_mu, color="#16a34a", linestyle="--", linewidth=1.2)
    ax.axvline(tb_mu, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Mean sentence COMET ($\\times 100$)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("COMET distribution: cultural success vs. failure", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_comet_by_label.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    res = pd.read_csv(RES_CSV, encoding="utf-8-sig")
    seg = pd.read_csv(SEG_CSV, encoding="utf-8-sig")
    fig_parity_comet(seg)
    fig_comet_by_label(res, seg)


if __name__ == "__main__":
    main()
