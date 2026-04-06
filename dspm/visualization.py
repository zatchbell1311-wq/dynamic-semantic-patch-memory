"""
visualization.py — IEEE publication-quality figures for DSPM.

Figures generated:
  fig_main_results.pdf    — 2x2 panel: token usage, TRR, per-turn growth, consistency
  fig_ablation.pdf        — per-technique ablation bar chart
  fig_baselines.pdf       — method comparison (Raw/Truncation/Compression/DSPM)
  fig_patch_types.pdf     — budget allocation pie chart
  fig_error_robustness.pdf — error & robustness stats per dialogue
"""

from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dspm.config import IEEE_STYLE, COLORS


# ── Figure 1: Main Results (2x2 Panel) ───────────────────────────────────

def plot_main_results(
    results    : List[Dict],
    stats_dict : Dict,
    save_path  : str = "fig_main_results.pdf",
):
    """
    2x2 panel figure:
      (a) Token usage: raw vs DSPM
      (b) Token Reduction Rate per dialogue
      (c) Per-turn token growth (first 3 dialogues)
      (d) Consistency scores per dialogue
    """
    plt.rcParams.update(IEEE_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5))
    names = [r["dialogue_name"].replace("_", " ") for r in results]
    x     = np.arange(len(results))
    w     = 0.35

    # ── Panel A: Token Usage ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(
        x - w / 2,
        [r["raw_tokens_final"]  for r in results],
        w, label="Raw",  color=COLORS["raw"],  alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        [r["dspm_tokens_final"] for r in results],
        w, label="DSPM", color=COLORS["dspm"], alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Tokens stored")
    ax.set_title("(a) Token usage: raw vs DSPM")
    ax.legend()

    # ── Panel B: TRR per dialogue ─────────────────────────────────────────
    ax   = axes[0, 1]
    bars = ax.bar(
        x,
        [r["trr_pct"] for r in results],
        color=COLORS["dspm"], alpha=0.85, width=0.5,
    )
    ax.axhline(
        55, color=COLORS["target55"], lw=1.2, ls="--", label="55% target"
    )
    ax.axhline(
        60, color=COLORS["target60"], lw=1.2, ls=":",  label="60% target"
    )
    for bar, r in zip(bars, results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{r['trr_pct']:.1f}%",
            ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Token reduction (%)")
    ax.set_title(
        f"(b) TRR: mean={stats_dict['trr_mean']}% "
        f"± {stats_dict['trr_std']}%"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # ── Panel C: Per-turn token growth ────────────────────────────────────
    ax      = axes[1, 0]
    markers = ["o", "s", "^"]
    for i, r in enumerate(results[:3]):
        turns = range(1, len(r["raw_tokens_turn"]) + 1)
        ax.plot(
            turns, r["raw_tokens_turn"],
            color=COLORS["raw"],  ls="--", marker=markers[i],
            ms=4, alpha=0.6, label=f"Raw {names[i][:10]}",
        )
        ax.plot(
            turns, r["dspm_tokens_turn"],
            color=COLORS["dspm"], ls="-",  marker=markers[i],
            ms=4, alpha=0.9, label=f"DSPM {names[i][:10]}",
        )
    ax.set_xlabel("Dialogue turn")
    ax.set_ylabel("Cumulative tokens")
    ax.set_title("(c) Per-turn token growth")
    ax.legend(fontsize=6, ncol=2)

    # ── Panel D: Consistency scores ───────────────────────────────────────
    ax     = axes[1, 1]
    scores = [r["consistency_score"] for r in results]
    ax.bar(x, scores, color=COLORS["dspm"], alpha=0.85, width=0.5)
    ax.axhline(
        stats_dict["cons_mean"],
        color="#e66101", lw=1.5,
        label=f"Mean = {stats_dict['cons_mean']:.2f}",
    )
    ax.axhline(
        3, color=COLORS["target55"], lw=1, ls="--", label="Neutral (3)"
    )
    for xi, s in zip(x, scores):
        ax.text(xi, s + 0.05, str(s), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Consistency score (1–5)")
    ax.set_title(f"(d) Consistency: mean={stats_dict['cons_mean']:.2f}")
    ax.set_ylim(0, 5.5)
    ax.legend()

    fig.suptitle(
        "DSPM — Dynamic Semantic Patch Memory: Main Results",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


# ── Figure 2: Ablation Study ──────────────────────────────────────────────

def plot_ablation(
    ablation_df : pd.DataFrame,
    save_path   : str = "fig_ablation.pdf",
):
    """
    Horizontal bar chart showing TRR for each ablation variant.
    Full DSPM baseline shown as dashed vertical line.
    """
    plt.rcParams.update(IEEE_STYLE)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    df     = ablation_df.sort_values("trr_pct")
    colors = [
        COLORS["dspm"] if "Full" in v else COLORS["ablation"]
        for v in df["variant"]
    ]
    bars = ax.barh(
        df["variant"], df["trr_pct"],
        color=colors, alpha=0.85, height=0.55,
    )

    full_trr = ablation_df.loc[
        ablation_df["variant"] == "Full DSPM", "trr_pct"
    ].values[0]
    ax.axvline(
        full_trr, color=COLORS["dspm"], lw=1.5,
        ls="--", label="Full DSPM baseline",
    )

    for bar, trr in zip(bars, df["trr_pct"]):
        ax.text(
            trr + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{trr:.1f}%",
            va="center", fontsize=8,
        )

    ax.set_xlabel("Token Reduction Rate (%)")
    ax.set_title("Ablation study — per-technique contribution")
    ax.legend()
    ax.set_xlim(0, 95)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


# ── Figure 3: Baseline Comparison ────────────────────────────────────────

def plot_baseline_comparison(
    results   : List[Dict],
    baselines : List[Dict],
    save_path : str = "fig_baselines.pdf",
):
    """
    Side-by-side bar charts comparing all methods:
      (a) Average token usage
      (b) Average Token Reduction Rate
    """
    plt.rcParams.update(IEEE_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    methods       = ["Raw", "Truncation", "Compression", "DSPM"]
    method_colors = [
        COLORS["raw"], COLORS["truncation"],
        COLORS["compression"], COLORS["dspm"],
    ]

    # Average tokens per method
    avg_tokens = [
        np.mean([r["raw_tokens_final"]              for r in results]),
        np.mean([b["truncation"]["tokens"]           for b in baselines]),
        np.mean([b["compression"]["tokens"]          for b in baselines]),
        np.mean([r["dspm_tokens_final"]              for r in results]),
    ]
    ax1.bar(methods, avg_tokens, color=method_colors, alpha=0.85, width=0.5)
    for i, (m, t) in enumerate(zip(methods, avg_tokens)):
        ax1.text(i, t + 5, f"{t:.0f}", ha="center", fontsize=8)
    ax1.set_ylabel("Avg tokens stored")
    ax1.set_title("(a) Token usage by method")

    # Average TRR per method
    avg_trr = [
        0.0,
        np.mean([b["truncation"]["trr"]             for b in baselines]),
        np.mean([b["compression"]["trr"]            for b in baselines]),
        np.mean([r["trr_pct"]                       for r in results]),
    ]
    ax2.bar(methods, avg_trr, color=method_colors, alpha=0.85, width=0.5)
    for i, (m, t) in enumerate(zip(methods, avg_trr)):
        ax2.text(i, t + 0.5, f"{t:.1f}%", ha="center", fontsize=8)
    ax2.axhline(
        55, color=COLORS["target55"], lw=1, ls="--", label="55% target"
    )
    ax2.axhline(
        60, color=COLORS["target60"], lw=1, ls=":",  label="60% target"
    )
    ax2.set_ylabel("Token Reduction Rate (%)")
    ax2.set_title("(b) TRR by method")
    ax2.set_ylim(0, 100)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


# ── Figure 4: Patch Type Budget Allocation ────────────────────────────────

def plot_patch_types(
    results   : List[Dict],
    save_path : str = "fig_patch_types.pdf",
):
    """
    Pie chart showing average token budget allocation across patch types.
    """
    plt.rcParams.update(IEEE_STYLE)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    from dspm.config import BASE_BUDGET_FRACTIONS
    labels = list(BASE_BUDGET_FRACTIONS.keys())
    sizes  = list(BASE_BUDGET_FRACTIONS.values())
    colors = [
        "#2166ac", "#d73027", "#4dac26",
        "#fdae61", "#a6d96a", "#999999",
    ]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels     = labels,
        colors     = colors,
        autopct    = "%1.1f%%",
        startangle = 140,
        pctdistance= 0.82,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Token budget allocation by patch type")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


# ── Figure 5: Error & Robustness ──────────────────────────────────────────

def plot_error_robustness(
    results   : List[Dict],
    save_path : str = "fig_error_robustness.pdf",
):
    """
    Bar chart showing error counts and robustness metrics per dialogue.
    """
    plt.rcParams.update(IEEE_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    names        = [r["dialogue_name"].replace("_", " ") for r in results]
    x            = np.arange(len(results))
    total_errors = [sum(r["errors_per_turn"]) for r in results]

    # Panel A: total errors per dialogue
    ax1.bar(x, total_errors, color=COLORS["error"], alpha=0.85, width=0.5)
    for xi, e in zip(x, total_errors):
        ax1.text(xi, e + 0.05, str(e), ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("Total extraction errors")
    ax1.set_title("(a) Errors per dialogue")

    # Panel B: CRR per dialogue
    crr_vals = [r["crr_pct"] for r in results]
    ax2.bar(x, crr_vals, color=COLORS["dspm"], alpha=0.85, width=0.5)
    ax2.axhline(
        np.mean(crr_vals),
        color="#e66101", lw=1.5,
        label=f"Mean CRR = {np.mean(crr_vals):.1f}%",
    )
    for xi, c in zip(x, crr_vals):
        ax2.text(xi, c + 0.3, f"{c:.1f}%", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("Critical Retention Rate (%)")
    ax2.set_title("(b) CRR per dialogue")
    ax2.set_ylim(0, 110)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")
