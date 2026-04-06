"""
run_pipeline.py — Single entry point for DSPM evaluation.

Usage:
    python run_pipeline.py

Set your Groq API key as environment variable before running:
    export GROQ_API_KEY="your_key_here"

Or for multiple keys (optional, for rate limit failover):
    export GROQ_API_KEY_2="your_second_key"
    export GROQ_API_KEY_3="your_third_key"
"""

import os
import json
import pandas as pd
from dspm.extractor import GroqClientPool, ERROR_LOG
from dspm.evaluator import DSPMEvaluator, compute_statistics
from dspm.baselines import compute_baselines
from dspm.ablation import run_ablation
from dspm.visualization import (
    plot_main_results,
    plot_ablation,
    plot_baseline_comparison,
    plot_patch_types,
    plot_error_robustness,
)
from dspm.dialogues import get_all_dialogues


def run_full_evaluation():

    # ── Step 1: Load API keys ─────────────────────────────────────────────
    keys = [
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
    ]
    keys = [k for k in keys if k]
    if not keys:
        raise RuntimeError(
            "No GROQ_API_KEY found.\n"
            "Run: export GROQ_API_KEY='your_key_here'"
        )

    pool = GroqClientPool(keys)
    print(f"Groq pool initialised with {len(keys)} key(s).\n")

    # ── Step 2: Load dialogues ────────────────────────────────────────────
    dialogues = get_all_dialogues()
    print(f"Loaded {len(dialogues)} dialogues.\n")

    # ── Step 3: Run DSPM evaluation on all dialogues ──────────────────────
    evaluator = DSPMEvaluator(pool)
    results   = []
    baselines = []

    for name, turns in dialogues.items():
        print(f"{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        result = evaluator.run_dialogue(name, turns, verbose=True)
        results.append(result)

        print(f"\n  TRR  : {result['trr_pct']}%")
        print(f"  CRR  : {result['crr_pct']}%")
        print(f"  Score: {result['consistency_score']}/5")
        print(f"  Reason: {result['consistency_reason']}\n")

        bl = compute_baselines(turns, pool)
        baselines.append(bl)

    # ── Step 4: Statistical summary ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    stats = compute_statistics(results)
    print(f"  Dialogues     : {stats['n_dialogues']}")
    print(f"  TRR mean ± std: {stats['trr_mean']}% ± {stats['trr_std']}%")
    print(f"  TRR 95% CI    : {stats['trr_ci95']}")
    print(f"  TRR median    : {stats['trr_median']}%")
    print(f"  CRR mean      : {stats['crr_mean']}%")
    print(f"  CRR min       : {stats['crr_min']}%")
    print(f"  Consistency   : {stats['cons_mean']} ± {stats['cons_std']}")
    print(f"  Wilcoxon p    : {stats['wilcoxon_p']}")

    # ── Step 5: Ablation study (on first dialogue) ────────────────────────
    print(f"\n{'='*60}")
    print("ABLATION STUDY")
    print(f"{'='*60}")
    first_name  = list(dialogues.keys())[0]
    first_turns = dialogues[first_name]
    ablation_evaluator = DSPMEvaluator(pool)
    ablation_result    = ablation_evaluator.run_dialogue(
        first_name, first_turns, verbose=False
    )
    # Collect all patches for ablation
    from dspm.extractor import extract_patches
    from dspm.engine import DSPMEngine
    engine      = DSPMEngine()
    all_patches = []
    for idx, (role, text) in enumerate(first_turns):
        patches = extract_patches(text, idx, pool, "", first_name)
        all_patches.extend(patches)

    ablation_df = run_ablation(first_name, first_turns, all_patches)

    # ── Step 6: Save results to CSV ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    rows = []
    for r in results:
        rows.append({
            "dialogue"          : r["dialogue_name"],
            "turns"             : r["turn_count"],
            "raw_tokens"        : r["raw_tokens_final"],
            "dspm_tokens"       : r["dspm_tokens_final"],
            "trr_pct"           : r["trr_pct"],
            "crr_pct"           : r["crr_pct"],
            "consistency_score" : r["consistency_score"],
            "total_patches"     : r["total_patches"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("dspm_results.csv", index=False)
    print("  Saved → dspm_results.csv")

    if ERROR_LOG:
        with open("error_log.json", "w") as f:
            json.dump(ERROR_LOG, f, indent=2)
        print(f"  Saved → error_log.json ({len(ERROR_LOG)} errors logged)")

    # ── Step 7: Generate all figures ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATING FIGURES")
    print(f"{'='*60}")
    plot_main_results(results, stats)
    plot_ablation(ablation_df)
    plot_baseline_comparison(results, baselines)
    plot_patch_types(results)
    plot_error_robustness(results)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  TRR  : {stats['trr_mean']}% ± {stats['trr_std']}%")
    print(f"  CRR  : {stats['crr_mean']}%")
    print(f"  Score: {stats['cons_mean']}/5.0")
    print(f"  API errors: {len(ERROR_LOG)}")

    return results, stats, ablation_df


if __name__ == "__main__":
    run_full_evaluation()
