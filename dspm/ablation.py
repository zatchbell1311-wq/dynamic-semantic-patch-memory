"""
ablation.py — Ablation study engine for DSPM.

AblationEngine : DSPMEngine subclass with per-technique on/off flags
run_ablation   : runs all 7 ablation variants on a dialogue
"""

from copy import deepcopy
from typing import List, Dict, Tuple

import pandas as pd

from dspm.config import (
    TOKEN_BUDGET, PATCH_TYPES, CRITICAL_TYPES,
    BASE_BUDGET_FRACTIONS,
)
from dspm.structures import SemanticPatch, count_tokens
from dspm.engine import DSPMEngine

# ── Ablation Variants ─────────────────────────────────────────────────────
# Each variant disables exactly one technique to measure its contribution.

ABLATION_VARIANTS = {
    "Full DSPM"            : {"t1":True,  "t2":True,  "t3":True,  "t4":True,  "t6":True,  "t7":True },
    "w/o Fingerprint (T1)" : {"t1":False, "t2":True,  "t3":True,  "t4":True,  "t6":True,  "t7":True },
    "w/o SlotFusion (T2)"  : {"t1":True,  "t2":False, "t3":True,  "t4":True,  "t6":True,  "t7":True },
    "w/o Δ-Diff (T3)"      : {"t1":True,  "t2":True,  "t3":False, "t4":True,  "t6":True,  "t7":True },
    "w/o Causal (T4)"      : {"t1":True,  "t2":True,  "t3":True,  "t4":False, "t6":True,  "t7":True },
    "w/o Shadow (T6)"      : {"t1":True,  "t2":True,  "t3":True,  "t4":True,  "t6":False, "t7":True },
    "w/o Adaptive (T7)"    : {"t1":True,  "t2":True,  "t3":True,  "t4":True,  "t6":True,  "t7":False},
}


class AblationEngine(DSPMEngine):
    """
    DSPMEngine subclass that selectively disables techniques.

    Parameters
    ----------
    budget : token budget (same as DSPMEngine)
    flags  : dict of technique flags e.g. {"t1": True, "t2": False, ...}
    """

    def __init__(self, budget: int, flags: Dict[str, bool]):
        super().__init__(budget)
        self.flags = flags

    def compress(self, patches, query, turn_index):
        if not patches:
            return [], {}

        working = deepcopy(patches)
        diag    = {}

        self._update_ema(query)

        # Selectively apply techniques based on flags
        if self.flags.get("t1", True):
            working, diag["t1_removed"] = self._fingerprint_dedup(working)

        if self.flags.get("t2", True):
            working, diag["t2_removed"] = self._slot_fusion(working)

        if self.flags.get("t3", True):
            working, diag["t3_deltas"]  = self._delta_encode(working)

        if self.flags.get("t4", True):
            working, diag["t4_removed"] = self._causal_prune(working)

        # T5 utility scoring always runs (needed for T6)
        working = self._score_utility(working, query, turn_index)

        # T7: adaptive budget vs fixed budget
        if self.flags.get("t7", True):
            type_budgets = self._adaptive_budget()
        else:
            type_budgets = {
                t: int(self.budget * BASE_BUDGET_FRACTIONS[t])
                for t in PATCH_TYPES
            }

        # T6: shadow scoring vs greedy selection
        if self.flags.get("t6", True):
            selected, diag["t6_removed"] = self._shadow_select(
                working, type_budgets
            )
        else:
            # Greedy fallback — no shadow filtering
            guaranteed = [
                p for p in working if p.patch_type in CRITICAL_TYPES
            ]
            candidates = sorted(
                [p for p in working if p.patch_type not in CRITICAL_TYPES],
                key=lambda p: p.utility / max(p.token_cost, 1),
                reverse=True,
            )
            selected = list(guaranteed)
            used = {
                t: sum(p.token_cost for p in guaranteed if p.patch_type == t)
                for t in PATCH_TYPES
            }
            for p in candidates:
                budget_left = (
                    type_budgets.get(p.patch_type, 0) - used[p.patch_type]
                )
                if p.token_cost <= budget_left:
                    selected.append(p)
                    used[p.patch_type] += p.token_cost
            diag["t6_removed"] = 0

        diag["total_tokens"]   = sum(p.token_cost for p in selected)
        diag["selected_count"] = len(selected)
        return selected, diag


# ── Ablation Runner ───────────────────────────────────────────────────────

def run_ablation(
    dialogue_name : str,
    turns         : List[Tuple[str, str]],
    all_patches   : List[SemanticPatch],
) -> pd.DataFrame:
    """
    Run all 7 ablation variants on a dialogue's extracted patches.

    Parameters
    ----------
    dialogue_name : name of the dialogue being ablated
    turns         : list of (role, text) tuples
    all_patches   : all SemanticPatch objects extracted from the dialogue

    Returns
    -------
    DataFrame with columns [variant, dspm_tokens, raw_tokens, trr_pct]
    """
    raw_final = count_tokens(
        "\n".join(f"{r}: {t}" for r, t in turns)
    )
    records = []

    for variant_name, flags in ABLATION_VARIANTS.items():
        engine       = AblationEngine(TOKEN_BUDGET, flags)
        patches_copy = deepcopy(all_patches)
        selected, _  = engine.compress(
            patches_copy, turns[-1][1], len(turns) - 1
        )
        dspm_tok = sum(p.token_cost for p in selected)
        trr      = (
            (1 - dspm_tok / raw_final) * 100
            if raw_final > 0 else 0.0
        )
        records.append({
            "variant"     : variant_name,
            "dspm_tokens" : dspm_tok,
            "raw_tokens"  : raw_final,
            "trr_pct"     : round(trr, 2),
        })

    df = pd.DataFrame(records)

    print(f"\nAblation results for: {dialogue_name}")
    print(f"  {'Variant':<25} {'TRR':>8}")
    print(f"  {'-'*35}")
    for _, row in df.iterrows():
        print(f"  {row['variant']:<25} {row['trr_pct']:>7.1f}%")

    return df
