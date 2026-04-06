"""
evaluator.py — DSPM Evaluation Pipeline + Statistical Analysis.

DSPMEvaluator  : runs full DSPM pipeline on a dialogue, collects metrics
compute_statistics : computes mean ± std, 95% CI, Wilcoxon test
"""

import re
import json
import time
from collections import Counter
from typing import List, Dict, Tuple, Any

import numpy as np
from scipy import stats

from dspm.config import (
    GROQ_MODEL, TOKEN_BUDGET, CRITICAL_TYPES,
    MAX_RESPONSE_TOKENS, MAX_CONSISTENCY_TOKENS,
    RESPONSE_TEMPERATURE,
)
from dspm.structures import SemanticPatch, count_tokens
from dspm.engine import DSPMEngine
from dspm.extractor import ERROR_LOG, GroqClientPool

# ── Prompts ───────────────────────────────────────────────────────────────

CONSISTENCY_SYSTEM = """You are an expert evaluator for long-context LLM \
memory compression systems.

Given two answers to the same question:
  ANSWER_A: generated from the FULL conversation history
  ANSWER_B: generated from COMPRESSED DSPM memory patches

Rate the SEMANTIC CONSISTENCY of B relative to A on a scale of 1-5:
  5 = Identical technical content, all constraints and decisions preserved
  4 = Minor omissions, core content intact
  3 = Moderate omissions or slight inaccuracies
  2 = Significant information loss
  1 = Fundamentally different or contradictory

CRITICAL: respond ONLY with a JSON object like {"score": 4, "reason": "brief explanation"}.
No markdown, no extra text.
"""

PROBE_QUESTION = (
    "Based on all constraints and decisions discussed, "
    "summarise the final architecture with all key technical specifications "
    "and constraints that must be respected."
)


# ── LLM Response Helper ───────────────────────────────────────────────────

def generate_llm_response(
    system_prompt : str,
    user_content  : str,
    pool          : GroqClientPool,
    max_tokens    : int = MAX_RESPONSE_TOKENS,
) -> str:
    """Single-shot LLM call with error handling and pool failover."""
    try:
        resp = pool.chat(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature = RESPONSE_TEMPERATURE,
            max_tokens  = max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        ERROR_LOG.append({
            "dialogue"   : "llm_response",
            "turn"       : -1,
            "error_type" : "llm_error",
            "message"    : str(e),
        })
        return "[LLM ERROR — no response]"


def compute_consistency_score(
    full_answer : str,
    dspm_answer : str,
    pool        : GroqClientPool,
) -> Tuple[int, str]:
    """
    Score semantic consistency of DSPM answer vs full-history answer.

    Returns
    -------
    score  : int 1-5
    reason : brief explanation string
    """
    try:
        raw = generate_llm_response(
            CONSISTENCY_SYSTEM,
            f"ANSWER_A:\n{full_answer}\n\nANSWER_B:\n{dspm_answer}",
            pool,
            max_tokens = MAX_CONSISTENCY_TOKENS,
        )
        jm = re.search(r'\{.*\}', raw, re.DOTALL)
        if jm:
            d = json.loads(jm.group())
            return int(d.get("score", 3)), d.get("reason", "")
    except Exception:
        pass
    return 3, "parsing error"


# ── DSPM Evaluator ────────────────────────────────────────────────────────

class DSPMEvaluator:
    """
    Runs the full DSPM pipeline on a dialogue and collects:
      - per-turn token counts (raw vs DSPM)
      - technique-level diagnostics
      - Token Reduction Rate (TRR)
      - Critical Retention Rate (CRR)
      - downstream answer consistency score (1-5)
      - per-turn error statistics for robustness analysis

    Parameters
    ----------
    pool : GroqClientPool instance for all LLM calls
    """

    def __init__(self, pool: GroqClientPool):
        self.pool   = pool
        self.engine = DSPMEngine(TOKEN_BUDGET)

    def run_dialogue(
        self,
        dialogue_name : str,
        turns         : List[Tuple[str, str]],
        verbose       : bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on one dialogue.

        Parameters
        ----------
        dialogue_name : identifier string for this dialogue
        turns         : list of (role, text) tuples
        verbose       : if True, print per-turn stats

        Returns
        -------
        dict with all metrics for this dialogue
        """
        from dspm.extractor import extract_patches

        self.engine.reset_ema()

        all_patches      : List[SemanticPatch] = []
        raw_tokens_turn  : List[int] = []
        dspm_tokens_turn : List[int] = []
        full_history     : List[str] = []
        diag_per_turn    : List[Dict] = []
        errors_per_turn  : List[int] = []

        for turn_idx, (role, text) in enumerate(turns):
            full_history.append(f"{role.upper()}: {text}")
            raw_tok = count_tokens("\n".join(full_history))
            raw_tokens_turn.append(raw_tok)

            errors_before = len(ERROR_LOG)
            recent_ctx    = "\n".join(
                p.to_prompt_str() for p in all_patches[-8:]
            )
            new_patches = extract_patches(
                text, turn_idx, self.pool, recent_ctx, dialogue_name
            )
            errors_per_turn.append(len(ERROR_LOG) - errors_before)

            all_patches.extend(new_patches)
            last_query = (
                text if role == "user"
                else (turns[turn_idx - 1][1] if turn_idx > 0 else text)
            )
            selected, diag = self.engine.compress(
                all_patches, last_query, turn_idx
            )

            dspm_tok = sum(p.token_cost for p in selected)
            dspm_tokens_turn.append(dspm_tok)
            diag_per_turn.append(diag)

            if verbose:
                print(
                    f"  Turn {turn_idx:02d} | raw={raw_tok:4d} | "
                    f"dspm={dspm_tok:3d} | patches={len(new_patches)} | "
                    f"selected={diag.get('selected_count', 0)} | "
                    f"errors={errors_per_turn[-1]}"
                )

        # ── Final metrics ─────────────────────────────────────────────────
        raw_final  = raw_tokens_turn[-1]
        dspm_final = dspm_tokens_turn[-1]
        trr        = (1 - dspm_final / raw_final) * 100 if raw_final > 0 else 0.0

        last_selected, _ = self.engine.compress(
            all_patches, turns[-1][1], len(turns) - 1
        )
        critical_total    = sum(
            1 for p in all_patches if p.patch_type in CRITICAL_TYPES
        )
        critical_selected = sum(
            1 for p in last_selected if p.patch_type in CRITICAL_TYPES
        )
        crr = (
            (critical_selected / critical_total * 100)
            if critical_total > 0 else 100.0
        )

        # ── Consistency scoring ───────────────────────────────────────────
        full_answer = generate_llm_response(
            "You are a technical assistant. Answer based ONLY on the "
            "provided conversation history.",
            f"CONVERSATION:\n{chr(10).join(full_history[-20:])}"
            f"\n\nQUESTION: {PROBE_QUESTION}",
            self.pool,
            max_tokens = MAX_RESPONSE_TOKENS,
        )
        dspm_context = self.engine.build_context(last_selected)
        dspm_answer  = generate_llm_response(
            "You are a technical assistant. Answer based ONLY on the "
            "provided compressed memory patches.",
            f"MEMORY PATCHES:\n{dspm_context}\n\nQUESTION: {PROBE_QUESTION}",
            self.pool,
            max_tokens = MAX_RESPONSE_TOKENS,
        )
        consistency, reason = compute_consistency_score(
            full_answer, dspm_answer, self.pool
        )

        # ── Error summary ─────────────────────────────────────────────────
        dialogue_errors = [
            e for e in ERROR_LOG if e["dialogue"] == dialogue_name
        ]
        error_counts = Counter(e["error_type"] for e in dialogue_errors)

        return {
            "dialogue_name"     : dialogue_name,
            "turn_count"        : len(turns),
            "raw_tokens_final"  : raw_final,
            "dspm_tokens_final" : dspm_final,
            "trr_pct"           : round(trr, 2),
            "crr_pct"           : round(crr, 2),
            "consistency_score" : consistency,
            "consistency_reason": reason,
            "total_patches"     : len(all_patches),
            "raw_tokens_turn"   : raw_tokens_turn,
            "dspm_tokens_turn"  : dspm_tokens_turn,
            "diag_per_turn"     : diag_per_turn,
            "errors_per_turn"   : errors_per_turn,
            "error_counts"      : dict(error_counts),
            "probe_question"    : PROBE_QUESTION,
        }


# ── Statistical Analysis ──────────────────────────────────────────────────

def compute_statistics(results: List[Dict]) -> Dict:
    """
    Compute aggregate statistics across all dialogue results.

    Includes: mean ± std, 95% CI, median, Wilcoxon signed-rank test.

    Parameters
    ----------
    results : list of dicts returned by DSPMEvaluator.run_dialogue()

    Returns
    -------
    dict with all statistical metrics
    """
    trr_vals  = [r["trr_pct"]           for r in results]
    crr_vals  = [r["crr_pct"]           for r in results]
    cons_vals = [r["consistency_score"] for r in results]

    def ci95(vals):
        n = len(vals)
        if n < 2:
            return (vals[0], vals[0])
        se = stats.sem(vals)
        h  = se * stats.t.ppf(0.975, df=n - 1)
        m  = np.mean(vals)
        return (round(m - h, 2), round(m + h, 2))

    try:
        _, p_cons = stats.wilcoxon([c - 3 for c in cons_vals])
    except Exception:
        p_cons = 1.0

    return {
        "n_dialogues" : len(results),
        "trr_mean"    : round(np.mean(trr_vals), 2),
        "trr_std"     : round(np.std(trr_vals, ddof=1), 2),
        "trr_ci95"    : ci95(trr_vals),
        "trr_median"  : round(np.median(trr_vals), 2),
        "crr_mean"    : round(np.mean(crr_vals), 2),
        "crr_min"     : round(min(crr_vals), 2),
        "cons_mean"   : round(np.mean(cons_vals), 2),
        "cons_std"    : round(np.std(cons_vals, ddof=1), 2),
        "cons_ci95"   : ci95(cons_vals),
        "wilcoxon_p"  : round(p_cons, 4),
        "trr_vals"    : trr_vals,
        "crr_vals"    : crr_vals,
        "cons_vals"   : cons_vals,
    }
