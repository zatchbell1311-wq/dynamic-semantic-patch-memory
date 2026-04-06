"""
engine.py — DSPM Compression Engine (7 Techniques).

T1  Semantic Fingerprinting   — hash-based deduplication
T2  SlotFusion                — canonical slot collapsing
T3  Δ-Diff Encoding           — patch-level change encoding
T4  Causal Chain Pruning      — intermediate node elimination
T5  Utility Scoring           — weighted patch ranking
T6  Shadow Scoring            — marginal-value filtering
T7  Adaptive Budget Redistrib — query-driven token reallocation
"""

import math
from copy import deepcopy
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dspm.config import (
    TOKEN_BUDGET, PATCH_TYPES, CRITICAL_TYPES,
    BASE_BUDGET_FRACTIONS, ALPHA_EMA, SHADOW_THRESHOLD,
    RECENCY_LAMBDA, DELTA_MIN_SAVING,
    W_ALIGN, W_DEP, W_RECENCY, W_COST,
    EMBED_MODEL,
)
from dspm.structures import SemanticPatch, count_tokens

# Load embedder once at module level
embedder = SentenceTransformer(EMBED_MODEL)


class DSPMEngine:
    """
    Dynamic Semantic Patch Memory compression engine.

    Applies seven deterministic, LLM-free compression operators
    in sequence to a list of SemanticPatch objects, returning a
    token-budget-constrained compressed context.
    """

    def __init__(self, budget: int = TOKEN_BUDGET):
        self.budget     = budget
        self.ema_query  = {t: 0.0 for t in PATCH_TYPES}
        self._stats     = defaultdict(int)

    def compress(
        self,
        patches   : List[SemanticPatch],
        query     : str,
        turn_index: int,
    ) -> Tuple[List[SemanticPatch], Dict[str, Any]]:
        """
        Run full 7-technique compression pipeline.

        Parameters
        ----------
        patches    : all accumulated SemanticPatch objects so far
        query      : current user query (drives EMA + utility scoring)
        turn_index : current dialogue turn number

        Returns
        -------
        selected   : compressed list of patches within token budget
        diag       : diagnostic dict with per-technique stats
        """
        if not patches:
            return [], {}

        working = deepcopy(patches)
        diag    = {}

        # T7 — update EMA before budget allocation
        self._update_ema(query)

        # T1 — Semantic Fingerprinting (deduplication)
        working, diag["t1_removed"] = self._fingerprint_dedup(working)

        # T2 — SlotFusion (canonical slot collapsing)
        working, diag["t2_removed"] = self._slot_fusion(working)

        # T3 — Δ-Diff Encoding
        working, diag["t3_deltas"]  = self._delta_encode(working)

        # T4 — Causal Chain Pruning
        working, diag["t4_removed"] = self._causal_prune(working)

        # T5 — Utility Scoring
        working = self._score_utility(working, query, turn_index)

        # T7 — Adaptive Budget Redistribution
        type_budgets = self._adaptive_budget()

        # T6 — Shadow Scoring + final selection
        selected, diag["t6_removed"] = self._shadow_select(working, type_budgets)

        diag["selected_count"]  = len(selected)
        diag["total_tokens"]    = sum(p.token_cost for p in selected)
        diag["type_budgets"]    = type_budgets
        diag["techniques_used"] = ["T1","T2","T3","T4","T5","T6","T7"]

        return selected, diag

    # ── T1: Semantic Fingerprinting ───────────────────────────────────────

    def _fingerprint_dedup(self, patches):
        seen: Dict[str, SemanticPatch] = {}
        for p in patches:
            fp = p.fingerprint
            if fp not in seen or p.turn_index > seen[fp].turn_index:
                seen[fp] = p
        removed = len(patches) - len(seen)
        self._stats["t1"] += removed
        return list(seen.values()), removed

    # ── T2: SlotFusion ────────────────────────────────────────────────────

    def _slot_fusion(self, patches):
        slots: Dict[str, List[SemanticPatch]] = defaultdict(list)
        for p in patches:
            slots[p.slot_key].append(p)
        fused = []
        for slot_patches in slots.values():
            winner = max(slot_patches, key=lambda p: (p.utility, p.turn_index))
            fused.append(winner)
        removed = len(patches) - len(fused)
        self._stats["t2"] += removed
        return fused, removed

    # ── T3: Δ-Diff Encoding ───────────────────────────────────────────────

    def _delta_encode(self, patches):
        slot_base: Dict[str, SemanticPatch] = {}
        delta_count = 0
        for p in sorted(patches, key=lambda x: x.turn_index):
            sk = p.slot_key
            if sk not in slot_base:
                slot_base[sk] = p
            else:
                base = slot_base[sk]
                delta_payload = self._compute_delta(p.payload, base.payload)
                if count_tokens(delta_payload) <= p.token_cost - DELTA_MIN_SAVING:
                    p.payload    = delta_payload
                    p.token_cost = count_tokens(delta_payload)
                    p.is_delta   = True
                    p.delta_base = base.patch_id
                    delta_count += 1
                slot_base[sk] = p
        return patches, delta_count

    @staticmethod
    def _compute_delta(new_text: str, base_text: str) -> str:
        new_words  = set(new_text.lower().split())
        base_words = set(base_text.lower().split())
        added      = new_words - base_words
        removed    = base_words - new_words
        parts = []
        if added:   parts.append("+[" + " ".join(sorted(added))   + "]")
        if removed: parts.append("-[" + " ".join(sorted(removed)) + "]")
        return " ".join(parts) if parts else new_text

    # ── T4: Causal Chain Pruning ──────────────────────────────────────────

    def _causal_prune(self, patches):
        id_map   = {p.patch_id: p for p in patches}
        children : Dict[str, List[str]] = defaultdict(list)
        for p in patches:
            for dep in p.dependencies:
                if dep in id_map:
                    children[dep].append(p.patch_id)

        depth: Dict[str, int] = {}
        queue = [p.patch_id for p in patches if not p.dependencies]
        for pid in queue: depth[pid] = 0
        visited = set(queue)
        while queue:
            nxt = []
            for pid in queue:
                for child in children.get(pid, []):
                    if child not in visited:
                        depth[child] = depth[pid] + 1
                        visited.add(child)
                        nxt.append(child)
            queue = nxt

        pruned  = []
        removed = 0
        for p in patches:
            d = depth.get(p.patch_id, 0)
            is_intermediate = (
                d > 0
                and bool(children.get(p.patch_id))
                and p.patch_type not in CRITICAL_TYPES
            )
            if is_intermediate:
                removed += 1
                self._stats["t4"] += 1
            else:
                p.causal_depth = d
                pruned.append(p)
        return pruned, removed

    # ── T5: Utility Scoring ───────────────────────────────────────────────

    def _score_utility(self, patches, query, turn_index):
        if not patches:
            return patches

        texts      = [query] + [p.payload for p in patches]
        embeddings = embedder.encode(texts, show_progress_bar=False)
        q_emb, p_embs = embeddings[0], embeddings[1:]

        dep_counts = Counter(dep for p in patches for dep in p.dependencies)
        max_dep    = max(dep_counts.values(), default=1)
        max_cost   = max(p.token_cost for p in patches) or 1

        type_boost = {
            "constraint": 0.20, "decision": 0.18, "code": 0.12,
            "equation"  : 0.10, "entity"  : 0.06, "structure": 0.04,
        }

        for i, p in enumerate(patches):
            cos_sim        = float(cosine_similarity([q_emb], [p_embs[i]])[0][0])
            align          = cos_sim + type_boost.get(p.patch_type, 0.0)
            dep_centrality = dep_counts.get(p.patch_id, 0) / max_dep
            recency        = math.exp(-RECENCY_LAMBDA * (turn_index - p.turn_index))
            cost_norm      = p.token_cost / max_cost
            p.utility = (
                W_ALIGN   * align +
                W_DEP     * dep_centrality +
                W_RECENCY * recency -
                W_COST    * cost_norm
            )
        return patches

    # ── T7: Adaptive Budget Redistribution ───────────────────────────────

    def _update_ema(self, query: str):
        query_lower = query.lower()
        for ptype in PATCH_TYPES:
            hit = 1.0 if ptype in query_lower else 0.0
            self.ema_query[ptype] = (
                ALPHA_EMA * hit + (1 - ALPHA_EMA) * self.ema_query[ptype]
            )

    def _adaptive_budget(self) -> Dict[str, int]:
        raw = {
            t: BASE_BUDGET_FRACTIONS[t] * (1 + ALPHA_EMA * self.ema_query[t])
            for t in PATCH_TYPES
        }
        total_weight = sum(raw.values())
        return {
            t: max(10, int(self.budget * raw[t] / total_weight))
            for t in PATCH_TYPES
        }

    # ── T6: Shadow Scoring & Selection ───────────────────────────────────

    def _shadow_select(self, patches, type_budgets):
        guaranteed  = [p for p in patches if p.patch_type in CRITICAL_TYPES]
        candidates  = [p for p in patches if p.patch_type not in CRITICAL_TYPES]

        selected    = list(guaranteed)
        used_tokens = {t: 0 for t in PATCH_TYPES}
        for p in guaranteed:
            used_tokens[p.patch_type] += p.token_cost

        candidates.sort(
            key=lambda p: p.utility / max(p.token_cost, 1), reverse=True
        )
        for p in candidates:
            budget_left = (
                type_budgets.get(p.patch_type, 0) - used_tokens[p.patch_type]
            )
            if p.token_cost <= budget_left:
                selected.append(p)
                used_tokens[p.patch_type] += p.token_cost

        if not selected:
            return selected, 0

        total_utility  = sum(p.utility for p in selected) or 1e-9
        pre_shadow_len = len(selected)
        selected = [
            p for p in selected
            if p.patch_type in CRITICAL_TYPES
            or (p.utility / total_utility) >= SHADOW_THRESHOLD
        ]
        shadow_removed = pre_shadow_len - len(selected)
        self._stats["t6"] += shadow_removed
        return selected, shadow_removed

    # ── Utilities ─────────────────────────────────────────────────────────

    def build_context(self, patches: List[SemanticPatch]) -> str:
        """Convert selected patches into a single prompt string."""
        return "\n".join(p.to_prompt_str() for p in patches)

    def reset_ema(self):
        """Reset EMA state between dialogues."""
        self.ema_query = {t: 0.0 for t in PATCH_TYPES}
