# DSPM: Dynamic Semantic Patch Memory

> **Token-efficient memory architecture for long-context reasoning in Large Language Models**
> 
> Under Review — IEEE SMC 2026

---

## Key Results

| Metric | Value |
|--------|-------|
| Token Reduction Rate (TRR) | **82.4% ± 4.21%** |
| Critical Retention Rate (CRR) | **100%** |
| Consistency Score | **3.57 / 5.0** |
| Dialogues Evaluated | 7 heterogeneous technical scenarios |
| Design Targets Surpassed | 55% ✅ and 60% ✅ |

---

## What is DSPM?

DSPM compresses long conversational context into structured **semantic patches** — typed, scored, and budget-constrained memory units — using **7 deterministic, LLM-free compression techniques**:

| Technique | Name | What it does |
|-----------|------|--------------|
| T1 | Semantic Fingerprinting | Hash-based deduplication |
| T2 | SlotFusion | Canonical slot collapsing |
| T3 | Δ-Diff Encoding | Patch-level change encoding |
| T4 | Causal Chain Pruning | Intermediate node elimination |
| T5 | Utility Scoring | Weighted patch ranking |
| T6 | Shadow Scoring | Marginal-value filtering |
| T7 | Adaptive Budget Redistribution | Query-driven token reallocation |

---

## Repository Structure
