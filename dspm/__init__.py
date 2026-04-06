"""
DSPM: Dynamic Semantic Patch Memory
=====================================
Token-efficient memory architecture for long-context reasoning in LLMs.

Author      : Dhruv Dubey
Affiliation : Bansal Institute of Engineering & Technology, AKTU, Lucknow
Venue       : IEEE SMC 2026 (Under Review)
Engine      : Groq API (llama-3.3-70b-versatile)

Key Result  : TRR = 82.4% ± 4.21% across 7 heterogeneous technical dialogues
              Consistency Score = 3.57/5.0 relative to full-history baseline
"""

from dspm.config import (
    GROQ_MODEL, TOKEN_BUDGET, PATCH_TYPES, CRITICAL_TYPES,
    BASE_BUDGET_FRACTIONS, SEED,
)
from dspm.structures import SemanticPatch, count_tokens
from dspm.engine import DSPMEngine
from dspm.extractor import GroqClientPool, extract_patches
from dspm.evaluator import DSPMEvaluator, compute_statistics
from dspm.baselines import compute_baselines
from dspm.ablation import AblationEngine, run_ablation

__version__ = "1.0.0"
__author__  = "Dhruv Dubey"

__all__ = [
    "GROQ_MODEL", "TOKEN_BUDGET", "PATCH_TYPES", "CRITICAL_TYPES",
    "SemanticPatch", "count_tokens",
    "DSPMEngine", "GroqClientPool", "extract_patches",
    "DSPMEvaluator", "compute_statistics",
    "compute_baselines", "AblationEngine", "run_ablation",
]
