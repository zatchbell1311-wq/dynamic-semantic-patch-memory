"""
config.py — All hyperparameters and constants for DSPM.
Single source of truth. Change values here, everything updates.
"""

import random
import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Model Configuration ───────────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOKENIZER_NAME  = "cl100k_base"

# ── DSPM Core Hyperparameters ─────────────────────────────────────────────
TOKEN_BUDGET      = 300    # max tokens in compressed memory
ALPHA_EMA         = 0.5    # EMA smoothing for query-type tracking
SHADOW_THRESHOLD  = 0.05   # min utility fraction to survive shadow filter
RECENCY_LAMBDA    = 0.1    # decay rate for recency scoring
DELTA_MIN_SAVING  = 2      # min token saving required to apply delta encoding

# ── Utility Scoring Weights (must sum to 1.0) ─────────────────────────────
W_ALIGN   = 0.40   # semantic alignment with query
W_DEP     = 0.25   # dependency centrality
W_RECENCY = 0.20   # recency decay
W_COST    = 0.15   # token cost penalty

# ── Patch Type Budget Fractions ───────────────────────────────────────────
BASE_BUDGET_FRACTIONS = {
    "constraint" : 0.30,
    "decision"   : 0.25,
    "code"       : 0.20,
    "equation"   : 0.10,
    "entity"     : 0.10,
    "structure"  : 0.05,
}

PATCH_TYPES    = list(BASE_BUDGET_FRACTIONS.keys())
CRITICAL_TYPES = {"constraint", "decision"}  # always retained

# ── Groq API Settings ─────────────────────────────────────────────────────
MAX_EXTRACTION_TOKENS  = 1200
MAX_RESPONSE_TOKENS    = 500
MAX_CONSISTENCY_TOKENS = 200
EXTRACTION_TEMPERATURE = 0.0
RESPONSE_TEMPERATURE   = 0.1
API_SLEEP_BETWEEN_CALLS = 2   # seconds between Groq calls

# ── Figure / Plot Settings ────────────────────────────────────────────────
IEEE_STYLE = {
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.labelsize"   : 10,
    "axes.titlesize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 9,
    "figure.dpi"       : 300,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
}

COLORS = {
    "dspm"        : "#2166ac",
    "raw"         : "#d73027",
    "truncation"  : "#fdae61",
    "compression" : "#a6d96a",
    "ablation"    : "#4dac26",
    "target55"    : "#999999",
    "target60"    : "#555555",
    "error"       : "#e8494a",
}
