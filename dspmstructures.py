"""
structures.py — Core data structures for DSPM.
Defines SemanticPatch: the atomic unit of compressed conversational memory.
"""

import hashlib
import tiktoken
import spacy

from dataclasses import dataclass, field
from typing import List

from dspm.config import TOKENIZER_NAME

# ── Tokenizer & NLP (loaded once) ────────────────────────────────────────
enc = tiktoken.get_encoding(TOKENIZER_NAME)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


# ── Helper Functions ──────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoder."""
    return len(enc.encode(text))


def _fingerprint(text: str) -> str:
    """
    MD5 fingerprint of order-normalised text.
    Used by T1 (Semantic Fingerprinting) for deduplication.
    """
    normalised = " ".join(sorted(text.lower().split()))
    return hashlib.md5(normalised.encode()).hexdigest()[:16]


def _topic_keyword(text: str) -> str:
    """
    Extract a canonical topic keyword from text.
    Prefers named entities, falls back to first noun, then first word.
    Used to build slot_key for T2 (SlotFusion).
    """
    doc = nlp(text[:200])
    for ent in doc.ents:
        return ent.text.lower().replace(" ", "_")
    nouns = [
        t.lemma_.lower()
        for t in doc
        if t.pos_ in {"NOUN", "PROPN"} and len(t.text) > 2
    ]
    return nouns[0] if nouns else text.split()[0].lower()


# ── SemanticPatch Dataclass ───────────────────────────────────────────────

@dataclass
class SemanticPatch:
    """
    Atomic unit of compressed conversational memory.

    Each patch represents one typed semantic unit extracted
    from a single dialogue turn. Patches are the currency
    of the DSPM compression pipeline.

    Attributes
    ----------
    patch_id     : unique identifier e.g. "p3_0"
    turn_index   : which dialogue turn this came from
    patch_type   : one of [constraint, decision, code,
                           equation, entity, structure]
    payload      : concise 8-20 word semantic summary
    dependencies : list of patch_ids this patch depends on
    utility      : computed by T5 utility scorer (float)
    token_cost   : auto-computed token length of payload
    fingerprint  : MD5 hash for T1 deduplication
    slot_key     : canonical key for T2 slot fusion
    is_delta     : True if T3 delta encoding was applied
    delta_base   : patch_id of the base patch for delta
    causal_depth : depth in causal chain (set by T4)
    """

    patch_id    : str
    turn_index  : int
    patch_type  : str
    payload     : str
    dependencies: List[str] = field(default_factory=list)
    utility     : float = 0.0
    token_cost  : int   = 0
    fingerprint : str   = ""
    slot_key    : str   = ""
    is_delta    : bool  = False
    delta_base  : str   = ""
    causal_depth: int   = 0

    def __post_init__(self):
        self.token_cost  = count_tokens(self.payload)
        self.fingerprint = _fingerprint(self.payload)
        self.slot_key    = f"{self.patch_type}::{_topic_keyword(self.payload)}"

    def to_prompt_str(self) -> str:
        """Format patch as a single prompt line for LLM context injection."""
        dep_str = (
            f" [deps: {', '.join(self.dependencies)}]"
            if self.dependencies else ""
        )
        return f"[{self.patch_type.upper()}] {self.payload}{dep_str}"
