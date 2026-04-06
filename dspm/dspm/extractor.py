"""
extractor.py — Groq API client pool + semantic patch extraction.

GroqClientPool : manages multiple API keys with automatic failover
extract_patches : extracts SemanticPatch objects from a dialogue turn
"""

import re
import json
import time
from collections import defaultdict
from typing import List, Dict

from groq import Groq, RateLimitError, APIStatusError

from dspm.config import (
    GROQ_MODEL, PATCH_TYPES,
    MAX_EXTRACTION_TOKENS, EXTRACTION_TEMPERATURE,
    API_SLEEP_BETWEEN_CALLS,
)
from dspm.structures import SemanticPatch

# ── Global error log (shared across all calls) ────────────────────────────
ERROR_LOG: List[Dict] = []

# ── Extraction System Prompt ──────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a precision semantic extraction engine \
for long-context LLM memory management.

Given a conversational turn, extract ALL semantically meaningful units \
as a JSON array.

Each object MUST have exactly these keys:
  "type"         : one of [constraint, decision, code, equation, entity, structure]
  "payload"      : concise 8-20 word summary preserving technical precision
  "dependencies" : list of patch_ids this patch logically depends on (can be empty)
  "patch_id"     : unique string "p{turn}_{index}" e.g. "p3_0", "p3_1"

Extraction rules:
- constraints : hard requirements, limits, non-negotiable specs
- decisions   : explicit choices made, options selected or rejected
- code        : technology stack, library, API, or implementation choices
- equation    : formulas, thresholds, numerical specs, mathematical relations
- entity      : named systems, tools, people, organisations, acronyms
- structure   : schemas, data formats, workflow steps, architectural patterns

CRITICAL: return ONLY a valid JSON array. No markdown, no explanation, no wrapping.
If nothing is extractable, return [].
"""


class GroqClientPool:
    """
    Manages a pool of Groq API clients with automatic failover.

    Switches to the next available key on rate-limit or service errors,
    with exponential backoff to respect API rate windows.

    Parameters
    ----------
    keys : list of Groq API key strings (empty/None values are filtered out)
    """

    def __init__(self, keys: List[str]):
        self._keys    = [k for k in keys if k]
        self._idx     = 0
        self._clients = [Groq(api_key=k) for k in self._keys]
        self._errors: Dict[str, int] = defaultdict(int)

        if not self._clients:
            raise ValueError(
                "No valid Groq API keys provided. "
                "Set GROQ_API_KEY in your environment or Colab Secrets."
            )

    @property
    def client(self):
        return self._clients[self._idx]

    def _next(self):
        self._idx = (self._idx + 1) % len(self._clients)

    def chat(self, **kwargs):
        """
        Make a Groq chat completion call with automatic failover.
        Retries across all keys with exponential backoff.
        """
        last_exc    = None
        backoff_seq = [5, 15, 30, 60]

        for attempt in range(len(self._clients) * 2):
            try:
                return self.client.chat.completions.create(**kwargs)

            except RateLimitError as e:
                self._errors[f"key_{self._idx + 1}_rate_limit"] += 1
                wait = backoff_seq[min(attempt, len(backoff_seq) - 1)]
                time.sleep(wait)
                self._next()
                last_exc = e

            except APIStatusError as e:
                self._errors[f"key_{self._idx + 1}_api_error"] += 1
                if e.status_code in (429, 503):
                    time.sleep(30)
                    self._next()
                    last_exc = e
                else:
                    raise

        time.sleep(60)
        raise RuntimeError(
            "All Groq API keys temporarily unavailable."
        ) from last_exc

    def error_summary(self) -> Dict[str, int]:
        """Return a summary of all API errors encountered."""
        return dict(self._errors)


# ── Patch Extraction ──────────────────────────────────────────────────────

def extract_patches(
    turn_text      : str,
    turn_index     : int,
    pool           : GroqClientPool,
    recent_context : str = "",
    dialogue_name  : str = "unknown",
) -> List[SemanticPatch]:
    """
    Extract SemanticPatch objects from a single dialogue turn via Groq.

    Parameters
    ----------
    turn_text      : raw text of the current dialogue turn
    turn_index     : index of this turn in the dialogue
    pool           : GroqClientPool instance to use for API calls
    recent_context : recent patch strings for context (optional)
    dialogue_name  : name of dialogue (used for error logging)

    Returns
    -------
    List of SemanticPatch objects (empty list if extraction fails)
    """
    user_content = (
        f"RECENT CONTEXT:\n{recent_context}\n\nCURRENT TURN:\n{turn_text}"
        if recent_context
        else f"CURRENT TURN:\n{turn_text}"
    )

    for attempt in range(3):
        try:
            response = pool.chat(
                model       = GROQ_MODEL,
                messages    = [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature = EXTRACTION_TEMPERATURE,
                max_tokens  = MAX_EXTRACTION_TOKENS,
            )
            raw = response.choices[0].message.content.strip()

            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if not json_match:
                ERROR_LOG.append({
                    "dialogue"   : dialogue_name,
                    "turn"       : turn_index,
                    "error_type" : "json_parse",
                    "message"    : "No JSON array found in response",
                })
                continue

            patch_dicts = json.loads(json_match.group())
            patches     = []

            for d in patch_dicts:
                if not all(k in d for k in ("type", "payload", "patch_id")):
                    continue
                if d["type"] not in PATCH_TYPES:
                    continue
                p = SemanticPatch(
                    patch_id    = d["patch_id"],
                    turn_index  = turn_index,
                    patch_type  = d["type"],
                    payload     = d["payload"][:200],
                    dependencies= [
                        dep for dep in d.get("dependencies", [])
                        if isinstance(dep, str)
                    ],
                )
                patches.append(p)

            time.sleep(API_SLEEP_BETWEEN_CALLS)
            return patches

        except json.JSONDecodeError as e:
            ERROR_LOG.append({
                "dialogue"   : dialogue_name,
                "turn"       : turn_index,
                "error_type" : "json_parse",
                "message"    : str(e),
            })

        except RateLimitError:
            pass

        except Exception as e:
            ERROR_LOG.append({
                "dialogue"   : dialogue_name,
                "turn"       : turn_index,
                "error_type" : "api_error",
                "message"    : str(e),
            })

    # All 3 attempts failed
    ERROR_LOG.append({
        "dialogue"   : dialogue_name,
        "turn"       : turn_index,
        "error_type" : "extraction_failure",
        "message"    : "All 3 attempts failed — returning empty patch list",
    })
    return []
