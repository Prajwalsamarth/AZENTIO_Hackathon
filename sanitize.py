"""Prompt-injection defense: deterministic filter first, SLM gate second.

Layer order is deliberate and matches the threat model:

  1. Field allowlist      -- only named fields can reach the prompt at all
  2. Type/pattern checks  -- identifiers must match their ID grammar
  3. Vocabulary allowlist -- closed-set fields must be a known value
  4. Normalization        -- NFKC, strip control/zero-width, length cap
  5. Pattern detection    -- known injection phrasing and role markers
  6. SLM gate             -- only for text that survived 1-5 but is unfamiliar

Nothing that fails a layer reaches the model as text; it is replaced with a
marker and flagged. The flag then feeds an output-side backstop in detector.py:
a suspected injection can never downgrade a high ensemble score.
"""

import re
import unicodedata

MAX_TEXT_LEN = 64
REDACTED = "[REDACTED]"
UNRECOGNIZED = "[UNRECOGNIZED]"

# Fields allowed to carry free text into the prompt. Anything not listed here
# is either numeric, a closed-set category, or simply never shown to the model.
FREE_TEXT_FIELDS = ("merchant_name", "merchant_city", "cust_city", "cust_state",
                    "acc_branch_city")

# Identifiers must match their grammar exactly -- an ID is never free text.
ID_PATTERNS = {
    "transaction_id": re.compile(r"^TXN_\d{1,12}$"),
    "account_id": re.compile(r"^ACC_\d{1,12}$"),
    "customer_id": re.compile(r"^CUST_\d{1,12}$"),
    "merchant_id": re.compile(r"^MER_\d{1,12}$"),
    "device_id": re.compile(r"^DEV_\d{1,12}$"),
}

# Characters that carry no meaning in a merchant name but do carry meaning to a
# tokenizer: control codes, zero-width joiners, bidi overrides.
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏‪-‮⁠-⁯﻿]")
_WS = re.compile(r"\s+")

INJECTION_PATTERNS = [
    (re.compile(r"(?i)\b(ignore|disregard|forget|override)\b.{0,32}"
                r"\b(previous|prior|above|earlier|all)\b"), "override_instruction"),
    (re.compile(r"(?i)\b(classify|mark|label|treat|rate|score)\b.{0,32}"
                r"\b(as|this)\b.{0,24}\b(safe|legitimate|not fraud|benign|low risk)\b"),
     "verdict_steering"),
    (re.compile(r"(?i)\byou are (now|a|an)\b|\bact as\b|\bpretend to be\b"), "role_reassignment"),
    (re.compile(r"(?i)<\|.*?\|>|\b(system|assistant|user)\s*:"), "role_marker"),
    (re.compile(r"(?i)\b(new|updated|revised)\s+(instructions?|rules?|prompt)\b"),
     "instruction_injection"),
    (re.compile(r"```|</?\s*(system|script|prompt)\b"), "delimiter_injection"),
    (re.compile(r"(?i)\b(is_fraud|confidence|justification)\b\s*[:=]"), "schema_steering"),
    (re.compile(r"https?://|\bwww\.", re.I), "embedded_url"),
    (re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"), "encoded_blob"),
]


def normalize(text: str) -> str:
    """NFKC-fold, strip invisible characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CONTROL.sub("", text)
    return _WS.sub(" ", text).strip()


def detect_injection(text: str):
    """-> list of pattern names that fired. Empty list means clean."""
    return [name for pattern, name in INJECTION_PATTERNS if pattern.search(text)]


def sanitize_text(value, vocabulary=None):
    """Deterministic stages 4-5 (+3 when a vocabulary is supplied).

    Returns (safe_value, findings) where findings is a list of reason strings.
    """
    if value is None:
        return None, []

    findings = []
    text = normalize(value)
    if text != str(value).strip():
        findings.append("normalized")

    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]
        findings.append("truncated")

    hits = detect_injection(text)
    if hits:
        return REDACTED, findings + [f"injection:{name}" for name in hits]

    # A closed-set field whose value is not in the set is, by definition, not
    # data this system produced. For merchant_name (35 known values) this alone
    # closes the injection channel completely.
    if vocabulary is not None and text not in vocabulary:
        return UNRECOGNIZED, findings + ["out_of_vocabulary"]

    return text, findings


def sanitize_identifier(field: str, value):
    if value is None:
        return None, []
    text = normalize(value)
    pattern = ID_PATTERNS.get(field)
    if pattern and not pattern.match(text):
        return REDACTED, [f"malformed_identifier:{field}"]
    return text, []


class Sanitizer:
    """Holds the closed-set vocabularies learned from the cleaned corpus."""

    def __init__(self, vocabularies=None, gate=None):
        self.vocabularies = vocabularies or {}
        # gate: optional callable(text) -> bool ("is this an instruction?").
        self.gate = gate
        self.gate_calls = 0

    @classmethod
    def fit(cls, frames: dict):
        """Learn allowed values for each free-text field from cleaned data."""
        vocabularies = {}
        sources = {
            "merchant_name": ("transactions", "merchant_name"),
            "merchant_city": ("transactions", "merchant_city"),
            "cust_city": ("customers", "city"),
            "cust_state": ("customers", "state"),
            "acc_branch_city": ("accounts", "branch_city"),
        }
        for field, (table, column) in sources.items():
            frame = frames.get(table)
            if frame is not None and column in frame.columns:
                values = frame[column].dropna().astype(str).map(normalize)
                vocabularies[field] = sorted(set(values) - {""})
        return cls(vocabularies)

    def sanitize_record(self, record: dict) -> tuple:
        """Sanitize every field that could reach the prompt.

        -> (clean_record, findings) where findings maps field -> reasons.
        """
        clean = dict(record)
        findings = {}

        for field in ID_PATTERNS:
            if field in clean:
                value, reasons = sanitize_identifier(field, clean[field])
                clean[field] = value
                if reasons:
                    findings[field] = reasons

        for field in FREE_TEXT_FIELDS:
            if field not in clean:
                continue
            vocabulary = self.vocabularies.get(field)
            vocabulary = set(vocabulary) if vocabulary else None
            value, reasons = sanitize_text(clean[field], vocabulary)

            # Stage B: clean, but unfamiliar. Ask the SLM whether the string is
            # data or an instruction -- in isolation, so a payload cannot see
            # or influence the transaction it was smuggled into.
            if value == UNRECOGNIZED and self.gate is not None:
                original = normalize(clean[field])
                self.gate_calls += 1
                if self.gate(original):
                    reasons = reasons + ["slm_gate:instruction"]
                else:
                    value = original
                    reasons = reasons + ["slm_gate:data"]

            clean[field] = value
            if reasons:
                findings[field] = reasons

        return clean, findings

    @staticmethod
    def is_suspected(findings: dict) -> bool:
        """True when any finding indicates an attack rather than mere tidying."""
        return any(
            reason.startswith(("injection:", "malformed_identifier:"))
            or reason in {"out_of_vocabulary", "slm_gate:instruction"}
            for reasons in findings.values() for reason in reasons
        )

    def to_dict(self):
        return {"vocabularies": self.vocabularies}

    @classmethod
    def from_dict(cls, payload, gate=None):
        return cls(vocabularies=payload.get("vocabularies"), gate=gate)


def sanitize_series(series):
    """pandas-facing wrapper so stage 1's cleaners.py can use the same rules."""
    return series.map(lambda v: v if v is None else sanitize_text(v)[0],
                      na_action="ignore")
