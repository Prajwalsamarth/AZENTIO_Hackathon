"""PII boundary for anything leaving this machine.

Three independent mechanisms, because one is not enough for a guarantee:

  1. ALLOWLIST   -- payloads are BUILT from named fields, never filtered down
                    from a record. A new PII column upstream is therefore
                    excluded by default instead of leaking until noticed.
  2. SCANNER     -- the serialized payload is pattern-matched for identifiers
                    and for every real customer name, and RAISES on a hit.
                    Fail-closed, so the guarantee does not rest on the
                    allowlist being maintained perfectly.
  3. DRY RUN     -- every payload is written to disk for inspection before any
                    network call, so the claim is verifiable rather than trusted.
"""

import re

# Fields permitted to leave the machine. Nothing else is ever serialized.
ALLOWED_FIELDS = (
    "amount", "currency", "transaction_type", "channel", "status",
    "merchant_category", "merchant_country", "merchant_city",
    "device_type", "is_new_device", "auth_method", "is_card_present",
    "is_foreign_transaction", "distance_from_home_km",
    "time_since_prev_txn_mins", "txn_count_last_24h", "txn_count_last_7d",
    "amount_to_account_avg_ratio", "balance_after_txn",
    "transaction_hour", "is_weekend",
    "acc_account_type", "acc_account_tier",
    "cust_risk_rating", "cust_kyc_status",
)

# Withheld deliberately. Kept as an explicit record of intent -- the allowlist
# already excludes them, but a reader should be able to see the reasoning.
WITHHELD = {
    "first_name": "direct identifier",
    "last_name": "direct identifier",
    "email": "direct identifier",
    "phone_number": "direct identifier",
    "date_of_birth": "direct identifier",
    "postal_code": "direct identifier",
    "ip_address": "direct identifier",
    "device_id": "device-level identifier",
    "customer_id": "links back to a real person",
    "account_id": "links back to a real person",
    "transaction_id": "links back to a real record",
    "merchant_id": "links back to a real record",
    "merchant_name": "not PII, but the one attacker-controlled free-text field",
    "cust_city": "quasi-identifier; distance_from_home_km conveys what matters",
    "cust_state": "quasi-identifier",
    "cust_age": "quasi-identifier (age + city + income re-identifies in n=124)",
    "cust_annual_income": "quasi-identifier",
}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+\d{1,3}[-\s]?)?\d{10,}")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ID_LIKE = re.compile(r"\b(?:CUST|ACC|TXN|MER|DEV)_\d+\b")


class PIILeakError(RuntimeError):
    """Raised when a payload about to be sent contains identifying data."""


def project(record: dict) -> dict:
    """Build an outbound payload from the allowlist. Never filters a record."""
    return {field: record.get(field) for field in ALLOWED_FIELDS
            if record.get(field) is not None}


def _name_pattern(names) -> re.Pattern:
    tokens = sorted({n.strip() for n in names if n and len(str(n).strip()) > 2},
                    key=len, reverse=True)
    if not tokens:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b",
                      re.IGNORECASE)


def build_name_scanner(customers_frame):
    """Compile a matcher over every real first and last name in the corpus."""
    names = []
    for column in ("first_name", "last_name"):
        if column in customers_frame.columns:
            names.extend(customers_frame[column].dropna().astype(str).tolist())
    return _name_pattern(names)


def scan(text: str, name_pattern: re.Pattern = None) -> list:
    """-> list of (kind, matched_text). Empty list means clean."""
    findings = []
    for kind, pattern in (("email", EMAIL), ("phone", PHONE),
                          ("ipv4", IPV4), ("identifier", ID_LIKE)):
        for match in pattern.findall(text):
            findings.append((kind, match if isinstance(match, str) else str(match)))
    if name_pattern is not None:
        for match in name_pattern.findall(text):
            findings.append(("customer_name", match))
    return findings


def assert_clean(text: str, name_pattern: re.Pattern = None, context: str = ""):
    """Fail closed. Called immediately before every outbound request."""
    findings = scan(text, name_pattern)
    if findings:
        kinds = sorted({kind for kind, _ in findings})
        raise PIILeakError(
            f"refusing to send{' ' + context if context else ''}: "
            f"payload contains {kinds} ({len(findings)} match(es)). "
            f"This is a bug in the allowlist -- fix redact.ALLOWED_FIELDS."
        )
    return True
