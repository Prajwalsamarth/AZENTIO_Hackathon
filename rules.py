"""Deterministic rule engine.

Produces a 0-1 score, human-readable reason codes, and a hard-rule flag. The
reason codes are the thread that runs all the way to the final justification,
which is what makes a verdict explainable rather than merely asserted.
"""

import config

# (code, weight, predicate, human-readable template)
# Weights are declared here as data so tuning never touches logic.
RULES = [
    ("FOREIGN_MERCHANT", 3,
     lambda r: r.get("merchant_country") not in (None, config.HOME_COUNTRY),
     "merchant is in {merchant_country}, outside the customer's home country"),

    ("HIGH_RISK_CATEGORY", 3,
     lambda r: r.get("merchant_category") in config.HIGH_RISK_CATEGORIES,
     "high-risk merchant category {merchant_category}"),

    ("NO_AUTH_HIGH_VALUE", 3,
     lambda r: r.get("auth_method") == "NONE"
     and (r.get("amount") or 0) >= config.HIGH_VALUE_AMOUNT,
     "no authentication on a {amount:,.0f} INR transaction"),

    ("AMOUNT_RATIO_EXTREME", 3,
     lambda r: (r.get("amount_to_account_avg_ratio") or 0) > 50,
     "amount is {amount_to_account_avg_ratio:.0f}x the account average"),

    ("AMOUNT_RATIO_HIGH", 2,
     lambda r: 10 < (r.get("amount_to_account_avg_ratio") or 0) <= 50,
     "amount is {amount_to_account_avg_ratio:.1f}x the account average"),

    ("IMPOSSIBLE_TRAVEL", 3,
     lambda r: bool(r.get("is_card_present"))
     and (r.get("distance_from_home_km") or 0) > config.FAR_FROM_HOME_KM,
     "card physically present {distance_from_home_km:,.0f} km from home"),

    ("VELOCITY_24H", 2,
     lambda r: (r.get("txn_count_last_24h") or 0) >= 3,
     "{txn_count_last_24h} transactions in the preceding 24h"),

    ("RAPID_SUCCESSION", 2,
     lambda r: r.get("time_since_prev_txn_mins") is not None
     and r["time_since_prev_txn_mins"] < 5,
     "only {time_since_prev_txn_mins:.1f} minutes since the previous transaction"),

    ("NIGHT_HOURS", 1,
     lambda r: r.get("transaction_hour") is not None
     and config.NIGHT_HOURS[0] <= r["transaction_hour"] <= config.NIGHT_HOURS[1],
     "occurred at {transaction_hour:02.0f}:00, outside normal activity hours"),

    ("NEW_DEVICE", 1,
     lambda r: bool(r.get("is_new_device")),
     "initiated from a device not seen before"),

    ("BALANCE_NEGATIVE", 1,
     lambda r: (r.get("balance_after_txn") is not None
                and r["balance_after_txn"] < 0),
     "drove the account balance negative"),

    ("CUSTOMER_HIGH_RISK", 1,
     lambda r: r.get("cust_risk_rating") == "HIGH",
     "customer carries a HIGH internal risk rating"),

    ("KYC_NOT_VERIFIED", 1,
     lambda r: r.get("cust_kyc_status") not in (None, "VERIFIED"),
     "customer KYC status is {cust_kyc_status}"),

    ("POLITICALLY_EXPOSED", 1,
     lambda r: bool(r.get("cust_is_politically_exposed")),
     "customer is politically exposed"),

    ("ORPHAN_ACCOUNT", 2,
     lambda r: not r.get("_has_account"),
     "references an account that does not exist in the account master"),
]

# Saturation point: the weighted sum that maps to a rule score of 1.0. Set to
# the weight of roughly four strong rules rather than the theoretical maximum,
# so a genuinely bad transaction reaches 1.0 instead of asymptotically nearing it.
SATURATION = 12.0

# Combinations that bypass the ensemble entirely and always reach the SLM. The
# ensemble is a statistical filter; these are cases no filter should be allowed
# to bury.
def is_hard_rule(record: dict, codes: set) -> bool:
    if {"FOREIGN_MERCHANT", "HIGH_RISK_CATEGORY"} <= codes:
        return True
    if "FOREIGN_MERCHANT" in codes and "NO_AUTH_HIGH_VALUE" in codes:
        return True
    if (record.get("amount_to_account_avg_ratio") or 0) > 100:
        return True
    if {"IMPOSSIBLE_TRAVEL", "HIGH_RISK_CATEGORY"} <= codes:
        return True
    return False


def _render(template: str, record: dict) -> str:
    try:
        return template.format(**{k: (v if v is not None else "unknown")
                                  for k, v in record.items()})
    except (KeyError, ValueError, TypeError):
        return template


def evaluate(record: dict) -> dict:
    """One enriched record -> rule score, triggered codes, and reasons."""
    codes, reasons, total = [], [], 0.0
    for code, weight, predicate, template in RULES:
        try:
            hit = bool(predicate(record))
        except (TypeError, ValueError):
            hit = False
        if hit:
            codes.append(code)
            reasons.append(_render(template, record))
            total += weight

    score = min(1.0, total / SATURATION)
    return {
        "rule_score": score,
        "rule_weight": total,
        "rule_codes": codes,
        "rule_reasons": reasons,
        "hard_rule": is_hard_rule(record, set(codes)),
    }
