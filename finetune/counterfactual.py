"""Counterfactual construction: make fraud rows by intervening on real ones.

A counterfactual pair is the same transaction twice, differing only in the
causal drivers of fraud and whatever those drivers mechanically determine. That
is what separates "this change made it fraud" from "these features correlate
with the flag" -- and it is why the label is KNOWN rather than inferred: the row
is fraud because we intervened to make it so.

Consistency repair is the part that matters. An intervention that raises the
amount without recomputing the account ratio and the resulting balance produces
a row that cannot physically exist, and a model trained on it learns noise.
Stage 1's validation invariants are reused here as generators: the same four
checks that verified the cleaning now gate the synthesis.
"""

import random
from datetime import datetime

import pandas as pd

HOME_COUNTRY = "IN"

# Foreign city/country pairs, so a geographic intervention stays coherent.
FOREIGN_LOCATIONS = [
    ("UA", "Kyiv", 9100.0), ("RU", "Moscow", 7900.0), ("NG", "Lagos", 9600.0),
    ("RO", "Bucharest", 8700.0), ("PH", "Manila", 4600.0), ("PA", "Panama City", 16200.0),
]
HIGH_RISK_CATEGORIES = ["GAMBLING", "CRYPTO", "GIFT_CARD", "MONEY_TRANSFER"]

# Merchant names are NOT set by interventions -- they are cleared. A synthetic
# row must not carry a real brand that the sanitizer's vocabulary would accept,
# and merchant_name never reaches the generation payload anyway.
RECIPES = ("GEO_IMPOSSIBLE", "HIGH_RISK_MCC", "AMOUNT_ESCALATION",
           "AUTH_STRIPPED", "VELOCITY_BURST", "NIGHT_SHIFT")


class InconsistentRow(RuntimeError):
    """A generated row violated an invariant the cleaned data satisfies."""


def account_average_amounts(transactions: pd.DataFrame) -> dict:
    """Recover each account's implied average txn amount.

    `amount_to_account_avg_ratio` is amount / average, so the average is
    recoverable from any row where both are present. Needed to keep the ratio
    truthful after an amount intervention.
    """
    frame = transactions[["account_id", "amount", "amount_to_account_avg_ratio"]].dropna()
    frame = frame[frame.amount_to_account_avg_ratio > 0]
    implied = frame.amount / frame.amount_to_account_avg_ratio
    return implied.groupby(frame.account_id).median().to_dict()


# --- interventions --------------------------------------------------------

def _geo_impossible(row, rng):
    country, city, distance = rng.choice(FOREIGN_LOCATIONS)
    row["merchant_country"] = country
    row["merchant_city"] = city
    row["distance_from_home_km"] = round(distance * rng.uniform(0.85, 1.15), 1)
    row["is_card_present"] = True          # present here AND at home: impossible
    return row


def _high_risk_mcc(row, rng):
    row["merchant_category"] = rng.choice(HIGH_RISK_CATEGORIES)
    return row


def _amount_escalation(row, rng):
    base = row.get("amount")
    if base is None or base == 0:
        base = 2000.0
    row["amount"] = round(abs(base) * rng.uniform(50, 200), 2)
    return row


def _auth_stripped(row, rng):
    row["auth_method"] = "NONE"
    row["is_new_device"] = True
    return row


def _velocity_burst(row, rng):
    row["time_since_prev_txn_mins"] = round(rng.uniform(0.5, 4.5), 2)
    count = rng.randint(5, 7)
    row["txn_count_last_24h"] = count
    row["txn_count_last_7d"] = max(count, row.get("txn_count_last_7d") or 0)
    return row


def _night_shift(row, rng):
    row["transaction_hour"] = rng.randint(2, 4)
    return row


INTERVENTIONS = {
    "GEO_IMPOSSIBLE": _geo_impossible,
    "HIGH_RISK_MCC": _high_risk_mcc,
    "AMOUNT_ESCALATION": _amount_escalation,
    "AUTH_STRIPPED": _auth_stripped,
    "VELOCITY_BURST": _velocity_burst,
    "NIGHT_SHIFT": _night_shift,
}


# --- consistency repair ---------------------------------------------------

def repair(row: dict, original: dict, account_average: float) -> dict:
    """Recompute every field the interventions mechanically determine."""
    # is_foreign_transaction follows from the merchant's country.
    country = row.get("merchant_country")
    if country is not None:
        row["is_foreign_transaction"] = bool(country != HOME_COUNTRY)

    # The timestamp is the source of truth for hour and weekend.
    timestamp = row.get("transaction_timestamp")
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    if isinstance(timestamp, (pd.Timestamp, datetime)):
        hour = row.get("transaction_hour")
        if hour is not None and int(hour) != timestamp.hour:
            timestamp = pd.Timestamp(timestamp).replace(hour=int(hour))
        row["transaction_timestamp"] = pd.Timestamp(timestamp)
        row["transaction_hour"] = int(row["transaction_timestamp"].hour)
        row["is_weekend"] = bool(row["transaction_timestamp"].dayofweek >= 5)

    # The ratio is amount / account average, so a new amount changes it.
    amount = row.get("amount")
    if amount is not None and account_average:
        row["amount_to_account_avg_ratio"] = round(abs(amount) / account_average, 3)

    # Spending more leaves less behind: shift the resulting balance by the delta.
    original_amount, original_balance = original.get("amount"), original.get("balance_after_txn")
    if None not in (amount, original_amount, original_balance):
        row["balance_after_txn"] = round(original_balance - (amount - original_amount), 2)

    # A synthetic row must not claim a real merchant identity.
    row["merchant_name"] = None
    row["merchant_id"] = None
    return row


def validate(row: dict, account_average: float, tolerance: float = 0.02):
    """Stage 1's invariants, reused to gate synthesis. Raises on violation."""
    problems = []
    timestamp = row.get("transaction_timestamp")
    if isinstance(timestamp, (pd.Timestamp, datetime)):
        if int(row.get("transaction_hour", -1)) != timestamp.hour:
            problems.append("transaction_hour != timestamp hour")
        if bool(row.get("is_weekend")) != bool(timestamp.dayofweek >= 5):
            problems.append("is_weekend != timestamp weekday")

    country = row.get("merchant_country")
    if country is not None and bool(row.get("is_foreign_transaction")) != (country != HOME_COUNTRY):
        problems.append("is_foreign_transaction != (merchant_country != IN)")

    amount, ratio = row.get("amount"), row.get("amount_to_account_avg_ratio")
    if None not in (amount, ratio) and account_average:
        expected = abs(amount) / account_average
        if abs(expected - ratio) > max(tolerance, tolerance * expected):
            problems.append(f"ratio {ratio} != amount/avg {expected:.3f}")

    if problems:
        raise InconsistentRow("; ".join(problems))
    return True


# --- generation -----------------------------------------------------------

def make_counterfactual(factual: dict, account_average: float, rng: random.Random,
                        n_recipes: int = 3) -> dict:
    """Apply n interventions to one real transaction and repair the result."""
    recipes = rng.sample(list(RECIPES), n_recipes)
    row = dict(factual)
    for recipe in recipes:
        row = INTERVENTIONS[recipe](row, rng)
    row = repair(row, factual, account_average)
    validate(row, account_average)
    row["_recipes"] = recipes
    return row


def changed_fields(factual: dict, counterfactual: dict) -> dict:
    """Diff for auditing -- proves only intended fields (and dependents) moved."""
    diff = {}
    for key in set(factual) | set(counterfactual):
        if key.startswith("_"):
            continue
        before, after = factual.get(key), counterfactual.get(key)
        if pd.isna(before) if not isinstance(before, (list, dict)) else False:
            before = None
        if before != after and not (before is None and after is None):
            diff[key] = {"from": before, "to": after}
    return diff
