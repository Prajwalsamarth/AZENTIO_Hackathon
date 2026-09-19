"""Enrichment and feature construction.

The single most important property here: `build_feature_row()` computes ONE
record's features from ONE enriched dict. Batch mode maps it over rows rather
than taking a vectorized shortcut, which is what guarantees the batch and
single-record paths cannot diverge.
"""

import math

import numpy as np
import pandas as pd

import config
import schema

# Categorical vocabularies are taken straight from the stage-1 schema, so the
# one-hot layout is pinned by the same source of truth that cleaned the data.
_CATS = {s.name: s.categories for s in schema.TRANSACTIONS if s.dtype == "category"}
_ACC_CATS = {s.name: s.categories for s in schema.ACCOUNTS if s.dtype == "category"}
_CUST_CATS = {s.name: s.categories for s in schema.CUSTOMERS if s.dtype == "category"}

ONE_HOT = {
    "channel": _CATS["channel"],
    "auth_method": _CATS["auth_method"],
    "device_type": _CATS["device_type"],
    "transaction_type": _CATS["transaction_type"],
    "merchant_category": _CATS["merchant_category"],
    "acc_account_type": _ACC_CATS["account_type"],
    "acc_account_tier": _ACC_CATS["account_tier"],
    "cust_risk_rating": _CUST_CATS["risk_rating"],
    "cust_kyc_status": _CUST_CATS["kyc_status"],
}

NUMERIC = [
    "amount", "amount_to_account_avg_ratio", "distance_from_home_km",
    "time_since_prev_txn_mins", "txn_count_last_24h", "txn_count_last_7d",
    "balance_after_txn", "acc_current_balance", "acc_avg_monthly_balance_6m",
    "acc_credit_utilization_pct", "acc_avg_monthly_txn_count",
    "cust_annual_income", "cust_age", "cust_num_complaints_last_year",
]

BOOLEAN = [
    "is_new_device", "is_card_present", "is_foreign_transaction", "is_weekend",
    "acc_overdraft_enabled", "acc_mobile_banking_enrolled", "acc_is_joint_account",
    "cust_is_politically_exposed",
]


def _clean(value):
    """NaN/NaT/pd.NA -> None, numpy scalars -> python scalars."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_index(df: pd.DataFrame, key: str) -> dict:
    """key -> record dict, for O(1) enrichment of a single transaction."""
    records = df.to_dict(orient="records")
    return {r[key]: {k: _clean(v) for k, v in r.items()} for r in records}


def enrich(transaction: dict, accounts_idx: dict, customers_idx: dict) -> dict:
    """Join one transaction to its account and customer.

    Orphans (8 rows, account_id absent from accounts.csv) are not an error --
    their account/customer fields stay None and the missing-indicator features
    carry that fact into the model.
    """
    txn = {k: _clean(v) for k, v in transaction.items()}
    out = dict(txn)

    account = accounts_idx.get(txn.get("account_id")) or {}
    for key, value in account.items():
        if key != "account_id":
            out[f"acc_{key}"] = value

    # Trust the account's owner over the transaction's own customer_id: stage 1
    # verified they never conflict, and the account is the authoritative link.
    customer_id = account.get("customer_id") or txn.get("customer_id")
    customer = customers_idx.get(customer_id) or {}
    for key, value in customer.items():
        if key != "customer_id":
            out[f"cust_{key}"] = value

    out["_has_account"] = bool(account)
    out["_has_customer"] = bool(customer)
    return out


def _num(record, key):
    value = record.get(key)
    return None if value is None else float(value)


def _bool(record, key):
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().upper() in {"1", "Y", "YES", "TRUE", "T"}
    return bool(value)


def build_feature_row(record: dict) -> dict:
    """One enriched record -> an ordered dict of raw (pre-imputation) features.

    Missing values stay None here; imputation happens in FeatureBuilder so the
    fitted medians come from the training corpus, not from each record.
    """
    row = {}

    for name in NUMERIC:
        row[name] = _num(record, name)
    for name in BOOLEAN:
        value = _bool(record, name)
        row[name] = None if value is None else float(value)

    amount = row["amount"]
    balance = row["acc_current_balance"]
    hour = _num(record, "transaction_hour")
    distance = row["distance_from_home_km"]

    # Derived signals -- the domain knowledge the raw columns don't express.
    row["log_amount"] = None if amount is None else math.log1p(abs(amount))
    row["amount_to_balance"] = (
        None if amount is None or not balance else abs(amount) / balance
    )
    # Hour as a circle, so 23:00 and 01:00 are neighbours rather than extremes.
    row["hour_sin"] = None if hour is None else math.sin(2 * math.pi * hour / 24)
    row["hour_cos"] = None if hour is None else math.cos(2 * math.pi * hour / 24)
    lo, hi = config.NIGHT_HOURS
    row["is_night"] = None if hour is None else float(lo <= hour <= hi)

    merchant_country = record.get("merchant_country")
    row["country_mismatch"] = (
        None if merchant_country is None else float(merchant_country != config.HOME_COUNTRY)
    )
    merchant_city, home_city = record.get("merchant_city"), record.get("cust_city")
    row["city_mismatch"] = (
        None if not merchant_city or not home_city else float(merchant_city != home_city)
    )

    card_present = row["is_card_present"]
    row["impossible_travel"] = (
        None if card_present is None or distance is None
        else float(card_present == 1.0 and distance > config.FAR_FROM_HOME_KM)
    )
    after = row["balance_after_txn"]
    row["balance_went_negative"] = None if after is None else float(after < 0)

    category = record.get("merchant_category")
    row["high_risk_category"] = (
        None if category is None else float(category in config.HIGH_RISK_CATEGORIES)
    )

    row["has_account"] = float(bool(record.get("_has_account")))
    row["has_customer"] = float(bool(record.get("_has_customer")))

    # Missingness is itself signal -- a transaction with no velocity history
    # differs from one with a velocity of zero. Imputation would erase that,
    # so record it explicitly before the median fills the value in.
    for name in NUMERIC:
        row[f"{name}_missing"] = float(row[name] is None)

    for field, categories in ONE_HOT.items():
        value = record.get(field)
        for category_value in categories:
            row[f"{field}={category_value}"] = (
                None if value is None else float(value == category_value)
            )

    return row


FEATURE_NAMES = list(build_feature_row({}).keys())

# Continuous and derived features only -- everything that is not a one-hot
# indicator. The autoencoder is fitted on this subspace; see anomaly.py for why.
CONTINUOUS_MASK = [("=" not in name) for name in FEATURE_NAMES]


class FeatureBuilder:
    """Fits imputation medians + standardization, then emits fixed-order vectors."""

    def __init__(self, feature_names=None, medians=None, mean=None, scale=None):
        self.feature_names = feature_names or FEATURE_NAMES
        self.medians = medians
        self.mean = mean
        self.scale = scale

    def fit(self, records):
        rows = [build_feature_row(r) for r in records]
        frame = pd.DataFrame(rows, columns=self.feature_names).astype("float64")
        self.medians = frame.median(numeric_only=True).fillna(0.0).to_dict()
        filled = self._impute(frame)
        self.mean = filled.mean().to_dict()
        scale = filled.std(ddof=0).replace(0.0, 1.0)
        self.scale = scale.to_dict()
        return self

    def _impute(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.fillna(value=self.medians).fillna(0.0)

    def transform(self, records):
        """-> (X_scaled, X_raw, missing_mask). Same code path for 1 or N records."""
        rows = [build_feature_row(r) for r in records]
        frame = pd.DataFrame(rows, columns=self.feature_names).astype("float64")
        missing = frame.isna().to_numpy()
        filled = self._impute(frame)
        mean = pd.Series(self.mean).reindex(self.feature_names)
        scale = pd.Series(self.scale).reindex(self.feature_names).replace(0.0, 1.0)
        scaled = (filled - mean) / scale
        return (scaled.to_numpy(dtype="float64"),
                filled.to_numpy(dtype="float64"), missing)

    def to_dict(self):
        return {"feature_names": self.feature_names, "medians": self.medians,
                "mean": self.mean, "scale": self.scale}

    @classmethod
    def from_dict(cls, payload):
        builder = cls(**payload)
        # Feature order is pinned: a schema change must fail loudly here rather
        # than silently shift every vector by one column.
        if builder.feature_names != FEATURE_NAMES:
            raise ValueError(
                "Persisted feature order does not match the current code. "
                "Re-run `run_fraud.py fit` after changing features or schema."
            )
        return builder
