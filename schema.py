"""Declarative schema for the Azentio hackathon CSVs.

This module is the single source of truth: every coercion the cleaner performs
is driven by a ColumnSpec here, so adding or retyping a column is a one-line
change and never a code change in cleaners.py / pipeline.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

# Tokens that mean "missing" regardless of column. NONE is included here, so
# any column where NONE is a legitimate value (card_type, auth_method) must
# override null_tokens -- see NULLS_KEEP_NONE below.
DEFAULT_NULL_TOKENS: Tuple[str, ...] = (
    "", "NA", "N/A", "NAN", "NULL", "NONE", "NOT_AVAILABLE", "NOT AVAILABLE",
    "UNKNOWN", "-", "--", "?",
)
NULLS_KEEP_NONE: Tuple[str, ...] = tuple(t for t in DEFAULT_NULL_TOKENS if t != "NONE")

# Ordered parse attempts. Explicit formats only -- never pandas inference,
# which would silently read 07/03/2026 as March 7th on some rows and July 3rd
# on others. Day-first is pinned deliberately for the slash/dash forms.
DATETIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    # Slash-separated rows are day-first and dash-separated rows are month-first.
    # This is not a guess: both orders were parsed for every such row and scored
    # against the independent is_weekend / transaction_hour columns. Day-first
    # matches 53/53 slash rows (month-first: 14/53); month-first matches 12/12
    # dash rows (day-first: 3/12, and 05-24-2026 has no valid day-first reading).
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y",
)
DATE_FORMATS: Tuple[str, ...] = DATETIME_FORMATS


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str                       # string|int|float|bool|date|datetime|category
    required: bool = False           # null after cleaning -> row is flagged/rejected
    unique: bool = False             # primary key
    categories: Tuple[str, ...] = ()
    aliases: Dict[str, str] = field(default_factory=dict)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    strip_currency: bool = False     # drop 'INR '/'Rs.'/currency symbol + thousands separators
    sanitize: bool = False           # free-text -> route through the injection-filter seam
    datetime_formats: Tuple[str, ...] = ()
    null_tokens: Tuple[str, ...] = DEFAULT_NULL_TOKENS


def _cat(name, categories, **kw):
    return ColumnSpec(name, "category", categories=tuple(categories), **kw)


CUSTOMERS = (
    ColumnSpec("customer_id", "string", required=True, unique=True),
    ColumnSpec("first_name", "string", sanitize=True),
    ColumnSpec("last_name", "string", sanitize=True),
    _cat("gender", ("M", "F", "OTHER")),
    ColumnSpec("date_of_birth", "date", datetime_formats=DATE_FORMATS),
    ColumnSpec("age", "int", min_value=18, max_value=120),
    ColumnSpec("email", "string", sanitize=True),
    ColumnSpec("phone_number", "string"),
    ColumnSpec("city", "string"),
    ColumnSpec("state", "string"),
    _cat("country", ("IN",)),
    ColumnSpec("postal_code", "string"),   # string: identifier, not a quantity
    _cat("occupation", (
        "BUSINESS_OWNER", "FREELANCER", "GOVERNMENT_EMPLOYEE", "HOMEMAKER",
        "RETIRED", "SALARIED_BANKING", "SALARIED_HEALTHCARE", "SALARIED_IT",
        "SELF_EMPLOYED", "STUDENT",
    )),
    ColumnSpec("annual_income", "float", min_value=0),
    _cat("marital_status", ("SINGLE", "MARRIED", "DIVORCED", "WIDOWED")),
    _cat("education_level", ("HIGH_SCHOOL", "DIPLOMA", "GRADUATE", "POST_GRADUATE", "DOCTORATE")),
    _cat("employment_status", ("EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED", "RETIRED", "STUDENT")),
    ColumnSpec("customer_since", "date", datetime_formats=DATE_FORMATS),
    _cat("customer_segment", ("RETAIL", "PREMIUM", "PRIVATE_BANKING", "SME")),
    _cat("kyc_status", ("VERIFIED", "PENDING", "EXPIRED", "REJECTED")),
    _cat("risk_rating", ("LOW", "MEDIUM", "HIGH")),
    ColumnSpec("is_politically_exposed", "bool"),
    _cat("preferred_channel", ("BRANCH", "INTERNET_BANKING", "MOBILE_APP", "PHONE_BANKING")),
    ColumnSpec("email_verified", "bool"),
    ColumnSpec("phone_verified", "bool"),
    ColumnSpec("num_complaints_last_year", "int", min_value=0),
)

ACCOUNTS = (
    ColumnSpec("account_id", "string", required=True, unique=True),
    ColumnSpec("customer_id", "string", required=True),
    _cat("account_type", ("SAVINGS", "CURRENT", "CREDIT_CARD", "SALARY", "NRE")),
    _cat("account_status", ("ACTIVE", "DORMANT", "FROZEN", "CLOSED")),
    _cat("currency", ("INR",)),
    ColumnSpec("open_date", "date", datetime_formats=DATE_FORMATS),
    ColumnSpec("close_date", "date", datetime_formats=DATE_FORMATS),
    ColumnSpec("branch_code", "string"),
    ColumnSpec("branch_city", "string"),
    ColumnSpec("current_balance", "float"),          # negatives legitimate (overdraft)
    ColumnSpec("avg_monthly_balance_6m", "float"),
    ColumnSpec("credit_limit", "float", min_value=0),
    ColumnSpec("credit_utilization_pct", "float", min_value=0, max_value=100),
    ColumnSpec("overdraft_enabled", "bool"),
    # NONE is a real card_type, not a missing marker.
    _cat("card_type", ("NONE", "CLASSIC", "GOLD", "PLATINUM", "SIGNATURE"),
         null_tokens=NULLS_KEEP_NONE),
    ColumnSpec("is_joint_account", "bool"),
    ColumnSpec("num_linked_devices", "int", min_value=0),
    ColumnSpec("mobile_banking_enrolled", "bool"),
    ColumnSpec("last_login_date", "date", datetime_formats=DATE_FORMATS),
    ColumnSpec("avg_monthly_txn_count", "int", min_value=0),
    _cat("account_tier", ("BASIC", "SILVER", "GOLD", "PLATINUM")),
)

TRANSACTIONS = (
    ColumnSpec("transaction_id", "string", required=True, unique=True),
    ColumnSpec("account_id", "string", required=True),
    ColumnSpec("customer_id", "string"),
    ColumnSpec("transaction_timestamp", "datetime", required=True,
               datetime_formats=DATETIME_FORMATS),
    ColumnSpec("transaction_hour", "int", min_value=0, max_value=23),
    ColumnSpec("is_weekend", "bool"),
    ColumnSpec("amount", "float", strip_currency=True),
    _cat("currency", ("INR",)),
    _cat("transaction_type", ("PURCHASE", "PAYMENT", "TRANSFER", "WITHDRAWAL")),
    _cat("channel", ("POS", "ATM", "ONLINE", "MOBILE_APP", "INTERNET_BANKING")),
    _cat("status", ("SUCCESS", "FAILED", "PENDING", "REVERSED")),
    ColumnSpec("merchant_id", "string"),
    ColumnSpec("merchant_name", "string", sanitize=True),
    _cat("merchant_category", (
        "APPAREL", "ATM_WITHDRAWAL", "CRYPTO", "ECOMMERCE", "ELECTRONICS",
        "ENTERTAINMENT", "FOOD_DELIVERY", "FUEL", "FUND_TRANSFER", "GAMBLING",
        "GIFT_CARD", "GROCERY", "HEALTHCARE", "INSURANCE", "JEWELLERY",
        "MONEY_TRANSFER", "P2P_TRANSFER", "RESTAURANT", "SUBSCRIPTION",
        "TRANSPORT", "TRAVEL", "UTILITIES",
    )),
    ColumnSpec("merchant_city", "string"),
    _cat("merchant_country", ("IN", "AE", "GB", "ID", "NG", "PA", "PH", "RO", "RU", "SG", "UA")),
    ColumnSpec("device_id", "string"),
    _cat("device_type", ("ANDROID", "IOS", "MACOS", "WINDOWS_DESKTOP",
                         "POS_TERMINAL", "ATM_MACHINE")),
    ColumnSpec("is_new_device", "bool"),
    ColumnSpec("ip_address", "string"),
    # NONE is a real auth_method (no authentication performed), not missing.
    _cat("auth_method", ("NONE", "PIN", "OTP", "BIOMETRIC", "SIGNATURE", "CVV", "3DS"),
         null_tokens=NULLS_KEEP_NONE),
    ColumnSpec("is_card_present", "bool"),
    ColumnSpec("is_foreign_transaction", "bool"),
    ColumnSpec("distance_from_home_km", "float", min_value=0),
    ColumnSpec("time_since_prev_txn_mins", "float", min_value=0),
    ColumnSpec("txn_count_last_24h", "int", min_value=0),
    ColumnSpec("txn_count_last_7d", "int", min_value=0),
    ColumnSpec("amount_to_account_avg_ratio", "float", min_value=0),
    ColumnSpec("balance_after_txn", "float"),        # negatives legitimate
)

TABLES = {
    "customers": CUSTOMERS,
    "accounts": ACCOUNTS,
    "transactions": TRANSACTIONS,
}

PRIMARY_KEY = {
    "customers": "customer_id",
    "accounts": "account_id",
    "transactions": "transaction_id",
}
