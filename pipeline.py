"""Orchestration: load -> clean -> dedupe -> repair -> flag -> validate -> write.

Raw files are read as text (dtype=str, keep_default_na=False) so that every
coercion is ours and explicit, never pandas' read_csv heuristics.
"""

import json
from pathlib import Path

import pandas as pd

import schema
from cleaners import clean_column

RAW_DIR = Path("/Users/psamarth/Downloads/AzentioHackathon")
OUT_DIR = Path(__file__).resolve().parent / "cleaned_data"


# --- load & clean ---------------------------------------------------------

def load_raw(table: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{table}.csv", dtype=str, keep_default_na=False)
    expected = [s.name for s in schema.TABLES[table]]
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    if missing:
        raise ValueError(f"{table}.csv is missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"{table}.csv has unexpected columns: {sorted(extra)}")
    return df[expected]


def clean_table(table: str, raw: pd.DataFrame):
    out, report = {}, {}
    for spec in schema.TABLES[table]:
        cleaned, issues = clean_column(raw[spec.name], spec)
        out[spec.name] = cleaned
        report[spec.name] = {
            "dtype": spec.dtype,
            "raw_cardinality": int(raw[spec.name].nunique()),
            "clean_cardinality": int(cleaned.nunique()),
            **issues,
        }
    return pd.DataFrame(out), report


# --- dedupe ---------------------------------------------------------------

def deduplicate(df: pd.DataFrame, table: str, report: dict):
    pk = schema.PRIMARY_KEY[table]
    full_dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Any PK collision left after dropping identical rows is a genuine
    # conflict (same id, different data) -- keep the first, quarantine the rest.
    conflicts = df[df.duplicated(subset=[pk], keep="first")]
    if len(conflicts):
        df = df.drop(index=conflicts.index).reset_index(drop=True)

    report["_dedupe"] = {
        "exact_duplicate_rows_dropped": full_dupes,
        "conflicting_primary_keys_quarantined": int(len(conflicts)),
        "conflicting_ids": sorted(conflicts[pk].dropna().tolist())[:20],
    }
    return df, conflicts


# --- deterministic repairs ------------------------------------------------

def repair_from_lookup(df, key_col, target_col, label, report):
    """Back-fill target_col from a key_col -> target_col map built from the
    table's own non-null rows. Only applied where the map is provably 1:1,
    so the fill is a recovery of known data, never an inference."""
    known = df[[key_col, target_col]].dropna()
    fanout = known.groupby(key_col)[target_col].nunique()
    if len(fanout) and fanout.max() > 1:
        report.setdefault("_repairs", {})[label] = {
            "skipped": "mapping is not 1:1",
            "ambiguous_keys": fanout[fanout > 1].index.tolist()[:10],
        }
        return df
    mapping = dict(zip(known[key_col], known[target_col]))
    gap = df[target_col].isna() & df[key_col].notna()
    filled = df.loc[gap, key_col].map(mapping)
    df.loc[gap, target_col] = filled
    report.setdefault("_repairs", {})[label] = {
        "rows_missing": int(gap.sum()),
        "rows_repaired": int(filled.notna().sum()),
        "rows_still_missing": int(df[target_col].isna().sum()),
    }
    return df


def repair_from_other_table(df, key_col, target_col, mapping, label, report):
    gap = df[target_col].isna() & df[key_col].notna()
    filled = df.loc[gap, key_col].map(mapping)
    df.loc[gap, target_col] = filled
    report.setdefault("_repairs", {})[label] = {
        "rows_missing": int(gap.sum()),
        "rows_repaired": int(filled.notna().sum()),
        "rows_still_missing": int(df[target_col].isna().sum()),
    }
    return df


# --- flagging -------------------------------------------------------------

def flag_rows(df, table, accounts, customers, report):
    """Add boolean quality flags. Nothing is dropped -- flagged rows stay in
    the clean output and are additionally copied to the rejects file, so the
    downstream fraud stage can filter on its own terms."""
    flags = {}

    for spec in schema.TABLES[table]:
        if spec.required and df[spec.name].isna().any():
            col = f"has_invalid_{spec.name}"
            flags[col] = df[spec.name].isna()

    if table == "transactions":
        flags["is_orphan_account"] = ~df["account_id"].isin(accounts["account_id"])
        flags["is_orphan_customer"] = (
            df["customer_id"].notna() & ~df["customer_id"].isin(customers["customer_id"])
        )
    elif table == "accounts":
        flags["is_orphan_customer"] = ~df["customer_id"].isin(customers["customer_id"])

    for name, mask in flags.items():
        df[name] = mask.astype("boolean")

    flag_cols = list(flags)
    if flag_cols:
        df["is_quarantined"] = df[flag_cols].any(axis=1).astype("boolean")
    else:
        df["is_quarantined"] = False

    report["_flags"] = {name: int(mask.sum()) for name, mask in flags.items()}
    report["_flags"]["rows_quarantined"] = int(df["is_quarantined"].sum())
    return df, flag_cols


# --- validation -----------------------------------------------------------

def validate(tables, report):
    """Consistency invariants. These are *checks*, not rewrites: a violation
    means our parsing went wrong, which makes them a regression test on the
    riskiest step (timestamp format resolution)."""
    c, a, t = tables["customers"], tables["accounts"], tables["transactions"]
    checks = {}

    ref = pd.Timestamp("2026-09-19")
    dob = c["date_of_birth"]
    derived_age = ((ref - dob).dt.days / 365.25).floordiv(1)
    comparable = dob.notna() & c["age"].notna()
    checks["age_matches_date_of_birth"] = int(
        (abs(derived_age[comparable] - c["age"][comparable]) > 1).sum()
    )

    ts = t["transaction_timestamp"]
    ok = ts.notna()
    checks["transaction_hour_matches_timestamp"] = int(
        (ts[ok].dt.hour != t["transaction_hour"][ok]).sum()
    )
    checks["is_weekend_matches_timestamp"] = int(
        ((ts[ok].dt.dayofweek >= 5) != t["is_weekend"][ok].astype(bool)).sum()
    )

    fx = t["merchant_country"].notna() & t["is_foreign_transaction"].notna()
    checks["is_foreign_matches_merchant_country"] = int(
        ((t["merchant_country"][fx] != "IN") != t["is_foreign_transaction"][fx].astype(bool)).sum()
    )

    checks["accounts_customer_id_orphans"] = int((~a["customer_id"].isin(c["customer_id"])).sum())
    checks["transactions_account_id_orphans"] = int((~t["account_id"].isin(a["account_id"])).sum())

    owner = dict(zip(a["account_id"], a["customer_id"]))
    both = t["customer_id"].notna() & t["account_id"].isin(owner)
    checks["transaction_customer_matches_account_owner"] = int(
        (t["customer_id"][both] != t["account_id"][both].map(owner)).sum()
    )

    report["_validation"] = {
        "violations": checks,
        "all_passed": all(v == 0 for k, v in checks.items()
                          if k != "transactions_account_id_orphans"),
    }
    return checks


# --- write ----------------------------------------------------------------

def write_outputs(tables, rejects, report):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"cleaned_{name}.csv", index=False, na_rep="",
                  date_format="%Y-%m-%d %H:%M:%S")
    for name, df in rejects.items():
        if len(df):
            df.to_csv(OUT_DIR / f"rejects_{name}.csv", index=False, na_rep="",
                      date_format="%Y-%m-%d %H:%M:%S")
    (OUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str))


def run(verbose: bool = True):
    report, cleaned, rejects = {}, {}, {}

    # Pass 1: per-column cleaning and dedupe, in dependency order.
    conflicts = {}
    for table in ("customers", "accounts", "transactions"):
        raw = load_raw(table)
        df, col_report = clean_table(table, raw)
        df, conflict_rows = deduplicate(df, table, col_report)
        col_report["_rows"] = {"raw": len(raw), "after_dedupe": len(df)}
        cleaned[table], report[table], conflicts[table] = df, col_report, conflict_rows

    # Pass 2: deterministic repairs (need the other tables in hand).
    t = cleaned["transactions"]
    t = repair_from_lookup(t, "merchant_id", "merchant_category",
                           "merchant_category_from_merchant_id", report["transactions"])
    t = repair_from_lookup(t, "merchant_id", "merchant_name",
                           "merchant_name_from_merchant_id", report["transactions"])
    owner = dict(zip(cleaned["accounts"]["account_id"], cleaned["accounts"]["customer_id"]))
    t = repair_from_other_table(t, "account_id", "customer_id", owner,
                                "customer_id_from_account_owner", report["transactions"])
    cleaned["transactions"] = t

    # Pass 3: flag, validate, write.
    for table in ("customers", "accounts", "transactions"):
        cleaned[table], flag_cols = flag_rows(
            cleaned[table], table, cleaned["accounts"], cleaned["customers"], report[table]
        )
        bad = cleaned[table][cleaned[table]["is_quarantined"].fillna(False)].copy()
        if len(bad):
            bad["reject_reason"] = bad[flag_cols].apply(
                lambda r: ";".join(c for c in flag_cols if bool(r[c])), axis=1
            )
        extra = conflicts[table]
        if len(extra):
            extra = extra.copy()
            extra["reject_reason"] = "conflicting_primary_key"
            bad = pd.concat([bad, extra], ignore_index=True)
        rejects[table] = bad

    validate(cleaned, report)
    report["_summary"] = {
        table: {
            "rows": len(cleaned[table]),
            "columns": len(cleaned[table].columns),
            "quarantined": int(len(rejects[table])),
        }
        for table in cleaned
    }
    write_outputs(cleaned, rejects, report)

    if verbose:
        _print_summary(cleaned, rejects, report)
    return cleaned, report


def _print_summary(cleaned, rejects, report):
    print(f"\nOutputs -> {OUT_DIR}\n")
    for table, df in cleaned.items():
        r = report[table]
        print(f"  {table:<14} {r['_rows']['raw']:>5} raw -> {len(df):>5} clean "
              f"({r['_dedupe']['exact_duplicate_rows_dropped']} exact dupes dropped, "
              f"{len(rejects[table])} quarantined)")
    print("\n  Validation:")
    for check, violations in report["_validation"]["violations"].items():
        mark = "ok  " if violations == 0 else "WARN"
        print(f"    [{mark}] {check}: {violations}")
    print()
