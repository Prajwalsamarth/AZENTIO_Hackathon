"""Pure, deterministic type-coercion primitives.

Every function takes a pd.Series of raw strings plus its ColumnSpec and returns
(cleaned_series, issues_dict). No I/O, no globals, no randomness -- the same
input always produces the same output, which is what makes the report an audit
trail rather than a summary.
"""

import re
import pandas as pd

from schema import ColumnSpec

# --- shared regexes -------------------------------------------------------
_WS = re.compile(r"\s+")
_NON_ALNUM_RUN = re.compile(r"[^A-Z0-9]+")
_CURRENCY = re.compile(r"(?i)^\s*(?:INR|RS\.?|₹|\$)\s*|\s*(?:INR|RS\.?|₹)\s*$")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

TRUE_TOKENS = {"1", "Y", "YES", "TRUE", "T"}
FALSE_TOKENS = {"0", "N", "NO", "FALSE", "F"}

# Formats whose field order is ambiguous when both components are <= 12.
# Parsing each such row under the opposite order too lets us *count* how many
# rows the chosen order actually decides, instead of hiding the assumption.
_AMBIGUOUS_SWAP = {
    "%d/%m/%Y %H:%M:%S": "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M": "%m/%d/%Y %H:%M",
    "%d/%m/%Y": "%m/%d/%Y",
    "%m-%d-%Y %H:%M:%S": "%d-%m-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M": "%d-%m-%Y %H:%M",
    "%m-%d-%Y": "%d-%m-%Y",
}


def to_null(s: pd.Series, spec: ColumnSpec):
    """Strip, then map the spec's null tokens (case-insensitive) to pd.NA.

    Runs first for every column, so no downstream primitive ever sees a
    sentinel string like 'NOT_AVAILABLE'.
    """
    out = s.astype("string").str.strip()
    tokens = {t.upper() for t in spec.null_tokens}
    mask = out.str.upper().isin(tokens) | out.isna()
    return out.mask(mask, pd.NA), {"nulls_from_tokens": int(mask.sum() - s.isna().sum())}


def sanitize_text(s: pd.Series) -> pd.Series:
    """Seam for the prompt-injection filter -- intentionally a no-op today.

    The shipped CSVs contain no free-text notes column and no injection
    payloads (verified: max value length across all three files is 36 chars).
    Wiring the call site now means the filter drops in as a single function
    body later, with no restructuring of the pipeline.
    """
    return s


def clean_string(s: pd.Series, spec: ColumnSpec):
    out = s.str.replace(_WS, " ", regex=True).str.strip()
    if spec.sanitize:
        out = sanitize_text(out)
    return out.astype("string"), {}


def _fold(value: str) -> str:
    """Canonical category form: upper, then any run of non-alphanumerics -> '_'.

    'Salaried - Banking' -> 'SALARIED_BANKING'; '  P2P Transfer ' ->
    'P2P_TRANSFER'; 'internet banking' -> 'INTERNET_BANKING'. One rule
    collapses every casing/spacing/separator variant in the data.
    """
    return _NON_ALNUM_RUN.sub("_", value.strip().upper()).strip("_")


def clean_category(s: pd.Series, spec: ColumnSpec):
    folded = s.map(_fold, na_action="ignore").astype("string")
    if spec.aliases:
        aliases = {_fold(k): v for k, v in spec.aliases.items()}
        folded = folded.replace(aliases)
    allowed = set(spec.categories)
    unknown = folded.notna() & ~folded.isin(allowed)
    issues = {"variants_collapsed": int(s.dropna().nunique() - folded.dropna().nunique())}
    if unknown.any():
        # Out-of-vocabulary values become NA *loudly* -- the offending strings
        # are recorded so an unseen variant surfaces instead of slipping past.
        issues["out_of_vocabulary"] = (
            folded[unknown].value_counts().to_dict()
        )
        folded = folded.mask(unknown, pd.NA)
    return folded.astype("string"), issues


def clean_numeric(s: pd.Series, spec: ColumnSpec):
    issues = {}
    work = s
    if spec.strip_currency:
        stripped = work.str.replace(_CURRENCY, "", regex=True)
        stripped = stripped.str.replace(_THOUSANDS, "", regex=True).str.strip()
        changed = int((stripped.fillna("") != work.fillna("")).sum())
        if changed:
            issues["currency_formatting_stripped"] = changed
        work = stripped

    parsed = pd.to_numeric(work, errors="coerce")
    unparseable = work.notna() & parsed.isna()
    if unparseable.any():
        issues["unparseable"] = int(unparseable.sum())
        issues["unparseable_examples"] = sorted(set(work[unparseable]))[:10]

    if spec.min_value is not None or spec.max_value is not None:
        lo = spec.min_value if spec.min_value is not None else float("-inf")
        hi = spec.max_value if spec.max_value is not None else float("inf")
        out_of_range = parsed.notna() & ((parsed < lo) | (parsed > hi))
        if out_of_range.any():
            issues["out_of_range"] = int(out_of_range.sum())
            issues["out_of_range_examples"] = sorted(set(parsed[out_of_range]))[:10]
            parsed = parsed.mask(out_of_range)

    if spec.dtype == "int":
        # Round-trip through float first: Int64 cannot take '15.0' directly.
        return parsed.round().astype("Int64"), issues
    return parsed.astype("Float64"), issues


def clean_bool(s: pd.Series, spec: ColumnSpec):
    upper = s.str.upper()
    true_mask = upper.isin(TRUE_TOKENS)
    false_mask = upper.isin(FALSE_TOKENS)
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[true_mask] = True
    out[false_mask] = False
    issues = {"encodings_seen": sorted(set(s.dropna().unique()))}
    unknown = s.notna() & ~true_mask & ~false_mask
    if unknown.any():
        issues["unparseable"] = int(unknown.sum())
        issues["unparseable_examples"] = sorted(set(s[unknown]))[:10]
    return out, issues


def clean_datetime(s: pd.Series, spec: ColumnSpec):
    """Parse with explicit formats in order, coalescing successes.

    pandas' inference is deliberately avoided: it decides day-vs-month order
    per-row, which would silently mis-date every slash-formatted row where the
    day happens to be <= 12.
    """
    issues = {}
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    per_format = {}
    ambiguous = 0

    for fmt in spec.datetime_formats:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        hit = parsed.notna()
        if not hit.any():
            continue
        per_format[fmt] = int(hit.sum())
        out.loc[parsed.index[hit]] = parsed[hit]

        swap = _AMBIGUOUS_SWAP.get(fmt)
        if swap:
            alt = pd.to_datetime(s[todo][hit], format=swap, errors="coerce")
            # Both orders parse and disagree -> the chosen order decided this row.
            ambiguous += int((alt.notna() & (alt != parsed[hit])).sum())

    unparsed = s.notna() & out.isna()
    if per_format:
        issues["parsed_by_format"] = per_format
    if ambiguous:
        issues["ambiguous_day_month_decided_by_format_order"] = ambiguous
    if unparsed.any():
        issues["unparseable"] = int(unparsed.sum())
        issues["unparseable_examples"] = sorted(set(s[unparsed]))[:10]

    if spec.dtype == "date":
        out = out.dt.normalize()
    return out, issues


DISPATCH = {
    "string": clean_string,
    "category": clean_category,
    "int": clean_numeric,
    "float": clean_numeric,
    "bool": clean_bool,
    "date": clean_datetime,
    "datetime": clean_datetime,
}


def clean_column(raw: pd.Series, spec: ColumnSpec):
    """Null-normalize, then coerce via the spec's dtype handler."""
    nulled, issues = to_null(raw, spec)
    nulls_before = int(nulled.isna().sum())
    cleaned, more = DISPATCH[spec.dtype](nulled, spec)
    issues.update(more)
    issues["nulls_before"] = nulls_before
    issues["nulls_after"] = int(cleaned.isna().sum())
    issues = {k: v for k, v in issues.items() if v not in (0, [], {}, None)}
    return cleaned, issues
