"""
normalizer.py — Post-extraction value normalization for the PA extraction pipeline.

Converts raw LLM output strings to the exact canonical formats defined in
PA_Business_Rules.xlsx and the Reference tab. Normalization runs after extraction
and before writing to CSV; format mismatches count as wrong answers in evaluation.

Canonical formats by parameter:
  age                         : ">=N" | "FDA labelled age" | "NA"
  step_therapy_text           : verbatim text | "NA"
  num_steps_brands            : "1", "2", ... | "NA"   (never "0")
  num_steps_generic           : "1", "2", ... | "NA"   (never "0")
  step_phototherapy           : "Yes" | "No" | "NA"
  tb_test                     : "Yes" | "No" | "NA"
  initial_auth_duration       : "N Months" | "NA"
  reauth_duration             : "N Months" | "NA"
  reauth_required             : "Yes" | "No" | "NA"
  reauth_requirements_text    : verbatim text | "NA"
  specialist_types            : "Dermatologist" | "Dermatologist or Rheumatologist" | ... | "NA"
  quantity_limits             : verbatim text | "NA"

All functions are pure, never raise, and default unrecognized values to "NA".
"""

import re
import logging
from typing import Callable

from src.config import FDA_LABELED_AGE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NA detection — shared by all normalizers
# ---------------------------------------------------------------------------

# "N/A" with optional whitespace around slash
_NA_SLASH_PATTERN = re.compile(r"^n\s*/\s*a$", re.IGNORECASE)

# All strings that mean "not present / not applicable"
_NA_SYNONYMS: frozenset[str] = frozenset({
    "na",
    "n/a",
    "n.a.",
    "n.a",
    "none",
    "not applicable",
    "not mentioned",
    "not stated",
    "not specified",
    "unspecified",       # webinar clarification: Unspecified → NA
    "not found",
    "not present",
    "not listed",
    "not required",
    "not available",     # common LLM phrasing for absent fields
    "not documented",    # common LLM phrasing for absent fields
    "not indicated",     # common LLM phrasing for absent fields
    "not addressed",     # common LLM phrasing for absent fields
    "no information",    # common LLM phrasing for absent fields
    "no data",           # common LLM phrasing for absent fields
    "unknown",
    "",
})


def _is_na(s: str) -> bool:
    """Return True if the stripped string represents a missing/NA value."""
    stripped = s.strip()
    return stripped.lower() in _NA_SYNONYMS or bool(_NA_SLASH_PATTERN.match(stripped))


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Duration: "up to N months", "N months", "N mo", "N mos"
_DURATION_NUM_RE = re.compile(
    r"(?:up\s+to\s+)?(\d+)\s*(?:months?|mo\.?|mos\.?)\b",
    re.IGNORECASE,
)
# Duration: "six months", "twelve months", etc.
_DURATION_WORD_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b",
    re.IGNORECASE,
)
_WORD_TO_NUM: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}
# Duration: "1 year", "2 years", "1-year", "one year"
_DURATION_YEAR_RE = re.compile(
    r"(?:up\s+to\s+)?(\d+)\s*[-\s]?years?\b",
    re.IGNORECASE,
)
_DURATION_YEAR_WORD_RE = re.compile(
    r"\b(one|two|three|four|five)\s+years?\b",
    re.IGNORECASE,
)
# Duration: "16 weeks", "8 weeks" — converted to months only when result >= 2
_DURATION_WEEK_RE = re.compile(
    r"(?:up\s+to\s+)?(\d+)\s*weeks?\b",
    re.IGNORECASE,
)

_DURATION_DAY_RE = re.compile(r"(?:up\s+to\s+)?(\d+)\s*days?\b", re.IGNORECASE)

# Age: ">=18", "≥18", "> 18", ">= 18"
_AGE_GTE_RE = re.compile(r"(?:≥|>=?)\s*(\d+)")
# Age: "18 years of age or older", "18 years or older", "18 years and older"
_AGE_YEARS_OLDER_RE = re.compile(
    r"(\d+)\s+years?\s+(?:of\s+age\s+)?(?:and\s+|or\s+)?older",
    re.IGNORECASE,
)
# Age: bare "N years" as last resort
_AGE_BARE_YEARS_RE = re.compile(r"(\d+)\s+years?\b", re.IGNORECASE)
# FDA labelled age — handles "FDA labeled age", "FDA labelled age",
# "age within FDA labeling", "age is within FDA labeling"
_AGE_FDA_RE = re.compile(
    r"fda\s+label+ed\s+age"          # "FDA labelled age" / "FDA labeled age"
    r"|age.*within.*fda"              # "age within FDA labeling"
    r"|fda.*label.*age",              # "FDA label ...age"
    re.IGNORECASE,
)

# Yes synonyms (lowercase)
_YES_SET: frozenset[str] = frozenset({"yes", "y", "true", "1", "required"})
# No synonyms (lowercase)
_NO_SET: frozenset[str] = frozenset({"no", "n", "false", "not required"})


# ---------------------------------------------------------------------------
# Normalizer functions
# ---------------------------------------------------------------------------


def normalize_yes_no_na(raw: str) -> str:
    """Normalize a Yes/No/NA field to canonical 'Yes', 'No', or 'NA'.

    Used for: step_phototherapy, tb_test, reauth_required.
    Business rules map:
      Yes  — explicitly required
      No   — explicitly not required
      NA   — not mentioned / not applicable / policy lists no criteria
    """
    if _is_na(raw):
        return "NA"

    s = raw.strip().lower()

    if s in _YES_SET:
        return "Yes"
    if s in _NO_SET:
        return "No"

    # Partial-match fallback for verbose responses like "Yes, required"
    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"

    logger.warning("normalize_yes_no_na: unrecognized value %r → NA", raw)
    return "NA"


def normalize_duration(raw: str) -> str:
    """Normalize authorization duration to canonical 'N Months' format.

    Handles: '3 months', '3 mo', 'up to 12 months', 'twelve months',
    '12-months', '1 year', '2 years', '16 weeks',
    'Unspecified' (webinar: Unspecified → NA).
    Returns 'NA' if no duration can be parsed or the value is ambiguous.

    Canonical format: integer + space + 'Months' (capital M).
    Examples: '3 Months', '6 Months', '12 Months'.

    Week conversion: when result rounds to ≥ 1 month (i.e. ≥ 4 weeks).
    Some policies have genuine 4-week initial approval windows (e.g. induction
    phases). These are preserved as '1 Month', not discarded as trial windows.
    """
    if _is_na(raw):
        return "NA"

    s = raw.strip()

    # 1. Numeric months: "3 months", "up to 12 months", "12-months"
    m = _DURATION_NUM_RE.search(s)
    if m:
        return f"{m.group(1)} Months"

    # 2. Word months: "six months", "twelve months"
    m = _DURATION_WORD_RE.search(s)
    if m:
        num = _WORD_TO_NUM.get(m.group(1).lower())
        if num:
            return f"{num} Months"

    # 3. Numeric years: "1 year", "2 years", "1-year"
    m = _DURATION_YEAR_RE.search(s)
    if m:
        return f"{int(m.group(1)) * 12} Months"

    # 4. Word years: "one year", "two years"
    m = _DURATION_YEAR_WORD_RE.search(s)
    if m:
        num = _WORD_TO_NUM.get(m.group(1).lower())
        if num:
            return f"{int(num) * 12} Months"

    # 5. Weeks → months (when result rounds to ≥ 1 month)
    #    4 weeks → ~1 month → '1 Month'  (e.g. Alaska Medicaid induction approval)
    #    8 weeks → 2 months → '2 Months'
    #   16 weeks → 4 months → '4 Months'
    m = _DURATION_WEEK_RE.search(s)
    if m:
        months = round(int(m.group(1)) / 4.33)
        if months >= 1:
            return f"{months} Months"
        # Fewer than 1 month (< ~4 weeks) is a genuine edge case — treat as NA
        logger.warning(
            "normalize_duration: %r is too short (%d weeks ≈ %d month) → NA",
            raw, int(m.group(1)), months,
        )
        return "NA"
    
    m = _DURATION_DAY_RE.search(s)
    if m:
        months = round(int(m.group(1)) / 30.4)
        if months >= 1:
            return f"{months} Months"
        logger.warning("normalize_duration: %r is too short (%d days) → NA", raw, int(m.group(1)))
        return "NA"
    

    logger.warning("normalize_duration: cannot parse %r → NA", raw)
    return "NA"


def normalize_age(raw: str) -> str:
    """Normalize age threshold to canonical '>=N' or 'FDA labelled age'.

    Handles: '>=18', '≥18', '≥ 18', '18 years of age or older',
    'adult patients', 'FDA labeled age', 'FDA labelled age',
    'age within FDA labeling'.

    Canonical format:
      ">=N"              when a specific numeric threshold is stated
      "FDA labelled age" when policy defers to FDA label (British 'labelled')
      "NA"               when age is not mentioned

    Note on spelling: PA_Business_Rules.xlsx uses 'labelled' (double L).
    """
    if _is_na(raw):
        return "NA"

    s = raw.strip()

    # FDA labelled age check first (before numeric checks)
    if _AGE_FDA_RE.search(s):
        return "FDA labelled age"

    # "adult" / "adults" without a specific number → >=18
    if re.search(r"\badults?\b", s, re.IGNORECASE) and not re.search(r"\d", s):
        return ">=18"

    # Explicit >=, ≥, or > pattern: ">=18", "≥ 18", "> 18"
    m = _AGE_GTE_RE.search(s)
    if m:
        return f">={m.group(1)}"

    # "N years of age or older", "N years or older"
    m = _AGE_YEARS_OLDER_RE.search(s)
    if m:
        return f">={m.group(1)}"

    # Bare "N years" as last resort
    m = _AGE_BARE_YEARS_RE.search(s)
    if m:
        return f">={m.group(1)}"

    logger.warning("normalize_age: cannot parse %r → keeping raw", raw)
    return s


def _parse_age_min(age_str: str) -> int | None:
    """Extract the minimum numeric age from strings like '>=18', '>=6', '>=4'.

    Returns None if no number is present (e.g. 'NA', 'FDA labelled age').
    Used by normalize_record for cross-brand age validation.
    """
    m = re.search(r"(\d+)", str(age_str))
    return int(m.group(1)) if m else None


def normalize_step_count(raw: str) -> str:
    """Normalize step count to a positive integer string or 'NA'.

    Business rule: never output '0' — zero required steps means NA.
    Handles: '1', '2', '1.0', '0', 'NA', 'N/A', '1 branded step'.
    Falls back to extracting the first digit from descriptive text.
    """
    if _is_na(raw):
        return "NA"

    s = raw.strip()

    # Direct numeric parse
    try:
        n = int(float(s))
        if n <= 0:
            return "NA"    # 0 → no steps required → NA per business rules
        return str(n)
    except (ValueError, TypeError):
        pass

    # Fallback: extract first positive integer from descriptive text
    # e.g., "1 branded step" → "1", "2 steps required" → "2"
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        if n > 0:
            return str(n)

    logger.warning("normalize_step_count: cannot parse %r → NA", raw)
    return "NA"


def normalize_free_text(raw: str) -> str:
    """Minimal normalization for verbatim text fields.

    Preserves original policy language including internal newlines and
    bullet point structure. Only strips leading/trailing whitespace and
    converts NA variants to 'NA'.

    Used for: step_therapy_text, reauth_requirements_text, quantity_limits.
    Internal whitespace is intentionally NOT collapsed so that multi-line
    quantity limit tables and bulleted reauth criteria retain their structure.
    """
    if _is_na(raw):
        return "NA"
    return raw.strip()


def normalize_specialist_types(raw: str) -> str:
    """Normalize specialist type names to title-case canonical form.

    Handles: comma-separated, 'or'-separated, mixed case, NA variants.
    Both 'Dermatologist, Rheumatologist' and 'Dermatologist or Rheumatologist'
    normalize to the same canonical 'Dermatologist or Rheumatologist'.

    Examples:
      'dermatologist'                    → 'Dermatologist'
      'Dermatologist, Rheumatologist'    → 'Dermatologist or Rheumatologist'
      'dermatologist or rheumatologist'  → 'Dermatologist or Rheumatologist'
      'GASTROENTEROLOGIST'               → 'Gastroenterologist'
    """
    if _is_na(raw):
        return "NA"

    s = raw.strip()

    # Split on BOTH ', ' and ' or ' so comma-form and or-form converge
    parts = re.split(r"\s*,\s*|\s+or\s+", s, flags=re.IGNORECASE)
    titled = " or ".join(part.strip().title() for part in parts if part.strip())
    return titled


# ---------------------------------------------------------------------------
# Parameter → normalizer mapping
# ---------------------------------------------------------------------------

NORMALIZERS: dict[str, Callable[[str], str]] = {
    "age":                      normalize_age,
    "step_therapy_text":        normalize_free_text,
    "num_steps_brands":         normalize_step_count,
    "num_steps_generic":        normalize_step_count,
    "step_phototherapy":        normalize_yes_no_na,
    "tb_test":                  normalize_yes_no_na,
    "initial_auth_duration":    normalize_duration,
    "reauth_duration":          normalize_duration,
    "reauth_required":          normalize_yes_no_na,
    "reauth_requirements_text": normalize_free_text,
    "specialist_types":         normalize_specialist_types,
    "quantity_limits":          normalize_free_text,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_record(record: dict[str, str], brand: str | None = None) -> dict[str, str]:
    """Apply canonical normalization to all 12 fields of an extraction record.

    Args:
        record: Raw extraction dict with EXPECTED_KEYS from extractor.py.
                Keys not in NORMALIZERS fall back to normalize_free_text.
        brand:  Uppercase brand name (e.g. "AMJEVITA"). When provided, the
                extracted age is cross-checked against FDA_LABELED_AGE.
                If the extracted age is more permissive (lower) than the FDA
                minimum, it is overridden — this catches cross-contamination
                cases where the LLM picks up an age from a different drug in
                the same multi-drug document.

    Returns:
        New dict with the same keys and normalized values.
        On any per-field error, the raw value is preserved and logged.
        Never raises.
    """
    normalized: dict[str, str] = {}
    for key, raw_value in record.items():
        normalizer = NORMALIZERS.get(key, normalize_free_text)
        try:
            normalized[key] = normalizer(raw_value)
        except Exception as exc:
            logger.error(
                "Normalization error for key=%r value=%r: %s — keeping raw",
                key, raw_value, exc,
            )
            normalized[key] = raw_value

    # Truncated quantity_limits detection.
    # When the LLM starts writing a QL value but stops mid-sentence, the result
    # ends with a colon and trailing whitespace (e.g. "quantity limit: ").
    # This is unambiguously a truncation artifact — set to NA and warn.
    ql = normalized.get("quantity_limits", "NA")
    if ql != "NA" and re.search(r":\s*$", ql):
        logger.warning(
            "normalize_record: quantity_limits appears truncated (ends with ':') — setting NA. "
            "Value was: %r",
            ql,
        )
        normalized["quantity_limits"] = "NA"

    # Cross-field override: reauth_required must be "Yes" if either
    # reauth_duration or reauth_requirements_text is non-NA.
    # Business rule: evidence of reauth content proves reauth is required,
    # even if the LLM did not explicitly state "Yes" for reauth_required.
    reauth_dur  = normalized.get("reauth_duration", "NA")
    reauth_text = normalized.get("reauth_requirements_text", "NA")
    if reauth_dur != "NA" or reauth_text != "NA":
        if normalized.get("reauth_required") != "Yes":
            logger.info(
                "normalize_record: overriding reauth_required → 'Yes' "
                "(reauth_duration=%r, reauth_requirements_text=%r)",
                reauth_dur,
                reauth_text,
            )
            normalized["reauth_required"] = "Yes"

    # Age cross-validation: if the extracted age is MORE PERMISSIVE (numerically
    # lower) than the FDA-approved minimum for this brand, the LLM almost certainly
    # picked up an age from a different drug in the same multi-drug document.
    # Insurers cannot approve beyond the FDA label, so a lower-than-FDA age is
    # always a cross-contamination error, never a genuine policy value.
    #
    # Examples caught:
    #   AMJEVITA (FDA >=18) extracted >=6  → >=6 was Stelara's age in the doc
    #   CIMZIA   (FDA >=18) extracted >=12 → >=12 was from another drug's section
    #
    # Does NOT fire when extracted age == 'FDA labelled age', 'NA', or when
    # the extracted age is more restrictive than FDA (e.g. FDA >=6, policy >=18
    # — that is a valid, more-restrictive payer requirement).
    if brand:
        fda_age = FDA_LABELED_AGE.get(brand.upper())
        if fda_age:
            extracted_age = normalized.get("age", "NA")
            if extracted_age not in ("NA", "FDA labelled age"):
                fda_min = _parse_age_min(fda_age)
                ext_min = _parse_age_min(extracted_age)
                if fda_min is not None and ext_min is not None and ext_min < fda_min:
                    logger.warning(
                        "normalize_record: age '%s' is more permissive than FDA "
                        "label '%s' for brand %s — overriding (likely cross-"
                        "contamination from another drug in the same document)",
                        extracted_age, fda_age, brand.upper(),
                    )
                    normalized["age"] = fda_age

    return normalized