"""
page_filter.py — Filter a PDF's pages dict to those relevant for extracting
plaque psoriasis prior-authorization criteria for a specific brand.

Designed for high recall: a missed relevant page is worse than an irrelevant
included page. Callers should expect some noise in the returned pages.
"""

import logging
import re

from src.config import BRAND_TO_GENERICS, INDICATION_KEYWORDS, EXCLUDE_INDICATIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Pages matching any of these are always included, even without a brand/indication match.
UNIVERSAL_MARKERS: list[str] = [
    "universal criteria",
    "general criteria",
    "applies to all",
    "applicable for all",
    "for all indications",
    "all biologics",
    "all drugs",
    "initial authorization request must include",
    # Severity bypass language — these pages determine whether step therapy is required
    # at all (0-step paths). Critical for correctness on num_steps_brands and
    # num_steps_generic. Including them ensures the LLM sees the full OR logic.
    "10% of body surface area",
    "10% bsa",
    "crucial body areas",
    "hands, feet, face",             # "hands, feet, face, neck, scalp" severity criterion
    "intractable pruritus",          # severe PsO indicator
    "serious emotional consequences", # severe PsO indicator
    "step therapy not required",     # blanket exception language
    "step therapy is not required",
    "not required for biologic",     # ¥-note style: "step therapy not required for biologic patients"
    "not required for established",  # variant: "not required for established biologic patients"
]

# Structural section headers that always contain duration/authorization data.
# These pages are included regardless of brand mention because they hold
# universal approval periods that apply to all drugs in the policy.
STRUCTURAL_SECTION_MARKERS: list[str] = [
    "approval length",
    "approval duration",
    "authorization period",
    "authorization duration",
    "length of authorization",
    "length of approval",
    "duration of approval",
    "quantity limits and duration",
    "quantity limits and authorization",
    "limitations of authorization",
    "approval limitations",
    "reauthorization duration",
    "renewal duration",
    "continuation of therapy",          # always include Continued Therapy sections
    "continued therapy",
    "re-authorization",
    "reauthorization criteria",
    "renewal criteria",
    "renewal evaluation",
    "renewal approval",
    # Step Table structure markers — critical for multi-drug PBM policies.
    # These pages define which tier each drug is at (Step 1a, 3a, etc.)
    # and must be included to correctly identify biologic prerequisites.
    "step table",
    "step 1a",
    "step 1b",
    "step 3a",
    "step 3b",
    "step 3c",
    "directed to one step",
    "directed to two step",
    "directed to three step",
    "non-preferred (directed to",       # Step Table header variants
    "preferred specialty management",    # PBM specialty management policy
]

BRAND_KEYWORDS = {
    "TREMFYA":    ["tremfya", "guselkumab"],
    "STELARA":    ["stelara", "ustekinumab", "stelera"],
    "SKYRIZI":    ["skyrizi", "risankizumab"],
    "COSENTYX":   ["cosentyx", "secukinumab"],
    "ENBREL":     ["enbrel", "etanercept"],
    "AMJEVITA":   ["amjevita", "adalimumab"],
    "OTEZLA":     ["otezla", "apremilast"],
    "SILIQ":      ["siliq", "brodalumab"],
    "BIMZELX":    ["bimzelx", "bimekizumab"],
    "ILUMYA":     ["ilumya", "tildrakizumab"],
    "CIMZIA":     ["cimzia", "certolizumab"],
    "REMICADE":   ["remicade", "infliximab"],
    "YESINTEK":   ["yesintek", "ustekinumab"],  # biosimilar
    "OTULFI":     ["otulfi", "ustekinumab"],    # biosimilar
    "ACITRETIN":  ["acitretin"],
}

# Competing brand terms — used to detect cross-contamination pages.
_ALL_BRAND_TERMS: dict[str, list[str]] = BRAND_KEYWORDS

# Keywords at or below this length, or fully uppercase, get \b word-boundary
# matching to avoid substring false positives (e.g. "TB" inside "STBs").
SHORT_KEYWORD_THRESHOLD: int = 4

# ── Noise-page detection thresholds ──────────────────────────────────────────
CITATION_THRESHOLD: int = 8
CRITERIA_THRESHOLD: int = 2
EXCLUDE_DOMINATION_THRESHOLD: int = 3
CODE_THRESHOLD: int = 10

# Plaque-specific terms used in the other-indication check.
PLAQUE_SPECIFIC_TERMS: list[str] = [
    "plaque psoriasis",
    "moderate to severe plaque",
    "chronic plaque",
    "chronic severe plaque",
    "PsO",
]

# PA criteria phrases — if a page has these, it is NOT administrative noise
CRITERIA_PHRASES: list[str] = [
    "medically necessary",
    "prior authorization",
    "initial approval",
    "must have",
    "tried and failed",
    "inadequate response",
    "authorization criteria",
    "coverage criteria",
    "approval criteria",
    "continuation of therapy",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _needs_word_boundary(keyword: str) -> bool:
    """Short or all-uppercase keywords need \\b boundaries to avoid substring false positives."""
    return len(keyword) <= SHORT_KEYWORD_THRESHOLD or keyword.isupper()


def _build_pattern(keywords: list[str]) -> re.Pattern:
    """
    Compile a case-insensitive OR-pattern from a list of keywords.

    Keywords that are short or all-uppercase are wrapped in \\b word boundaries;
    all others are matched as plain substrings after regex escaping.
    """
    parts: list[str] = []
    for kw in keywords:
        escaped = re.escape(kw)
        if _needs_word_boundary(kw):
            parts.append(rf"\b{escaped}\b")
        else:
            parts.append(escaped)
    return re.compile("|".join(parts), re.IGNORECASE)


def _page_has_match(page_text: str, pattern: re.Pattern) -> bool:
    """Return True if the pattern matches anywhere in the page text."""
    return bool(pattern.search(page_text))


# Module-level compiled patterns for noise detection (compiled once at import).
_CITATION_PATTERN = re.compile(
    r"et\s+al[.,)]"          # "et al." / "et al," / "et al)"
    r"|\(\s*\d{4}\s*\)"      # (2022)
    r"|doi\s*:"              # DOI links
    r"|\d{4};\s*\d+\s*[:(]" # journal vol/year format like "2022;45("
    r"|^\s*\d{1,3}\.\s+[A-Z][a-z]",  # numbered reference list "1. Smith..."
    re.MULTILINE | re.IGNORECASE,
)
_CRITERIA_PATTERN = re.compile(
    "|".join(re.escape(p) for p in CRITERIA_PHRASES),
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(
    r"\bJ\d{4}\b"               # HCPCS J-codes (e.g. J3357)
    r"|\b[A-Z]\d{2}\.\d[\dA-Z]*\b"  # ICD-10 (e.g. L40.0, K50.00)
    r"|\b9\d{4}\b"              # HCPCS Q-codes (e.g. Q5098)
)


def _is_noise_page(
    page_text: str,
    exclude_pattern: re.Pattern,
    plaque_pattern: re.Pattern,
) -> bool:
    """Return True if the page is noise that should be dropped."""
    # If the page has PA criteria language, never drop it regardless of other signals.
    criteria_count = len(_CRITERIA_PATTERN.findall(page_text))
    if criteria_count > CRITERIA_THRESHOLD:
        return False

    # Signal 1: References page (many citations, few criteria phrases).
    citation_count = len(_CITATION_PATTERN.findall(page_text))
    if citation_count >= CITATION_THRESHOLD:
        return True

    # Signal 2: Other-indication-only (EXCLUDE_INDICATIONS dominate, no plaque).
    exclude_count = len(exclude_pattern.findall(page_text))
    plaque_count = len(plaque_pattern.findall(page_text))
    if exclude_count >= EXCLUDE_DOMINATION_THRESHOLD and plaque_count == 0:
        return True

    # Signal 3: Administrative/coding page (many medical codes, few criteria).
    code_count = len(_CODE_PATTERN.findall(page_text))
    if code_count >= CODE_THRESHOLD:
        return True

    return False


def _is_cross_contamination_page(
    page_text: str,
    brand_pattern: re.Pattern,
    brand: str,
) -> bool:
    """Return True if a page mentions ONLY a competing brand and not the queried brand.

    Fixes Pattern R (cross-contamination): when a multi-drug document passes pages
    for Brand A into the LLM call for Brand B, the model may extract Brand A's
    criteria. This detects pages where only a competing brand is prominently named.
    """
    # If the queried brand appears anywhere, the page is not contaminated.
    if _page_has_match(page_text, brand_pattern):
        return False

    # Count mentions of every OTHER brand to detect competitor dominance.
    text_lower = page_text.lower()
    for other_brand, other_terms in _ALL_BRAND_TERMS.items():
        if other_brand == brand.upper():
            continue
        mention_count = sum(
            len(re.findall(re.escape(term), text_lower))
            for term in other_terms
            # Exclude shared generics that appear in both brands (adalimumab, ustekinumab)
            if term.lower() not in ("adalimumab", "ustekinumab", "etanercept")
        )
        if mention_count >= 3:
            return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_pages(
    pages: dict[int, str],
    brand: str,
    buffer: int = 2,
    always_include_first: int = 5,
) -> dict[int, str]:
    """
    Filter a PDF's pages dict to those relevant for extracting plaque psoriasis
    criteria for the given brand.

    A page is a "seed" if it matches both the brand pattern and the indication
    pattern, or if it matches universal-criteria markers (including severity bypass
    language), or if it matches structural section markers (APPROVAL LENGTH,
    LIMITATIONS, Step Table headers, etc.) that contain duration/reauth/step-tier
    data. The returned set includes all seed pages plus a buffer of adjacent pages
    on each side, and always the first `always_include_first` pages (covers
    table-of-contents and scope statements that rarely mention the brand by name).

    always_include_first is increased from 3 to 5 to ensure broader coverage of
    document preambles that often contain universal/class-level criteria.

    Args:
        pages: Mapping of 1-indexed page number to extracted text.
        brand: Uppercase brand name (e.g. "STELARA").
        buffer: Number of pages to include before and after each seed page.
        always_include_first: Always include pages 1 through this value.

    Returns:
        Filtered dict with the same page-keyed structure, sorted by page number.
        Returns an empty dict if `pages` is empty.
    """
    total = len(pages)
    logger.info("filter_pages called: brand=%s, total_pages=%d", brand, total)

    if not pages:
        return {}

    # Build brand term list: brand name + all known generics.
    generics = BRAND_TO_GENERICS.get(brand)
    if generics is None:
        logger.warning(
            "Brand %r not found in BRAND_TO_GENERICS; searching by brand name only", brand
        )
        brand_terms = [brand]
    else:
        brand_terms = [brand] + generics

    brand_pattern = _build_pattern(brand_terms)
    indication_pattern = _build_pattern(INDICATION_KEYWORDS)
    universal_pattern = _build_pattern(UNIVERSAL_MARKERS)
    structural_pattern = _build_pattern(STRUCTURAL_SECTION_MARKERS)
    exclude_pattern = _build_pattern(EXCLUDE_INDICATIONS)
    plaque_pattern = _build_pattern(PLAQUE_SPECIFIC_TERMS)

    # Identify seed pages using four independent conditions.
    #
    # 1. is_relevant:      brand keyword AND indication keyword on the same page.
    # 2. is_universal:     universal-criteria marker (applies-to-all-drugs,
    #                      severity bypass, step-therapy-not-required language).
    # 3. is_criteria_page: indication keyword AND PA criteria language, even without
    #                      a brand name. Catches class-level and decision-tree policies.
    # 4. is_structural:    structural section headers (APPROVAL LENGTH, Step Table,
    #                      etc.) These always contain duration/reauth/step-tier data.
    all_page_nums = sorted(pages.keys())
    page_num_set = set(all_page_nums)

    seed_pages: list[int] = []
    contaminated_pages: set[int] = set()

    for page_num in all_page_nums:
        text = pages[page_num]
        is_relevant = _page_has_match(text, brand_pattern) and _page_has_match(
            text, indication_pattern
        )
        is_universal = _page_has_match(text, universal_pattern)
        is_criteria_page = _page_has_match(text, indication_pattern) and _page_has_match(
            text, _CRITERIA_PATTERN
        )
        is_structural = _page_has_match(text, structural_pattern)

        if is_relevant or is_universal or is_criteria_page or is_structural:
            seed_pages.append(page_num)

            # Check for cross-contamination: flag pages where only a competitor
            # brand appears (not the queried brand) even though the page is a seed.
            if not is_relevant and _is_cross_contamination_page(text, brand_pattern, brand):
                contaminated_pages.add(page_num)
                logger.debug(
                    "Cross-contamination flag on page %d for brand=%s "
                    "(structural/criteria match but competing brand dominates; page kept)",
                    page_num, brand,
                )

    # Build include set: seeds + buffer expansion + always_include_first pages.
    # Buffer: back 1 page, forward 4 pages (asymmetric — criteria often continue
    # forward). This is preserved from the original design.
    include_set: set[int] = set()

    for seed in seed_pages:
        for p in range(seed - 1, seed + 5):
            if p in page_num_set:
                include_set.add(p)

    # always_include_first increased to 5 (from 3) to capture more preamble context.
    for p in range(1, always_include_first + 1):
        if p in page_num_set:
            include_set.add(p)

    # Post-filter: remove noise pages. Pages in the always_include_first range are exempt.
    exempt_pages = {p for p in range(1, always_include_first + 1) if p in page_num_set}
    noise_removed: list[int] = []
    clean_include_set: set[int] = set()
    for p in include_set:
        if p in exempt_pages:
            clean_include_set.add(p)
        elif _is_noise_page(pages[p], exclude_pattern, plaque_pattern):
            noise_removed.append(p)
        else:
            clean_include_set.add(p)

    include_set = clean_include_set

    result = {p: pages[p] for p in sorted(include_set)}

    seed_str = ", ".join(str(p) for p in seed_pages) if seed_pages else "none"
    noise_str = f"; noise-removed: {sorted(noise_removed)}" if noise_removed else ""
    contam_str = f"; cross-contamination-flagged: {sorted(contaminated_pages)}" if contaminated_pages else ""
    logger.info(
        "Filter result for %s: kept %d of %d pages "
        "(seed pages: %s; brand+indication|universal|criteria|structural seeds; "
        "buffer back 1 fwd 4 + first %d added%s%s)",
        brand, len(result), total, seed_str, always_include_first, noise_str, contam_str,
    )

    return result


def filter_pages_for_parameter(
    base_pages: dict[int, str],
    parameter: str,
    param_keywords: list[str],
    buffer: int = 1,
) -> dict[int, str]:
    """Stage 2 filter: narrow Stage 1 base pages using parameter-specific keywords.

    Searches only within base_pages (the Stage 1 output). For each page in
    base_pages that matches any param_keywords, includes that page plus a small
    buffer of adjacent pages (also constrained to base_pages).

    For duration parameters (initial_auth_duration, reauth_duration), the keyword
    list is automatically extended with structural section terms to ensure
    APPROVAL LENGTH, QUANTITY LIMITS AND DURATION, and CONTINUATION OF THERAPY
    pages are always included.

    For step_therapy parameters, the keyword list is automatically extended with
    severity bypass markers and Step Table markers to ensure 0-step path language
    is always included even when it doesn't appear near the drug name.

    Fallback: if no pages match param_keywords, returns base_pages unchanged so
    the caller always has context to work with.

    Args:
        base_pages: Stage 1 filter output — pages already filtered for brand +
            indication.
        parameter: Parameter name string used only for logging (e.g. "tb_test").
        param_keywords: List of keywords specific to this parameter.
        buffer: Pages to include before and after each matching page.
            Default 1 (smaller than Stage 1 buffer since we are already in a
            focused page set).

    Returns:
        Narrowed dict of pages. Always a subset of base_pages. Never empty
        (falls back to base_pages if no match).
    """
    if not base_pages or not param_keywords:
        return base_pages

    logger.info(
        "Stage 2 filter: parameter=%s, searching %d base pages",
        parameter, len(base_pages),
    )

    # For duration-related parameters, extend the keyword list with structural
    # section markers to capture APPROVAL LENGTH, QUANTITY LIMITS AND DURATION,
    # CONTINUATION OF THERAPY sections.
    effective_keywords = list(param_keywords)
    if parameter in ("initial_auth_duration", "reauth_duration", "duration"):
        effective_keywords.extend(STRUCTURAL_SECTION_MARKERS)

    # For step_therapy parameters, extend with severity bypass and Step Table markers
    # to ensure 0-step path language is included in the extracted context.
    # This is critical for correctly applying Rule 3 (least restrictive).
    if parameter in ("step_therapy", "num_steps_brands", "num_steps_generic",
                     "step_therapy_text", "step_phototherapy"):
        severity_bypass_markers = [
            "body surface area",
            "BSA",
            "crucial body areas",
            "10% of body surface",
            "intractable pruritus",
            "serious emotional consequences",
            "step therapy not required",
            "not required",
            "step table",
            "step 1a",
            "step 1b",
            "step 3a",
            "step 3b",
            "step 3c",
            "directed to one",
            "directed to two",
            "directed to three",
            "¥",
        ]
        effective_keywords.extend(severity_bypass_markers)

    param_pattern = _build_pattern(effective_keywords)
    base_page_nums = sorted(base_pages.keys())
    base_set = set(base_page_nums)

    # Find pages within base_pages that match parameter keywords.
    param_seeds: list[int] = [
        p for p in base_page_nums
        if _page_has_match(base_pages[p], param_pattern)
    ]

    if not param_seeds:
        logger.warning(
            "Stage 2 filter: no pages matched parameter=%s keywords; "
            "falling back to full base_pages (%d pages)",
            parameter, len(base_pages),
        )
        return base_pages

    # Expand seeds by buffer, constrained to base_pages only.
    include_set: set[int] = set()
    for seed in param_seeds:
        for p in range(seed - buffer, seed + buffer + 1):
            if p in base_set:
                include_set.add(p)

    result = {p: base_pages[p] for p in sorted(include_set)}

    logger.info(
        "Stage 2 filter: parameter=%s kept %d of %d base pages (param seeds: %s)",
        parameter, len(result), len(base_pages),
        ", ".join(str(p) for p in param_seeds),
    )

    return result
