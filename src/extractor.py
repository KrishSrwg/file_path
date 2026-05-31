"""
extractor.py — Per-parameter-group PA extraction using XML prompts.

For each (pages, brand) pair, applies a two-stage filtering pipeline and
makes 7 grouped LLM calls — one per PARAMETER_GROUPS entry — each with
its own focused page context and XML prompt. Returns a flat dict of all
12 extracted parameter values.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from src.page_filter import _build_pattern, _page_has_match

from src.config import (
    BRAND_TO_GENERICS,
    INTER_CALL_DELAY_GROQ,
    INTER_CALL_DELAY_OPENROUTER,
    PARAMETER_GROUPS,
    PARAMETER_KEYWORDS,
    PARAMETER_MODELS,
    STAGE2_MAX_PAGES_8B,
    STAGE2_MAX_PAGES_70B,
    MODEL_8B,
)
from src.llm_client import call_llm
from src.page_filter import filter_pages, filter_pages_for_parameter

logger = logging.getLogger(__name__)

# Directory containing the 7 XML prompt files.
PROMPTS_DIR: Path = Path(__file__).parent / "prompts"

# All 12 extraction keys in submission-tab order.
# Imported by run_pipeline.py — do not remove or rename.
EXPECTED_KEYS: list[str] = [
    "age",
    "step_therapy_text",
    "num_steps_brands",
    "num_steps_generic",
    "step_phototherapy",
    "tb_test",
    "initial_auth_duration",
    "reauth_duration",
    "reauth_required",
    "reauth_requirements_text",
    "specialist_types",
    "quantity_limits",
]

# Seconds to wait between consecutive LLM calls within a single row.
# Groq: prevents per-minute token exhaustion. OpenRouter: no hard wall, so 0.
_USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "false").lower() == "true"
INTER_CALL_DELAY: int = INTER_CALL_DELAY_OPENROUTER if _USE_OPENROUTER else INTER_CALL_DELAY_GROQ

# For these groups, if no keywords are found in the document at all,
# the parameter is definitively absent — skip the LLM call and return NA.
# This prevents hallucination on groups where the LLM infers from context.
KEYWORD_REQUIRED_GROUPS: frozenset[str] = frozenset({"specialist_types", "quantity_limits"})

# ---------------------------------------------------------------------------
# Mismatch detection patterns
# ---------------------------------------------------------------------------

# Some documents cover only IV/medical-benefit administration of a drug
# (e.g., UHC medical benefit policy for intravenous Tremfya for UC/CD
# induction) while explicitly stating that the self-administered subcutaneous
# formulation — the one used for plaque psoriasis — is handled under the
# pharmacy benefit in a separate policy document.
#
# These documents contain zero PsO PA criteria. The brand name appears
# (so the standard mismatch check does not fire), but the plaque psoriasis
# indication only appears in informational FDA sections, never in the
# coverage criteria. Attempting extraction produces cross-indication noise
# (e.g., "Gastroenterologist" from UC criteria).
#
# Signal: any page contains text matching this pattern → all-NA immediately.
_IV_ONLY_REDIRECT_PATTERN: re.Pattern = re.compile(
    # UHC-style explicit redirect sentence
    r"self[\s\-]?administered\s+subcutaneous\s+injection\s+is\s+obtained\s+under\s+the\s+pharmacy\s+benefit"
    r"|subcutaneous\s+injection\s+is\s+obtained\s+under\s+the\s+pharmacy\s+benefit"
    # Broader form: "for self-administered … pharmacy benefit"
    r"|for\s+self[\s\-]?administered.{0,60}pharmacy\s+benefit"
    # Direct "pharmacy benefit" + subQ adjacency
    r"|subcutaneous.{0,40}pharmacy\s+benefit.{0,40}separate",
    re.IGNORECASE | re.DOTALL,
)


def _cap_pages(pages: dict[int, str], model: str) -> dict[int, str]:
    """Enforce a hard page cap on Stage 2 output before building the LLM prompt.

    The 8B model has a 6,000 TPM hard wall; the 70B model has 12,000.
    When Stage 2 returns more pages than the cap, select `limit` pages using
    EVENLY-DISTRIBUTED SAMPLING across the full sorted page range.

    Rationale for even distribution over first+last:
      - Single-drug policies (3–30 pages): evenly distributed ≈ all pages, so no loss.
      - Large multi-drug handbooks (100–300 pages): the target drug's section is
        wherever it alphabetically falls — often in the middle. First+last would
        systematically send the document header and an unrelated drug's section,
        while completely missing the target drug.
      - Even distribution guarantees at least one sample from every ~(N/limit)-page
        band, giving the LLM a representative cross-section of the full document.

    Args:
        pages: Page dict returned by filter_pages_for_parameter.
        model: Provider-resolved model ID string (checked for '70b' substring).

    Returns:
        Original dict unchanged if already within cap; truncated dict otherwise.
    """
    limit = STAGE2_MAX_PAGES_70B if "70b" in model.lower() else STAGE2_MAX_PAGES_8B
    if len(pages) <= limit:
        return pages

    sorted_keys = sorted(pages.keys())
    n = len(sorted_keys)

    # Evenly distributed: pick `limit` indices spread uniformly across [0, n-1].
    # For limit=8, n=65: picks indices 0, 9, 18, 27, 36, 45, 54, 64 → good coverage.
    # For limit=1: just the first page.
    if limit == 1:
        kept_keys = [sorted_keys[0]]
    else:
        step = (n - 1) / (limit - 1)
        kept_keys = sorted({sorted_keys[round(i * step)] for i in range(limit)})
        # If dedup reduced count (rare with float rounding), pad with adjacent pages
        while len(kept_keys) < limit and len(kept_keys) < n:
            all_set = set(sorted_keys)
            used_set = set(kept_keys)
            remaining = sorted(all_set - used_set)
            if remaining:
                kept_keys = sorted(kept_keys + [remaining[len(kept_keys) % len(remaining)]])
            else:
                break

    logger.warning(
        "_cap_pages: %d pages → capped to %d (evenly distributed) for model %s",
        len(pages), limit, model,
    )
    return {k: pages[k] for k in kept_keys[:limit]}


def _load_prompt(group_name: str) -> str:
    """Read XML prompt template for the given group from PROMPTS_DIR.

    Raises FileNotFoundError if the file does not exist.
    """
    prompt_path = PROMPTS_DIR / f"{group_name}.xml"
    return prompt_path.read_text(encoding="utf-8")


def _fill_prompt(
    template: str,
    brand: str,
    generics_list: str,
    document_text: str,
) -> str:
    """Replace {brand}, {generics_list}, {document_text} placeholders."""
    return (
        template
        .replace("{brand}", brand)
        .replace("{generics_list}", generics_list)
        .replace("{document_text}", document_text)
    )


def _pages_to_text(pages: dict[int, str]) -> str:
    """Format filtered pages as document text.

    Each page is prefixed with '=== PAGE N ===' and separated by two newlines.
    """
    return "\n\n".join(
        f"=== PAGE {page_num} ===\n{text}"
        for page_num, text in sorted(pages.items())
    )


def _get_group_keywords(group_name: str) -> list[str]:
    """Return deduplicated union of PARAMETER_KEYWORDS for all params in the group."""
    seen: set[str] = set()
    keywords: list[str] = []
    for param in PARAMETER_GROUPS[group_name]:
        for kw in PARAMETER_KEYWORDS.get(param, []):
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)
    return keywords


def _parse_group_response(
    raw: str | None,
    group_params: list[str],
) -> dict[str, str]:
    """Parse the LLM's JSON response for a group.

    Handles three real-world response shapes from OpenRouter / Groq:
      1. Raw JSON object             (ideal)
      2. ```json ... ``` fenced JSON (OpenRouter 70B frequently wraps responses)
      3. None content                (OpenRouter occasionally returns no content)

    For each key in group_params:
      - If key is present in parsed JSON: use its string value.
      - If key is missing or parsing fails: use "NA".
    Logs a warning on every non-ideal path.
    Never raises — always returns a complete dict for the group.
    """
    # ── Mode 3: None content ──────────────────────────────────────────────────
    # OpenRouter occasionally returns None for message.content.
    # json.loads(None) raises TypeError, not JSONDecodeError, so the bare
    # except-JSONDecodeError in the old code let this propagate as an uncaught
    # exception up to extract_fields' broad except-Exception handler.
    if raw is None:
        logger.warning(
            "LLM returned None content for group params %s — returning NA",
            group_params,
        )
        return {key: "NA" for key in group_params}

    # ── Mode 2: strip markdown code fences ───────────────────────────────────
    # OpenRouter 70B often wraps valid JSON in ```json … ``` blocks.
    # Strip them before attempting to parse so the content is rescued.
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped).strip()

    # ── Mode 1: parse raw (or stripped) JSON ─────────────────────────────────
    try:
        parsed: dict = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning(
            "JSON parse failure for group params %s. Raw response (first 200 chars): %s",
            group_params,
            raw[:200],
        )
        return {key: "NA" for key in group_params}

    result: dict[str, str] = {}
    for key in group_params:
        if key in parsed:
            result[key] = str(parsed[key])
        else:
            logger.warning("Missing key %r in LLM response for params %s", key, group_params)
            result[key] = "NA"
    return result


def extract_fields(
    pages: dict[int, str],
    brand: str,
    model: str = MODEL_8B,  # kept for API compatibility; routing is per-group internally
) -> dict[str, str]:
    """Extract all 12 PA parameters for a (brand, full-pages) pair.

    Applies Stage 1 brand+indication page filtering, then for each of the
    7 parameter groups applies Stage 2 group-specific filtering, loads the
    corresponding XML prompt, calls the LLM, and parses the response.

    Args:
        pages: Full page dict from pdf_reader.extract_text().
        brand: Uppercase brand name matching BRAND_TO_GENERICS keys.
        model: Ignored internally; kept so callers need no changes.

    Returns:
        Dict with exactly EXPECTED_KEYS. Any key whose extraction failed
        defaults to "NA". Never raises.
    """
    result: dict[str, str] = {key: "NA" for key in EXPECTED_KEYS}

    generics = BRAND_TO_GENERICS.get(brand, [])
    generics_list = ", ".join(generics) if generics else brand.lower()

    logger.info("Starting extraction: brand=%s, total_pages=%d", brand, len(pages))

    base_pages = filter_pages(pages, brand)
    if not base_pages:
        logger.warning(
            "Stage 1 filter returned no pages for brand=%s; returning all-NA result", brand
        )
        return result

    logger.info("Stage 1 complete: %d pages kept for %s", len(base_pages), brand)

    # Document mismatch check: if no page in base_pages actually contains a brand
    # keyword, Stage 1 only kept pages via the always_include_first fallback (first 3
    # pages) rather than a genuine brand+indication match. This is a definitive signal
    # that the PDF does not contain PA criteria for this brand (e.g. a mis-filed
    # document mapped to the wrong row in the submissions tab, such as a Multiple
    # Myeloma policy being processed for STELARA).
    #
    # In this state continuing extraction would only waste LLM calls and risk
    # hallucination — the LLM has no brand-relevant content to work with and will
    # invent values from its training knowledge.  We return all-NA immediately.
    #
    # Safety note: always_include_first=3 seeds pages 1-3 unconditionally; the
    # brand+indication seed then adds any page where both appear.  If the brand
    # truly exists in the document, at least one of those pages will contain it.
    # A brand absent from ALL filtered pages is genuinely absent from the document.
    brand_terms_check = [brand.lower()] + BRAND_TO_GENERICS.get(brand, [])
    brand_found_in_base = any(
        any(term in text.lower() for term in brand_terms_check)
        for text in base_pages.values()
    )
    if not brand_found_in_base:
        logger.warning(
            "DOCUMENT MISMATCH: no brand keywords for '%s' found in any filtered page. "
            "This document does not contain PA criteria for this brand "
            "(e.g. a mis-filed document or wrong-row mapping in the submissions tab). "
            "Returning all-NA immediately — no LLM calls will be made for this row.",
            brand,
        )
        return result  # all fields already initialised to "NA"

    # Benefit-type mismatch check: detect documents that cover only IV/medical-
    # benefit administration while explicitly redirecting the subcutaneous
    # (self-administered) formulation — the one used for plaque psoriasis — to
    # the pharmacy benefit under a separate policy.
    #
    # Example: UHC 2025D0134B — Tremfya for intravenous UC/CD induction only.
    # Page 1 says: "Tremfya for self-administered subcutaneous injection is
    # obtained under the pharmacy benefit." There are no PsO PA criteria
    # anywhere in the document. Continuing extraction would produce
    # cross-indication noise (e.g., Gastroenterologist from UC/CD criteria).
    # Returning all-NA immediately is the correct result.
    iv_only_redirect = any(
        _IV_ONLY_REDIRECT_PATTERN.search(text)
        for text in base_pages.values()
    )
    if iv_only_redirect:
        logger.warning(
            "BENEFIT MISMATCH: Document for '%s' explicitly redirects the "
            "self-administered subcutaneous formulation (used for PsO) to the "
            "pharmacy benefit. This document covers only IV/medical-benefit "
            "administration and contains no plaque psoriasis PA criteria. "
            "Returning all-NA — no LLM calls will be made for this row.",
            brand,
        )
        return result  # all fields already initialised to "NA"

    for group_name, group_params in PARAMETER_GROUPS.items():
        try:
            group_keywords = _get_group_keywords(group_name)
            group_pages    = filter_pages_for_parameter(base_pages, group_name, group_keywords)
            group_model    = PARAMETER_MODELS[group_name]  # resolved early for cap logic

            # Hard page cap: keeps first + last halves, see _cap_pages docstring.
            group_pages = _cap_pages(group_pages, group_model)

            # For keyword-required groups: if no keywords exist anywhere in the document,
            # the parameter is definitively absent — skip the LLM call entirely.
            if group_name in KEYWORD_REQUIRED_GROUPS:
                kw_pattern = _build_pattern(group_keywords)
                any_keyword_found = any(
                    _page_has_match(text, kw_pattern)
                    for text in base_pages.values()
                )
                if not any_keyword_found:
                    for key in group_params:
                        result[key] = "NA"
                    logger.info(
                        "Group '%s': no keywords found in document — returning NA, LLM skipped",
                        group_name,
                    )
                    time.sleep(INTER_CALL_DELAY)
                    continue

            document_text = _pages_to_text(group_pages)
            template = _load_prompt(group_name)
            prompt = _fill_prompt(template, brand, generics_list, document_text)

            logger.info(
                "Calling LLM for group '%s' with model %s (%d pages)",
                group_name, group_model, len(group_pages),
            )

            raw = call_llm(prompt, model=group_model, response_format="json_object")

            # Bug fix: if the LLM returns None content (malformed/empty response),
            # retry ONCE with a short delay before giving up and returning NA.
            # Previously, None went straight to NA with zero retry, losing ~31% of
            # calls for reauth/step_therapy groups. This single retry recovers most
            # of those cases since the failure is usually transient (API flakiness).
            if raw is None:
                logger.warning(
                    "Group '%s' got None content on first call — retrying once",
                    group_name,
                )
                time.sleep(INTER_CALL_DELAY * 2)
                raw = call_llm(prompt, model=group_model, response_format="json_object")
                if raw is None:
                    logger.warning(
                        "Group '%s' got None content on retry too — returning NA",
                        group_name,
                    )

            group_result = _parse_group_response(raw, group_params)
            result.update(group_result)

            logger.debug("Group '%s' result: %s", group_name, group_result)

        except Exception as exc:
            logger.error("Group %s failed: %s", group_name, exc)

        time.sleep(INTER_CALL_DELAY)

    logger.info("Extraction complete for %s: %s", brand, result)
    return result
