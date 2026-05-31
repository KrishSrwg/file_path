"""
llm_client.py — Thin wrapper around Groq or OpenRouter with disk caching and retry.

Provider is controlled by USE_OPENROUTER in .env:
  USE_OPENROUTER=true  → OpenRouter  (meta-llama models, no daily quota)
  USE_OPENROUTER=false → Groq        (for final submission, default)

Single public entry point: call_llm(). All LLM calls in the pipeline go through
here so caching, retry logic, and client initialization are handled in one place.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.config import DEFAULT_MODEL, LLM_CACHE_DIR

logger = logging.getLogger(__name__)

# ── Provider selection ────────────────────────────────────────────────────────
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"

# When True, cached responses that were None (NoneType failures) are treated as
# cache misses and retried fresh. All other cached responses are still served from
# cache as normal. Set via env var:  RETRY_NA_MODE=true python run_pipeline.py
# Revert to False (default) once NoneType rows are resolved.
RETRY_NA_MODE: bool = os.getenv("RETRY_NA_MODE", "false").lower() == "true"

# When True, cached responses that are either None OR returned all-NA values
# are bypassed and retried fresh.  Use this when you want multiple diverse runs
# for consensus voting — each run retries cells where previous runs got NA.
#
# Unlike RETRY_NA_MODE (which only retries NoneType), this also retries valid
# JSON responses where every field value was "NA".  This catches the case where
# the LLM found no content (stochastically) even though the document has it.
#
# Note: quantity_limits is often legitimately all-NA; RETRY_ALL_NA_MODE will
# still retry those, potentially wasting calls.  Accept that trade-off.
# Set via env var:  RETRY_ALL_NA_MODE=true python run_pipeline.py
RETRY_ALL_NA_MODE: bool = os.getenv("RETRY_ALL_NA_MODE", "false").lower() == "true"

if USE_OPENROUTER:
    from openai import OpenAI
    import openai as _provider_module

    _api_key = os.getenv("OPEN_API_KEY")
    if not _api_key:
        raise RuntimeError(
            "OPEN_API_KEY is not set. Add it to your .env file before importing llm_client."
        )
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_api_key,
    )
    _RETRY_EXCEPTIONS = (
        _provider_module.RateLimitError,
        _provider_module.APIConnectionError,
        _provider_module.InternalServerError,
    )
    # Groq model alias → OpenRouter model ID (same weights, different naming)
    _MODEL_MAP = {
        "llama-3.1-8b-instant":    "meta-llama/llama-3.1-8b-instruct",
        "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
    }
    logger.info("LLM provider: OpenRouter")

else:
    import groq as _provider_module
    from groq import Groq

    _api_key = os.getenv("GROQ_API_KEY")
    if not _api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment before importing llm_client."
        )
    _client = Groq(api_key=_api_key)
    _RETRY_EXCEPTIONS = (
        _provider_module.RateLimitError,
        _provider_module.APIConnectionError,
        _provider_module.InternalServerError,
    )
    # Pass through unchanged — Groq uses its own model names
    _MODEL_MAP = {
        "llama-3.1-8b-instant":    "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    }
    logger.info("LLM provider: Groq")


# ── Core API call (with retry) ────────────────────────────────────────────────
# _RETRY_EXCEPTIONS is fully defined above before this decorator is evaluated.
@retry(
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_api(
    prompt: str,
    model: str,           # already-resolved provider model ID
    temperature: float,
    response_format: str | None,
) -> str:
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if response_format is not None:
        kwargs["response_format"] = {"type": response_format}

    provider_label = "OpenRouter" if USE_OPENROUTER else "Groq"
    logger.info("Calling %s [%s]: %s...", provider_label, model, prompt[:60])
    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _cached_response_is_none(cached_data: dict) -> bool:
    """Return True if a cached LLM response was None (a NoneType failure).

    Only retries genuine None responses — not all-NA JSON, not parse failures.
    All-NA JSON responses are often legitimately correct (the policy has no step
    therapy, no quantity limits, etc.), so we don't retry those.

    This function is only consulted when RETRY_NA_MODE=true.
    """
    return cached_data.get("response") is None


# Fields that are supplementary (text/reasoning) and do NOT affect scoring.
# When checking whether a cached response is "worth retrying", these fields are
# excluded from the primary-NA check. A response where ALL primary (non-supplementary)
# fields are NA is retried even when a supplementary text field is non-NA.
#
# Rationale (per group):
#   step_therapy : step_therapy_text is supplementary.  If the LLM found the
#                  policy text but left num_steps_brands, num_steps_generic, and
#                  step_phototherapy all-NA, the count extraction failed and the
#                  call is worth retrying.
#   reauth       : reauth_requirements_text is supplementary. If reauth_required
#                  is NA but the requirements text has content, the decision field
#                  failed and the call is worth retrying.
#   _reasoning   : debug field added to step_therapy output — always excluded.
_SUPPLEMENTARY_FIELDS: frozenset[str] = frozenset({
    "step_therapy_text",        # step_therapy group — text extracted, counts may still be NA
    "reauth_requirements_text", # reauth group     — text extracted, required flag may be NA
    "_reasoning",               # debug/chain-of-thought field, never a scorable value
})


def _cached_response_is_all_na(cached_data: dict) -> bool:
    """Return True if a cached response is worth retrying under RETRY_ALL_NA_MODE.

    Used by RETRY_ALL_NA_MODE to find cells worth retrying for consensus voting.
    A response whose primary (scorable) fields are all 'NA' may mean the LLM
    stochastically found nothing, even if the document contains the content.

    Check 1 — NoneType: the API returned no content at all.
    Check 2 — All fields NA: every field in the response (including text fields)
              is NA. This is the original all-NA behaviour and catches complete
              extraction failures for single-field groups (age, tb_test, etc.).
    Check 3 — All PRIMARY fields NA: supplementary text/reasoning fields are
              excluded from the NA test. If every non-supplementary field is NA,
              the count/decision extraction failed even though text was found.
              This catches the partial-extraction case for step_therapy (text
              present but step counts all NA) and reauth (text present but
              reauth_required is NA).

    Returns True for:
    - NoneType responses (null content)
    - Valid JSON where ALL fields are NA
    - Valid JSON where all PRIMARY (non-supplementary) fields are NA
    - Unparseable responses (treat as failed → worth retrying)
    """
    import re as _re
    import json as _json

    # Check 1 — NoneType
    response = cached_data.get("response")
    if response is None:
        return True

    try:
        stripped = response.strip()
        if stripped.startswith("```"):
            stripped = _re.sub(r"^```(?:json)?\s*\n?", "", stripped)
            stripped = _re.sub(r"\n?```\s*$", "", stripped).strip()
        parsed = _json.loads(stripped)

        # Check 2 — ALL fields (including text fields) are NA
        if all(str(v).strip().upper() == "NA" for v in parsed.values()):
            return True

        # Check 3 — All PRIMARY fields are NA (text/reasoning fields excluded)
        primary_values = [
            str(v).strip().upper()
            for k, v in parsed.items()
            if k not in _SUPPLEMENTARY_FIELDS
        ]
        if primary_values and all(v == "NA" for v in primary_values):
            return True

        return False

    except Exception:
        return True  # parse failure → treat as failed, worth retrying


# ── Public entry point ────────────────────────────────────────────────────────
def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    use_cache: bool = True,
    response_format: str | None = None,
) -> str:
    """Call the active LLM provider and return the raw response text.

    Args:
        prompt: The user message to send. Callers are responsible for embedding
            any system instructions directly into this string.
        model: Internal model alias (e.g. "llama-3.1-8b-instant"). The alias is
            resolved to the provider-specific model ID before the call.
        temperature: Sampling temperature. Defaults to 0.0 for deterministic output.
        use_cache: If True, serve from disk cache on hit and write on miss.
            Cache keys include the resolved model ID, so Groq and OpenRouter
            runs never share cached responses.
        response_format: Pass "json_object" to enable JSON mode; None for plain text.

    Returns:
        Raw response content string from the model.
    """
    resolved_model = _MODEL_MAP.get(model, model)

    # Cache key uses the resolved model ID so Groq ↔ OpenRouter runs stay separate
    cache_key = hashlib.sha256(
        f"{resolved_model}|{temperature}|{response_format}|{prompt}".encode()
    ).hexdigest()
    cache_path = LLM_CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cached = json.load(f)

        # In RETRY_NA_MODE, bypass cache if the stored response was None.
        # In RETRY_ALL_NA_MODE, bypass cache if the response was None OR all-NA.
        # Both modes let subsequent runs produce diverse results for consensus voting.
        should_retry = (
            (RETRY_NA_MODE and _cached_response_is_none(cached)) or
            (RETRY_ALL_NA_MODE and _cached_response_is_all_na(cached))
        )
        if should_retry:
            mode = "RETRY_ALL_NA_MODE" if RETRY_ALL_NA_MODE else "RETRY_NA_MODE"
            logger.info(
                "%s: cache hit but response was None/all-NA — retrying fresh: %s...",
                mode, prompt[:60],
            )
        else:
            logger.info("Cache hit: %s...", prompt[:60])
            return cached["response"]

    response_text = _call_api(prompt, resolved_model, temperature, response_format)

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_entry = {
        "model": resolved_model,
        "temperature": temperature,
        "response_format": response_format,
        "prompt": prompt,
        "response": response_text,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache_entry, f, indent=2, ensure_ascii=False)

    return response_text
