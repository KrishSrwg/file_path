"""
access_score.py — Compute Access Score (0–100) from extracted PA policy parameters.

The Access Score measures how accessible a drug is under a specific payer policy
compared to the FDA-approved label for that brand.

Scale:
  100 = No restrictions beyond FDA label (best possible access)
   75 = Minor restrictions only (good access)
   50 = Moderate restrictions (typical payer policy with step therapy)
   25 = Significant restrictions (multiple branded steps or severe admin burden)
    0 = Near-impossible access (extreme step therapy + all admin barriers)

Formula: score = 100 - category1_penalty - category2_penalty - category3_penalty
  Category 1 (Clinical Gatekeeping, max 65 pts): branded steps + generic steps + phototherapy
  Category 2 (Administrative Friction,  max 25 pts): auth duration + reauth + specialist + QL + TB
  Category 3 (Eligibility,              max 10 pts): age restriction

All comparisons are against the brand-specific FDA label baseline sourced directly
from FDA.gov / DailyMed (verified June 2026).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# FDA label details — verified from official FDA sources (accessdata.fda.gov /
# DailyMed), sourced June 2026.
# ---------------------------------------------------------------------------
# Fields per brand:
#   min_age       : FDA-approved minimum age for plaque psoriasis (integer years)
#   tb_in_label   : True if TB screening is required/recommended in the FDA
#                   prescribing information (all biologics: True; non-biologics: False)
#   drug_type     : "biologic" | "targeted_synthetic" | "conventional"
#                   (informational — affects TB baseline comparison)
# ---------------------------------------------------------------------------

FDA_LABEL: dict[str, dict] = {
    # ── Ustekinumab originator & biosimilars (IL-12/23 antagonist) ───────────
    "STELARA": {
        "min_age": 6,         # Approved ≥6 years for PsO (FDA label 2025)
        "tb_in_label": True,  # "Evaluate patients for TB infection prior to initiating"
        "drug_type": "biologic",
    },
    "YESINTEK": {
        "min_age": 6,         # Biosimilar of Stelara; same indication ≥6 years
        "tb_in_label": True,
        "drug_type": "biologic",
    },
    "OTULFI": {
        "min_age": 6,         # Biosimilar of Stelara; same indication ≥6 years
        "tb_in_label": True,
        "drug_type": "biologic",
    },
    # ── Guselkumab (IL-23 antagonist) ────────────────────────────────────────
    "TREMFYA": {
        "min_age": 6,         # FDA approved ≥6 years (≥40 kg) September 29, 2025
        "tb_in_label": True,  # "Monitor patients for signs and symptoms of active TB"
        "drug_type": "biologic",
    },
    # ── Secukinumab (IL-17A antagonist) ──────────────────────────────────────
    "COSENTYX": {
        "min_age": 6,         # Approved ≥6 years for PsO (FDA label 2023)
        "tb_in_label": True,  # "Evaluate patients for tuberculosis infection"
        "drug_type": "biologic",
    },
    # ── Etanercept (TNF antagonist) ───────────────────────────────────────────
    "ENBREL": {
        "min_age": 4,         # Approved ≥4 years for chronic moderate-severe PsO
        "tb_in_label": True,  # TB screening stated in Warnings and Precautions
        "drug_type": "biologic",
    },
    # ── Infliximab originator (anti-TNF) ──────────────────────────────────────
    "REMICADE": {
        "min_age": 18,        # Approved adults only for chronic severe plaque PsO
        "tb_in_label": True,  # BOXED WARNING: serious infections including TB
        "drug_type": "biologic",
    },
    # ── Brodalumab (IL-17RA antagonist) ──────────────────────────────────────
    "SILIQ": {
        "min_age": 18,        # Adults only; safety/effectiveness not established in pediatrics
        "tb_in_label": True,  # "Evaluate patients for tuberculosis infection"
        "drug_type": "biologic",
    },
    # ── Certolizumab pegol (TNF antagonist) ──────────────────────────────────
    "CIMZIA": {
        "min_age": 18,        # Adults only for plaque psoriasis
        "tb_in_label": True,  # BOXED WARNING: serious infections including TB
        "drug_type": "biologic",
    },
    # ── Bimekizumab (IL-17A and IL-17F antagonist) ────────────────────────────
    "BIMZELX": {
        "min_age": 18,        # Adults only; approved Oct 2023 for PsO in adults
        "tb_in_label": True,  # "Evaluate for active and latent tuberculosis prior to initiating"
        "drug_type": "biologic",
    },
    # ── Risankizumab (IL-23 antagonist) ──────────────────────────────────────
    "SKYRIZI": {
        "min_age": 18,        # Adults only; "safety/effectiveness not established in pediatric patients"
        "tb_in_label": True,  # "Evaluate patients for TB infection prior to initiating"
        "drug_type": "biologic",
    },
    # ── Tildrakizumab (IL-23 antagonist) ─────────────────────────────────────
    "ILUMYA": {
        "min_age": 18,        # Adults only (≥18 years) for plaque psoriasis
        "tb_in_label": True,  # "Evaluate patients for tuberculosis prior to initiating"
        "drug_type": "biologic",
    },
    # ── Adalimumab biosimilar (TNF antagonist) ────────────────────────────────
    "AMJEVITA": {
        "min_age": 18,        # Adults only for plaque psoriasis (PsO indication is adults)
        "tb_in_label": True,  # Same as reference Humira: TB screening in label
        "drug_type": "biologic",
    },
    # ── Apremilast (PDE4 inhibitor — targeted synthetic, NOT a biologic) ──────
    "OTEZLA": {
        "min_age": 6,         # Approved ≥6 years (≥20 kg) for PsO — April 2024
        "tb_in_label": False, # PDE4 inhibitor; TB screening NOT required in FDA label
                              # (no immunosuppression of TB-relevant pathways)
        "drug_type": "targeted_synthetic",
    },
    # ── Acitretin (retinoid — conventional systemic, NOT a biologic) ──────────
    "ACITRETIN": {
        "min_age": 18,        # Adults only for severe plaque psoriasis (teratogenicity concerns)
        "tb_in_label": False, # Retinoid; no TB screening required in FDA label
        "drug_type": "conventional",
    },
}


# ---------------------------------------------------------------------------
# Helper parsers
# ---------------------------------------------------------------------------

def _is_na(val) -> bool:
    """Return True if val represents a missing/NA value."""
    if val is None:
        return True
    return str(val).strip().lower() in {
        "na", "nan", "none", "", "n/a", "not applicable",
        "not mentioned", "not stated", "not specified",
    }


def _parse_step_count(val) -> int | None:
    """Parse a step count like '1', '2', '3' → int. NA/None → None."""
    if _is_na(val):
        return None
    s = str(val).strip()
    # Direct integer
    try:
        n = int(float(s))
        return n if n > 0 else None
    except (ValueError, TypeError):
        pass
    # Extract first digit from descriptive text
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        return n if n > 0 else None
    return None


def _parse_months(val) -> int | None:
    """Parse '6 Months', '12 Months', etc. → int months. NA/None → None."""
    if _is_na(val):
        return None
    s = str(val).strip()
    m = re.search(r"(\d+)\s*months?", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_age(val) -> int | None:
    """Parse '>=18', '>=6', 'FDA labelled age' → int. NA/None/'FDA labelled age' → None."""
    if _is_na(val):
        return None
    s = str(val).strip()
    if "fda" in s.lower():
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_access_score(record: dict[str, str], brand: str) -> int:
    """Compute Access Score (0–100) for a single (record, brand) pair.

    Args:
        record : Normalized extraction dict from normalize_record().
                 Keys use the canonical parameter names from EXPECTED_KEYS.
        brand  : Uppercase brand name, e.g. "STELARA".

    Returns:
        Integer in [0, 100]. Higher = better access (fewer payer restrictions
        relative to the FDA label baseline for that brand).
    """
    fda = FDA_LABEL.get(brand.upper())
    if fda is None:
        # Unknown brand — return neutral score
        return 50

    # ── Category 1: Clinical Gatekeeping (max 65 pts) ─────────────────────
    # These are the primary drivers of access: step therapy requirements
    # that restrict which patients can qualify for the drug.
    cat1 = 65.0

    # 1a. Branded step therapy (FDA baseline: none required)
    brands = _parse_step_count(record.get("num_steps_brands"))
    if brands == 1:
        cat1 -= 25   # First biologic step: largest single barrier; excludes
                     # biologically naive patients (the majority of the pool)
    elif brands == 2:
        cat1 -= 42   # 25 + 17 — diminishing: second step works on a smaller
                     # already-filtered patient population
    elif brands is not None and brands >= 3:
        cat1 -= 55   # 25 + 17 + 13 — further diminishing; near-impossible for most

    # 1b. Generic/conventional step therapy (FDA baseline: none required)
    generic = _parse_step_count(record.get("num_steps_generic"))
    if generic == 1:
        cat1 -= 8    # Standard-of-care in dermatology; moderate barrier
    elif generic == 2:
        cat1 -= 14   # 8 + 6 — diminishing
    elif generic is not None and generic >= 3:
        cat1 -= 18   # 8 + 6 + 4 — further diminishing

    # 1c. Phototherapy requirement (FDA baseline: not required)
    photo = str(record.get("step_phototherapy", "NA")).strip()
    if photo == "Yes":
        cat1 -= 10   # AND-required phototherapy: significant burden (clinic visits
                     # 2–3× per week for months)

    # ── Category 2: Administrative Friction (max 25 pts) ──────────────────
    # These parameters add ongoing burden but rarely prevent access entirely.
    cat2 = 25.0

    # 2a. Initial authorization duration (FDA baseline: no PA, unlimited duration)
    auth_months = _parse_months(record.get("initial_auth_duration"))
    if auth_months is not None:
        if auth_months <= 3:
            cat2 -= 12   # Very restrictive: quarterly PA filing
        elif auth_months <= 6:
            cat2 -= 8    # Moderately restrictive: semi-annual filing
        elif auth_months <= 12:
            cat2 -= 4    # Standard in the industry; minor friction
        else:
            cat2 -= 2    # >12 months: minimal friction

    # 2b. Reauthorization required (FDA baseline: no reauth required)
    reauth = str(record.get("reauth_required", "NA")).strip()
    if reauth == "Yes":
        cat2 -= 5

    # 2c. Specialist prescriber required (FDA baseline: not required)
    specialist = record.get("specialist_types", "NA")
    if not _is_na(specialist):
        cat2 -= 3    # Adds friction but most PsO patients already see dermatologists

    # 2d. Quantity limits (FDA baseline: none)
    ql = record.get("quantity_limits", "NA")
    if not _is_na(ql):
        cat2 -= 3

    # 2e. TB test requirement — compared against brand-specific FDA baseline.
    #     For biologics: FDA label requires TB screening → payer "Yes" = parity.
    #     For non-biologics (OTEZLA, ACITRETIN): FDA does NOT require TB test.
    tb = str(record.get("tb_test", "NA")).strip()
    if fda["tb_in_label"]:
        # Biologic: FDA says screen → payer "Yes" = parity (no change)
        # Payer "NA" = slightly more permissive than FDA → small bonus
        if tb == "NA":
            cat2 += 2
    else:
        # Non-biologic: FDA does NOT require → payer "Yes" = more restrictive
        if tb == "Yes":
            cat2 -= 2

    # ── Category 3: Eligibility (max 10 pts) ──────────────────────────────
    # Compares payer age threshold against the brand-specific FDA minimum.
    cat3 = 10.0

    policy_age = _parse_age(record.get("age"))
    fda_min_age = fda["min_age"]

    if policy_age is not None:
        if policy_age > fda_min_age:
            # Payer is more restrictive than FDA label → penalty
            age_excess = policy_age - fda_min_age
            if age_excess <= 6:
                cat3 -= 4    # Modest restriction (e.g., >=12 when FDA allows >=6)
            else:
                cat3 -= 7    # Significant restriction (e.g., >=18 when FDA allows >=6)
        # If policy_age <= fda_min_age: parity or more permissive → no change

    # ── Final calculation ──────────────────────────────────────────────────
    raw = cat1 + cat2 + cat3
    return max(0, min(100, round(raw)))
