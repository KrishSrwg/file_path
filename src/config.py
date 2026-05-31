"""
Central configuration for the PA extraction pipeline.

Contains only constants — no logic. Import this module wherever you need
brand mappings, keyword lists, model names, or filesystem paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"
EXTRACTED_TEXT_DIR: Path = DATA_DIR / "extracted_text"
LLM_CACHE_DIR: Path = DATA_DIR / "llm_responses"
RECON_NOTES_DIR: Path = DATA_DIR / "recon_notes"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Named constants — reference these everywhere instead of raw strings.
MODEL_8B: str  = "llama-3.1-8b-instant"
MODEL_70B: str = "llama-3.3-70b-versatile"

# Default for any call that doesn't specify a group.
DEFAULT_MODEL: str = MODEL_8B

# ---------------------------------------------------------------------------
# Brand → generic name(s)
# Keys are UPPERCASE to match the Submissions tab format.
# ---------------------------------------------------------------------------

BRAND_TO_GENERICS: dict[str, list[str]] = {
    "TREMFYA": ["guselkumab"],
    "STELARA": ["ustekinumab", "stelera"],  # stelera is a known typo in some payer documents (e.g. HAP)
    "AMJEVITA": ["adalimumab"],
    "COSENTYX": ["secukinumab"],
    "ENBREL": ["etanercept"],
    "REMICADE": ["infliximab"],
    "SILIQ": ["brodalumab"],
    "CIMZIA": ["certolizumab pegol", "certolizumab"],
    "BIMZELX": ["bimekizumab"],
    "SKYRIZI": ["risankizumab"],
    "OTEZLA": ["apremilast"],
    "YESINTEK": ["ustekinumab-kfce", "ustekinumab"],
    "OTULFI": ["ustekinumab"],
    "ILUMYA": ["tildrakizumab"],
    "ACITRETIN": ["acitretin"],
}

# ---------------------------------------------------------------------------
# Indication filtering
# ---------------------------------------------------------------------------

INDICATION_KEYWORDS: list[str] = [
    "plaque psoriasis",
    "chronic plaque psoriasis",
    "psoriasis vulgaris",
    "PsO",
    "moderate-to-severe plaque psoriasis",
    "severe plaque psoriasis",
    "psoriasis",
    "(Ps)",
]

EXCLUDE_INDICATIONS: list[str] = [
    # Other psoriasis variants
    "psoriatic arthritis",
    "PsA",
    "juvenile psoriatic arthritis",
    "JPsA",
    "guttate psoriasis",
    "inverse psoriasis",
    "pustular psoriasis",
    "generalized pustular psoriasis",
    "GPP",
    "erythrodermic psoriasis",
    "palmoplantar psoriasis",
    "nail psoriasis",
    "scalp psoriasis",
    "facial psoriasis",
    "intertriginous psoriasis",
    # Other indications commonly co-covered
    "Crohn's disease",
    "ulcerative colitis",
    "rheumatoid arthritis",
    "ankylosing spondylitis",
    "hidradenitis suppurativa",
    "non-radiographic axial spondyloarthritis",
]

# ---------------------------------------------------------------------------
# Per-parameter keyword lists
# Used by the page filter and as prompt hints.
# ---------------------------------------------------------------------------

PARAMETER_KEYWORDS: dict[str, list[str]] = {
    "age": [
        "adult",
        "adult patients",
        "pediatric",
        "≥18 years",
        "18 years of age",
        "years of age",
        "minimum age",
        "FDA labeled age",
        "age restriction",
    ],
    "step_therapy_text": [
        "tried and failed",
        "trial of",
        "trial and failure",
        "inadequate response",
        "intolerance",
        "contraindication",
        "previously received",
        "documented failure",
        "preferred agent",
        "prior therapy",
        "treatment failure",
        "adverse reaction",
        # Severity bypass language — critical for detecting 0-step paths
        "body surface area",
        "BSA",
        "crucial body areas",
        "10% of body surface",
        "hands, feet",
        "intractable pruritus",
        "¥",  # blanket exception note marker
        "step table",
        "step 1",
        "step 2",
        "step 3",
        "not required",          # blanket exception: "step therapy not required"
    ],
    "num_steps_brands": [
        "TNF blocker",
        "TNF inhibitor",
        "biologic",
        "biologic immunomodulator",
        "targeted immune modulator",
        "biologic DMARD",
        "branded product",
        "preferred biologic",
        # Step Table tier markers — critical for multi-drug PBM policies
        "step 1",
        "step 2",
        "step 3a",
        "step 3b",
        "step 3c",
        "step 1a",
        "step 1b",
        "directed to one",
        "directed to two",
        "directed to three",
        "non-preferred",
        "preferred agent",
        # Common biologic names used as step prerequisites
        "Humira",
        "Enbrel",
        "Cosentyx",
        "Skyrizi",
        "Stelara",
        "Tremfya",
        "Otezla",
        "preferred products",
    ],
    "num_steps_generic": [
        "topical agent",
        "topical corticosteroid",
        "methotrexate",
        "cyclosporine",
        "acitretin",
        "conventional agent",
        "conventional therapy",
        "non-biologic",
        "oral DMARD",
        "vitamin D analog",
        "tazarotene",
        "systemic agent",
        "calcipotriene",
        "anthralin",
        "coal tar",
        "tacrolimus",
        "pimecrolimus",
        # Severity bypass / 0-step path detection
        "body surface area",
        "BSA",
        "crucial body areas",
        "10% of body surface",
        "hands, feet",
        "¥",
        "not required",
        "contraindication to all",
        "FDA labeled contraindication",
    ],
    "step_phototherapy": [
        "phototherapy",
        "UVB",
        "PUVA",
        "ultraviolet",
        "psoralens",
        "light therapy",
    ],
    "tb_test": [
    "tuberculosis",
    "TB",
    "TB test",
    "latent TB",
    "TB infection",
    "TB screening",
    "TST",
    "tuberculin",
    "IGRA",
    "PPD",
    "QuantiFERON",
    # additions — cover phrasing found in recon notes for missed docs
    "documented negative",
    "negative tuberculosis",
    "latent tuberculosis",
    "interferon-release assay",
    "interferon-gamma",
    "screening testing for TB",
    "tested for latent",
    "TB prior to initiating",
    "evaluate for tuberculosis",
    "active TB",
    # additions from 14-PDF markdown analysis:
    # TB test appears in general sections, not PsO-specific ones
    "general coverage criteria",   # HAP document header containing TB requirement
    "all medication requests",     # HAP "Applicable to all medication requests"
    "for all indications",         # Aetna "For all indications: Member has had a documented negative TB test"
    ],
    "initial_auth_duration": [
        "duration of approval",
        "initial approval",
        "authorization duration",
        "length of approval",
        "approve for",
        "coverage duration",
        "initial authorization",
        # Additional label variants found across payers:
        "renewal approval",        # Aetna "Initial Approval:" / "Renewal Approval:" sections
        "initial pa requests",     # ForwardHealth "Initial PA requests may be approved for up to"
        "renewal pa requests",     # ForwardHealth "Renewal PA requests may be approved for up to"
        "authorization of",        # Aetna "Authorization of N months may be granted"
        "approve for the duration",
    ],
    "reauth_duration": [
        "reauthorization",
        "renewal approval",
        "recertification",
        "continuation",
        "approval duration",
        # Additional label variants found across payers:
        "renewal pa requests",     # ForwardHealth renewal duration label
        "currently receiving",     # "Patient is Currently Receiving [drug]" reauth path
        "reauthorization criteria",
    ],
    "reauth_required": [
        "reauthorization",
        "renewal",
        "continuation",
        "recertification",
        "continued approval",
        "currently receiving",     # alternative reauth path label
        "continuation of therapy", # Aetna / Wellmark reauth section header
    ],
    "reauth_requirements_text": [
        "PASI score",
        "BSA",
        "body surface area",
        "clinical response",
        "clinical benefit",
        "improvement",
        "positive response",
        "medical necessity",
        "chart notes",
        "ongoing benefit",
        "continuation of therapy",  # Aetna / Wellmark BCBS reauth section header
        "currently receiving",      # alternate reauth section label (WHA, 215824)
    ],
    "specialist_types": [
        "dermatologist",
        "rheumatologist",
        "prescriber specialty",
        "consultation with",
        "prescribed by",
        "specialist",
    ],
    "quantity_limits": [
    "quantity limit",
    "quantity restriction",
    "maximum units",
    "billable units",
    "day supply",
    "per dose",
    "coverage limited to",
    "(QL)",
    # additions — cover phrasing found in recon notes for missed docs
    "dosing limits",
    "dosing limit",
    "Medical Specialty Medication",
    "program quantity limit",
    "QL Amount",
    "Day Supply",
    "unit limit",
    "dispensing limit",
    # additions from 14-PDF markdown analysis:
    "quantity level limit",  # CRITICAL: Aetna uses "Quantity Level Limit:" (not just "quantity limit")
    ],
}

# ---------------------------------------------------------------------------
# Parameter groups for LLM extraction
#
# Each group is extracted in one LLM call. Parameters are grouped when they
# come from the same source section and require the same reasoning context.
# 7 calls per (filename, brand) instead of 12.
# ---------------------------------------------------------------------------

PARAMETER_GROUPS: dict[str, list[str]] = {
    # Step therapy group: all four parameters come from the step therapy
    # section and require holistic reasoning. Counted together to ensure
    # consistency in branded vs generic step counts.
    "step_therapy": [
        "step_therapy_text",
        "num_steps_brands",
        "num_steps_generic",
        "step_phototherapy",
    ],
    # Duration group: initial and reauth durations are typically stated
    # in the same "Duration of Approval" section.
    "duration": [
        "initial_auth_duration",
        "reauth_duration",
    ],
    # Reauth group: whether reauth is required and what it requires are
    # answered from the same continuation/renewal section.
    "reauth": [
        "reauth_required",
        "reauth_requirements_text",
    ],
    # Standalone parameters: each lives in a different section.
    "age":              ["age"],
    "tb_test":          ["tb_test"],
    "specialist_types": ["specialist_types"],
    "quantity_limits":  ["quantity_limits"],
}

# Maps each group to the model to use for that call.
# 70B for groups requiring multi-step reasoning.
# 8B for groups requiring simple value extraction.
PARAMETER_MODELS: dict[str, str] = {
    "step_therapy":     MODEL_70B,  # union/OR logic, holistic reasoning
    "duration":         MODEL_8B,   # numeric lookup
    "reauth":           MODEL_70B,  # requirements text can be nuanced
    "age":              MODEL_8B,   # simple value
    "tb_test":          MODEL_8B,   # Yes/No
    "specialist_types": MODEL_8B,   # simple value
    "quantity_limits":  MODEL_70B,   # table lookup
}


# ---------------------------------------------------------------------------
# Stage 2 hard page caps per model tier.
#
# 8B has a hard 6,000 TPM limit — template overhead is ~1,400 tokens,
# so 8 pages (≈4,400 content tokens) is the safe ceiling.
# 70B has a 12,000 TPM limit — 25 pages leaves comfortable headroom for
# most documents; increased from 15 to capture criteria deep in large docs.
# When Stage 2 returns more pages than the cap, keep the first half and
# last half of the sorted page list. PA docs front-load clinical criteria
# and back-load renewal/QL sections, so this strategy preserves the most
# diagnostically useful pages.
# ---------------------------------------------------------------------------

STAGE2_MAX_PAGES_8B: int  = 8
STAGE2_MAX_PAGES_70B: int = 25   # increased from 15 to handle large documents

# Inter-call delay between consecutive LLM calls within a single row.
# Groq: 4 s to avoid per-minute token exhaustion.
# OpenRouter: 0 s — billed per token, no per-minute hard wall.
INTER_CALL_DELAY_GROQ: int        = 4
INTER_CALL_DELAY_OPENROUTER: int  = 0

# ---------------------------------------------------------------------------
# FDA-approved minimum age for plaque psoriasis, by brand.
# Populate these manually after researching each drug's prescribing information.
# ---------------------------------------------------------------------------

FDA_LABELED_AGE: dict[str, str] = {
    "TREMFYA":   ">=6",    # pediatric ≥6 years (≥40 kg) approved September 29, 2025
    "STELARA":   ">=6",    # pediatric ≥6 years
    "AMJEVITA":  ">=18",   # adults only for plaque psoriasis
    "COSENTYX":  ">=6",    # pediatric ≥6 years
    "ENBREL":    ">=4",    # pediatric ≥4 years
    "REMICADE":  ">=18",   # adults only
    "SILIQ":     ">=18",   # adults only
    "CIMZIA":    ">=18",   # adults only
    "BIMZELX":   ">=18",   # adults only
    "SKYRIZI":   ">=18",   # adults only
    "OTEZLA":    ">=6",    # pediatric ≥6 years + ≥20 kg weight requirement
    "YESINTEK":  ">=6",    # biosimilar of Stelara
    "OTULFI":    ">=6",    # biosimilar of Stelara
    "ILUMYA":    ">=18",   # adults only
    "ACITRETIN": ">=18",   # adults only
}

# ---------------------------------------------------------------------------
# FDA baseline values for Access Score calculation.
#
# The Access Score measures how much more restrictive a PA policy is
# compared to what FDA approval implies. FDA approval = no additional
# barriers beyond clinical indication and age. Insurers add barriers on top.
#
# "0" for step counts = FDA requires no step therapy.
# "No" for reauth = FDA does not require periodic reauthorization.
# "NA" for durations = FDA does not impose PA time limits.
# "Yes" for TB test = FDA prescribing information for biologics includes
#   TB screening as a required precaution. Both FDA and insurer agree here,
#   so this does not reduce the access score.
# Age is brand-specific — use FDA_LABELED_AGE[brand] for comparisons.
# ---------------------------------------------------------------------------

FDA_BASELINE: dict[str, str] = {
    "num_steps_brands":        "NA",     # FDA requires no branded step therapy
    "num_steps_generic":       "NA",     # FDA requires no generic step therapy
    "step_phototherapy":       "No",    # FDA does not require phototherapy
    "tb_test":                 "Yes",   # FDA label includes TB screening
    "initial_auth_duration":   "NA",    # FDA imposes no PA duration limits
    "reauth_duration":         "NA",    # FDA imposes no reauth duration limits
    "reauth_required":         "No",    # FDA does not require periodic reauth
    "specialist_types":        "NA",    # FDA does not require specialist prescribing
    "quantity_limits":         "NA",    # FDA does not impose quantity limits
    # step_therapy_text and reauth_requirements_text are text fields —
    # scored on the countable parameters above, not on the text itself.
}
