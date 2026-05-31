"""
run_pipeline.py — Entry point for the PA extraction pipeline.

Reads (filename, brand) pairs from the Submissions tab of PA_Business_Rules.xlsx,
runs PDF extraction and LLM field extraction for each row, and writes the results
to an auto-incremented result_vN.csv (e.g. result_v3.csv if result_v2.csv exists).

Usage:
    python run_pipeline.py                    # normal run, uses/writes LLM cache
    RETRY_NA_MODE=true python run_pipeline.py # re-runs only NoneType-cached rows
    USE_OPENROUTER=true python run_pipeline.py # use OpenRouter instead of Groq

-------------------------------------------------------------------------------
FINAL SUBMISSION NOTES — GROQ TOKEN STRATEGY
(Do NOT implement yet — apply in submission.py when switching to Groq)
-------------------------------------------------------------------------------

These changes are required before the final hackathon submission run on Groq.
Reference this block if the conversation history is lost.

1. SWITCH PROVIDER
   Set USE_OPENROUTER=false (or remove the env var). The pipeline defaults to Groq.

2. RESTRUCTURE 7 LLM CALLS → 3 CALLS PER ROW  (config.py change)
   Current groups: step_therapy, duration, reauth, age, tb_test, specialist_types,
                   quantity_limits  (7 calls × model-specific delay = very slow)
   New groups:
     - step_therapy  (70B): step_therapy_text, num_steps_brands, num_steps_generic,
                            step_phototherapy
     - reauth        (70B): reauth_required, reauth_requirements_text
     - simple_params (8B):  age, initial_auth_duration, reauth_duration,
                            tb_test, specialist_types, quantity_limits

   Implementation: update PARAMETER_GROUPS and PARAMETER_MODELS in config.py.
   Create src/prompts/simple_params.xml (combined 8B prompt for 6 parameters).

3. REDUCE PAGE CAPS  (config.py change)
   STAGE2_MAX_PAGES_70B  : 15 → 6   (saves ~4000 tokens per 70B call)
   STAGE2_MAX_PAGES_8B   :  8 → 2   (saves ~2000 tokens per 8B call)
   STAGE2_MAX_PAGES_SIMPLE: 4        (new: for combined simple_params call)
   Also add to extractor.py: use STAGE2_MAX_PAGES_SIMPLE when group == "simple_params"

4. PER-MODEL INTER-CALL DELAYS  (config.py + extractor.py change)
   Replace single INTER_CALL_DELAY_GROQ = 4 with:
     INTER_CALL_DELAY_GROQ_70B = 55   # seconds  (12000 TPM / 5000 tokens/call ≈ 1.4/min)
     INTER_CALL_DELAY_GROQ_8B  = 35   # seconds  (6000 TPM / 3000 tokens/call ≈ 2/min)
   In extractor.py, choose delay based on group_model at sleep time.

5. ADD _strip_boilerplate()  (extractor.py change)
   Apply before _pages_to_text() to strip references, policy history, clinical
   background — typically cuts token usage by 30–60% on large documents.
   Strip sections containing: "references", "bibliography", "revision history",
   "policy history", "clinical evidence", "background information",
   "instructions for use", "et al.", "doi:", "pubmed", academic journal names.

6. COMPRESS XML PROMPTS  (prompts/ change)
   All 7 existing prompts + new simple_params.xml must target < 800 tokens each.
   Remove verbose examples, condense multi-sentence rules to 2-3 sentences.

7. TOKEN BUDGET CHECK FOR 10 TEST FILES
   70B: 10 files × 2 calls × 5000 tokens = 100,000  (TPD limit = 100K ✓ barely)
   8B:  10 files × 1 call  × 3000 tokens =  30,000  (TPD limit = 500K ✓)
   Total run time estimate: ~25 minutes with per-model sleep delays.

8. NEW simple_params.xml PROMPT STRUCTURE
   Single 8B call extracting all 6 simple parameters from a combined page set.
   Page set = union of keywords from age + duration + tb_test + specialist + QL groups,
   capped at STAGE2_MAX_PAGES_SIMPLE = 4 pages.

-------------------------------------------------------------------------------
"""

import logging
import re
import time
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, DATA_DIR, PDF_DIR, DEFAULT_MODEL
from src.extractor import extract_fields, EXPECTED_KEYS
from src.pdf_reader import extract_text
from src.normalizer import normalize_record
from access_score import compute_access_score

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

KEY_TO_COLUMN: dict[str, str] = {
    "age": "Age",
    "step_therapy_text": "Step Therapy Requirements Documented in Policy",
    "num_steps_brands": "Number of Steps through Brands",
    "num_steps_generic": "Number of Steps through Generic",
    "step_phototherapy": "Step through-phototherapy",
    "tb_test": "TB Test required",
    "quantity_limits": "Quantity Limits",
    "specialist_types": "Specialist Types",
    "initial_auth_duration": "Initial Authorization Duration(in-months)",
    "reauth_duration": "Reauthorization Duration(in-months)",
    "reauth_required": "Reauthorization Required",
    "reauth_requirements_text": "Reauthorization Requirements Documented in Policy",
}

'''BASELINE_ROWS: list[tuple[str, str]] = [
    ("361486-4654549.pdf", "TREMFYA"),
    ("337814-4985370.pdf", "STELARA"),
    ("176207-4867884.pdf", "TREMFYA"),
    ("148593-4960549.pdf", "STELARA"),
    ("247339-4770410.pdf", "TREMFYA"),
    ("269830-4591538.pdf", "STELARA"),
    ("9023-4381765.pdf", "ENBREL"),
    ("8898-4735285.pdf", "COSENTYX"),
    ("195158-4643510.pdf", "SKYRIZI"),
    ("66156-4274314.pdf", "CIMZIA"),
]'''



SUBMISSIONS_XLSX: Path = DATA_DIR / "PA_Business_Rules.xlsx"

def read_submission_rows() -> list[tuple[str, str]]:
    """Read (filename, brand) pairs from the Submissions tab of the business rules Excel.
    
    This is the single source of truth for which rows to process.
    Any changes to the Submissions tab are automatically picked up.
    """
    df = pd.read_excel(SUBMISSIONS_XLSX, sheet_name="Submissions")
    return list(zip(df["Filename"], df["Brand"]))

def _next_result_csv(project_root: Path) -> Path:
    """Return the next auto-incremented result CSV path.

    Scans project_root for result_vN.csv files and returns result_v(N+1).csv.
    Falls back to result_v3.csv if no versioned files exist yet.

    This means every pipeline run automatically produces a new file —
    no manual renaming required between RETRY_NA_MODE iterations.
    """
    existing = [
        p for p in project_root.glob("result_v*.csv")
        if re.search(r"v(\d+)", p.stem)
    ]
    if existing:
        last_n = max(int(re.search(r"v(\d+)", p.stem).group(1)) for p in existing)
        return project_root / f"result_v{last_n + 1}.csv"
    return project_root / "result_v3.csv"


OUTPUT_CSV: Path = _next_result_csv(PROJECT_ROOT)

# Column order in the output CSV matches the Submissions tab layout.
OUTPUT_COLUMNS: list[str] = [
    "Filename",
    "Brand",
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-phototherapy",
    "TB Test required",
    "Quantity Limits",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Access Score",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def process_row(filename: str, brand: str, model: str = DEFAULT_MODEL) -> dict[str, str]:
    """Run the pipeline for a single (filename, brand) pair.

    Returns a dict with exactly the 12 keys in EXPECTED_KEYS. On any error,
    returns an all-NA dict and logs the failure. Never raises.
    """
    try:
        pdf_path = PDF_DIR / filename
        pages = extract_text(pdf_path, use_cache=True)
        raw = extract_fields(pages, brand)
        normalized = normalize_record(raw, brand=brand)

        # Compute Access Score (13th parameter) from normalized extraction values.
        # Pure Python function — no LLM call, works identically on OpenRouter and Groq.
        access_score = compute_access_score(normalized, brand)
        normalized["access_score"] = str(access_score)

        # Audit: warn when duration or reauth fields are all-NA — useful for
        # diagnosing missed APPROVAL LENGTH / combined-section extractions.
        dur_fields = ["initial_auth_duration", "reauth_duration",
                      "reauth_required", "reauth_requirements_text"]
        if all(normalized.get(f, "NA") == "NA" for f in dur_fields):
            logger.warning(
                "All duration/reauth fields are NA for %s | %s — "
                "check APPROVAL LENGTH / Quantity Limits and duration sections",
                filename, brand,
            )
        return normalized
    except Exception as exc:
        logger.error(
            "Failed processing %s | %s — %s: %s",
            filename,
            brand,
            type(exc).__name__,
            exc,
        )
        return {key: "NA" for key in EXPECTED_KEYS}


def write_results(rows: list[dict], output_path: Path) -> None:
    """Write extracted results to CSV with columns matching the Submissions tab order."""
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rows = read_submission_rows()          # ← read from Excel
    total = len(rows)
    logger.info("Starting pipeline run with %d rows from Submissions tab", total)

    results: list[dict] = []

    for i, (filename, brand) in enumerate(rows, start=1):
        logger.info("Row %d/%d: %s | %s", i, total, filename, brand)
        t0 = time.monotonic()

        fields = process_row(filename, brand)

        elapsed = time.monotonic() - t0
        logger.info("Row %d/%d complete in %.1fs", i, total, elapsed)

        row: dict = {"Filename": filename, "Brand": brand}
        row.update({KEY_TO_COLUMN[key]: fields[key] for key in EXPECTED_KEYS})
        row["Access Score"] = fields.get("access_score", "NA")
        results.append(row)

    logger.info("All rows complete. Writing to %s", OUTPUT_CSV)
    write_results(results, OUTPUT_CSV)
    logger.info("Done. Wrote %d rows to %s", len(results), OUTPUT_CSV)
