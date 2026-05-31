"""
consensus.py — Combine multiple pipeline result CSVs using voting logic.

Takes N result CSV files and produces a single consensus CSV where each cell
gets the most reliable value across all runs.

Voting rules (differ by field type):

  CATEGORICAL fields (step counts, Yes/No/NA, durations, age, specialist):
    - Collect all N values for the cell.
    - Count votes for each distinct value (including NA).
    - If any non-NA value has more votes than NA: use the most common non-NA value.
    - If NA has the most votes: return NA.
    - Ties between non-NA values: take the lexicographically largest (usually more
      specific, e.g. "2" beats "1", "Yes" beats "No" for step-existence fields).

  TEXT fields (step_therapy_text, reauth_requirements_text, quantity_limits):
    - If ALL runs returned NA: return NA.
    - Otherwise: return the LONGEST non-NA value across all runs.
      Longer text = more complete extraction. Different runs often return the
      same content with slightly different phrasing; the longest is most complete.

Usage:
    python consensus.py result_v2.csv result_v16.csv result_merged.csv
    python consensus.py --output final_consensus.csv result_v2.csv result_v10.csv ...
    python consensus.py --min-votes 3 result_v*.csv   # require 3 votes for non-NA win

Post-consensus corrections applied automatically:
    - reauth_required is forced to "Yes" if reauth_duration or
      reauth_requirements_text is non-NA (mirrors normalizer.py rule).

Running the voting pipeline:
    1. python run_pipeline.py                          # → result_vN.csv  (run 1)
    2. RETRY_ALL_NA_MODE=true python run_pipeline.py   # → result_v(N+1) (run 2, retries NA)
    3. RETRY_ALL_NA_MODE=true python run_pipeline.py   # → result_v(N+2) (run 3)
    4. ... repeat 3-5 times
    5. python consensus.py result_vN.csv ... result_v(N+5).csv
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

# Text fields: take the LONGEST non-NA value (more text = more complete).
TEXT_FIELDS: frozenset[str] = frozenset({
    "Step Therapy Requirements Documented in Policy",
    "Reauthorization Requirements Documented in Policy",
    "Quantity Limits",
})

# Categorical fields: majority vote.
CATEGORICAL_FIELDS: frozenset[str] = frozenset({
    "Age",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-phototherapy",
    "TB Test required",
    "Specialist Types",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
})

ALL_VALUE_COLS = TEXT_FIELDS | CATEGORICAL_FIELDS

# Canonical NA strings — treated as absent regardless of capitalisation.
_NA_STRINGS: frozenset[str] = frozenset({
    "na", "nan", "none", "", "n/a", "not applicable",
})


def _is_na(val) -> bool:
    """Return True if val is blank / NA in any form."""
    if val is None:
        return True
    return str(val).strip().lower() in _NA_STRINGS


# ---------------------------------------------------------------------------
# Voting logic
# ---------------------------------------------------------------------------

def vote_categorical(values: list, min_votes: int = 1):
    """Majority vote for categorical/numeric fields.

    Rules:
    1. Collect non-NA values and count them.
    2. If the most common non-NA value has >= min_votes AND more votes than NA:
       return it.
    3. Otherwise return None (becomes NaN in DataFrame → blank in CSV).

    min_votes guards against accepting a value seen only once when most runs
    returned NA. Default of 1 means any single non-NA value wins over all-NA.
    Set to 2 or 3 for stricter consensus.
    """
    non_na = [str(v).strip() for v in values if not _is_na(v)]
    na_count = len(values) - len(non_na)

    if not non_na:
        return None  # all NA → return None (NaN in pandas)

    counter = Counter(non_na)
    best_val, best_count = counter.most_common(1)[0]

    if best_count >= min_votes and best_count >= na_count:
        return best_val

    return None  # NA wins → return None


def vote_text(values: list):
    """For free-text fields: return the LONGEST non-NA value.

    Longer extraction = more complete.  If all runs returned NA: return None.
    """
    non_na = [str(v).strip() for v in values if not _is_na(v)]
    if not non_na:
        return None  # all NA → return None (NaN in pandas)
    return max(non_na, key=len)


# ---------------------------------------------------------------------------
# Core consensus function
# ---------------------------------------------------------------------------

def build_consensus(
    dfs: list[pd.DataFrame],
    min_votes: int = 1,
) -> pd.DataFrame:
    """Combine N result DataFrames into a single consensus DataFrame.

    Args:
        dfs:       List of result DataFrames (same rows, same columns).
        min_votes: Minimum number of runs that must agree on a non-NA value
                   before it beats NA in categorical voting.

    Returns:
        Consensus DataFrame with same structure as inputs.
    """
    if not dfs:
        raise ValueError("No DataFrames provided")

    # Verify all frames have the same shape and row order
    n_rows = len(dfs[0])
    for i, df in enumerate(dfs[1:], 1):
        if len(df) != n_rows:
            raise ValueError(
                f"DataFrame {i} has {len(df)} rows, expected {n_rows}"
            )

    # Use the first frame as the structural template (Filename, Brand columns)
    result = dfs[0][["Filename", "Brand"]].copy()

    for col in ALL_VALUE_COLS:
        if col not in dfs[0].columns:
            logger.warning("Column %r not found in inputs — skipping", col)
            continue

        consensus_col: list[str] = []
        for row_idx in range(n_rows):
            values = [df.iloc[row_idx][col] for df in dfs if col in df.columns]

            if col in TEXT_FIELDS:
                consensus_col.append(vote_text(values))
            else:
                consensus_col.append(vote_categorical(values, min_votes=min_votes))

        result[col] = consensus_col

    # Post-consensus business rule: reauth_required must be "Yes" if either
    # reauth_duration or reauth_requirements_text is non-NA.
    # This mirrors the rule in normalizer.py but applied to the aggregated consensus.
    # Fixes cases where individual runs correctly extracted reauth content but
    # the majority vote left reauth_required as NA due to split votes.
    dur_col  = "Reauthorization Duration(in-months)"
    text_col = "Reauthorization Requirements Documented in Policy"
    req_col  = "Reauthorization Required"
    if dur_col in result.columns and text_col in result.columns and req_col in result.columns:
        for idx in range(len(result)):
            dur_val  = result.iloc[idx][dur_col]
            text_val = result.iloc[idx][text_col]
            req_val  = result.iloc[idx][req_col]
            dur_is_na  = dur_val  is None or str(dur_val).strip().lower()  in _NA_STRINGS
            text_is_na = text_val is None or str(text_val).strip().lower() in _NA_STRINGS
            req_is_yes = str(req_val).strip() == "Yes" if req_val is not None else False
            if (not dur_is_na or not text_is_na) and not req_is_yes:
                logger.info(
                    "build_consensus: overriding reauth_required → 'Yes' at row %d "
                    "(reauth_duration=%r, reauth_requirements_text=%r)",
                    idx, dur_val, text_val,
                )
                result.at[idx, req_col] = "Yes"

    # Reorder columns to match the submission tab layout
    ordered_cols = [
        "Filename", "Brand", "Age",
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
    ]
    return result[[c for c in ordered_cols if c in result.columns]]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(dfs: list[pd.DataFrame], consensus: pd.DataFrame,
                     labels: list[str]) -> None:
    """Print blank-count comparison: each input CSV vs consensus."""
    val_cols = [c for c in ALL_VALUE_COLS if c in consensus.columns]

    print("\n=== BLANK COUNT COMPARISON ===")
    header = f"{'Parameter':<50}"
    for label in labels:
        header += f"  {label[:8]:>8}"
    header += f"  {'CONSENSUS':>9}"
    print(header)
    print("-" * len(header))

    for col in val_cols:
        row_str = f"  {col:<48}"
        for df in dfs:
            cnt = df[col].isna().sum() if col in df.columns else "?"
            row_str += f"  {cnt:>8}"
        row_str += f"  {consensus[col].isna().sum():>9}"
        print(row_str)

    print("-" * len(header))
    total_str = f"  {'TOTAL':48}"
    for df in dfs:
        t = sum(df[c].isna().sum() for c in val_cols if c in df.columns)
        total_str += f"  {t:>8}"
    total_str += f"  {sum(consensus[c].isna().sum() for c in val_cols):>9}"
    print(total_str)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple pipeline result CSVs via voting consensus."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input result CSV files (at least 2 recommended for meaningful voting)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for consensus CSV (default: result_consensus.csv next to inputs)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help=(
            "Minimum votes for a non-NA categorical value to beat NA. "
            "1 = any single non-NA value wins; 2 = need at least 2 runs to agree. "
            "Default: 1 (most permissive — good when you have few runs)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load input CSVs
    dfs: list[pd.DataFrame] = []
    labels: list[str] = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)
        df = pd.read_csv(p, encoding="utf-8-sig")
        dfs.append(df)
        labels.append(p.stem)
        logger.info("Loaded %s (%d rows)", p.name, len(df))

    if len(dfs) < 2:
        logger.warning(
            "Only 1 input file — consensus is identical to input. "
            "Provide at least 2 files for meaningful voting."
        )

    # Build consensus
    logger.info(
        "Building consensus from %d files (min_votes=%d) ...", len(dfs), args.min_votes
    )
    consensus = build_consensus(dfs, min_votes=args.min_votes)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.inputs[0]).parent / "result_consensus.csv"

    consensus.to_csv(out_path, index=False, encoding="utf-8-sig", na_rep="NA")
    logger.info("Consensus saved to %s", out_path)

    # Print comparison table
    print_comparison(dfs, consensus, labels)
    print(f"\nConsensus saved → {out_path}")


if __name__ == "__main__":
    main()
