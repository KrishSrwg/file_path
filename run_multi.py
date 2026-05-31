"""
run_multi.py — Run the pipeline N times with RETRY_ALL_NA_MODE, then build consensus.

Each run retries cells where the previous cached response was all-NA.
After all runs complete, consensus.py is called automatically on all result files.

Access Score handling
---------------------
• Every individual run already computes and writes a numeric "Access Score" (0–100)
  column to its result_vN.csv (handled by run_pipeline.py + access_score.py).
• After each run, this script logs an Access Score summary (avg / min / max / coverage)
  so you can monitor extraction quality across runs.
• After the final consensus is built on the 12 extracted parameters, this script
  appends an "Access Score" column to result_consensus.csv where each cell is the
  **integer-rounded average** of that row's Access Score across all input CSVs
  (existing + newly produced).  CSVs that pre-date access_score.py and lack the
  column are silently skipped during averaging.

Usage:
    python run_multi.py              # 5 runs (default), then consensus
    python run_multi.py --runs 3     # 3 runs, then consensus
    python run_multi.py --runs 3 --no-consensus   # 3 runs, skip consensus
    python run_multi.py --include-existing result_v2.csv result_merged.csv --runs 3
        # include existing CSVs when building consensus (recommended)

The final consensus CSV is written to result_consensus.csv in the project root.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Locate project root (same directory as this script) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

# Column name used throughout — must match run_pipeline.py OUTPUT_COLUMNS.
ACCESS_SCORE_COL = "Access Score"


# ---------------------------------------------------------------------------
# Access Score helpers
# ---------------------------------------------------------------------------

def _access_score_stats(csv_path: Path) -> dict | None:
    """Read a result CSV and return Access Score statistics for the run.

    Returns a dict with keys: avg, min, max, valid, total.
    Returns None if the file cannot be read or the column is absent.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if ACCESS_SCORE_COL not in df.columns:
            return None
        scores = pd.to_numeric(df[ACCESS_SCORE_COL], errors="coerce").dropna()
        if scores.empty:
            return None
        return {
            "avg":   round(float(scores.mean()), 1),
            "min":   int(scores.min()),
            "max":   int(scores.max()),
            "valid": int(len(scores)),
            "total": int(len(df)),
        }
    except Exception as exc:
        logger.warning("Could not compute Access Score stats for %s: %s", csv_path.name, exc)
        return None


def _append_averaged_access_score(
    consensus_path: Path,
    all_csv_paths: list[Path],
) -> None:
    """Compute per-row averaged Access Score and write it into the consensus CSV.

    Algorithm:
      1. Load every CSV that contains an "Access Score" column.
      2. All CSVs are assumed to have the same row order (same Submissions tab),
         so column positions correspond 1-to-1.
      3. For each row, compute mean of all valid (non-NA) numeric scores across
         the N CSVs, round to the nearest integer.
      4. Load the consensus CSV, add/overwrite the "Access Score" column, save.

    CSVs without the column (e.g., pre-access-score legacy files) are skipped.
    If NO csv has the column, a warning is logged and the file is left unchanged.
    """
    score_series: list[pd.Series] = []

    for p in all_csv_paths:
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception as exc:
            logger.warning("Could not load %s for Access Score averaging: %s", p.name, exc)
            continue

        if ACCESS_SCORE_COL not in df.columns:
            logger.debug("No '%s' column in %s — skipping for average", ACCESS_SCORE_COL, p.name)
            continue

        numeric = pd.to_numeric(df[ACCESS_SCORE_COL], errors="coerce")
        score_series.append(numeric.reset_index(drop=True))
        logger.debug("  + %s: %d valid scores", p.name, int(numeric.notna().sum()))

    if not score_series:
        logger.warning(
            "No '%s' data found in any of the %d CSVs — "
            "consensus will not include Access Score.",
            ACCESS_SCORE_COL, len(all_csv_paths),
        )
        return

    # Stack into a (rows × n_csv) matrix and take row-wise mean, ignoring NaN.
    scores_matrix = pd.concat(score_series, axis=1)
    avg_scores = scores_matrix.mean(axis=1).round(0)

    # Convert to nullable int so rows with no valid score become NA not NaN.
    avg_scores_int = avg_scores.astype("Int64")

    # Load consensus, inject column, save.
    try:
        consensus_df = pd.read_csv(consensus_path, encoding="utf-8-sig")
    except Exception as exc:
        logger.error("Cannot load consensus file %s: %s", consensus_path.name, exc)
        return

    # Write as string; replace pandas "<NA>" token with "NA" for CSV cleanliness.
    consensus_df[ACCESS_SCORE_COL] = avg_scores_int.astype(str).str.replace("<NA>", "NA")

    try:
        consensus_df.to_csv(consensus_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        logger.error("Failed to save updated consensus: %s", exc)
        return

    valid_count = int(avg_scores.notna().sum())
    grand_avg   = float(avg_scores.dropna().mean()) if valid_count else float("nan")
    logger.info(
        "Access Score averaged from %d CSV(s) → written to %s  "
        "(avg=%.1f, valid=%d/%d rows)",
        len(score_series), consensus_path.name,
        grand_avg, valid_count, len(consensus_df),
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_once(env: dict) -> Path | None:
    """Run run_pipeline.py once and return the path of the CSV it produced.

    Determines the output path by scanning for the highest-numbered
    result_vN.csv before and after the run.
    """
    def _existing_results() -> set[Path]:
        return set(PROJECT_ROOT.glob("result_v*.csv"))

    before = _existing_results()

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_pipeline.py")],
        env=env,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        logger.error("run_pipeline.py exited with code %d", result.returncode)
        return None

    after = _existing_results()
    new_files = after - before
    if not new_files:
        logger.warning(
            "No new result CSV found after run — "
            "pipeline may have overwritten an existing file"
        )
        return None

    produced = sorted(new_files)[0]
    logger.info("Run produced: %s", produced.name)
    return produced


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pipeline N times with RETRY_ALL_NA_MODE, then build consensus."
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=5,
        help="Number of pipeline runs (default: 5)",
    )
    parser.add_argument(
        "--no-consensus",
        action="store_true",
        help="Skip consensus step after runs (just collect CSVs)",
    )
    parser.add_argument(
        "--include-existing",
        nargs="*",
        metavar="CSV",
        default=[],
        help=(
            "Existing result CSVs to include in the final consensus "
            "(e.g. result_v2.csv result_merged.csv). "
            "These are NOT re-run; they just feed into consensus voting."
        ),
    )
    parser.add_argument(
        "--consensus-output",
        default="result_consensus.csv",
        help="Output filename for consensus CSV (default: result_consensus.csv)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="Minimum votes for categorical consensus (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Build env: inherit current env + set RETRY_ALL_NA_MODE=true
    env = os.environ.copy()
    env["RETRY_ALL_NA_MODE"] = "true"

    logger.info("Starting %d pipeline run(s) with RETRY_ALL_NA_MODE=true", args.runs)
    logger.info("Project root: %s", PROJECT_ROOT)

    produced_files: list[Path] = []

    for run_num in range(1, args.runs + 1):
        logger.info("=" * 60)
        logger.info("RUN %d / %d", run_num, args.runs)
        logger.info("=" * 60)

        csv_path = run_pipeline_once(env)

        if csv_path:
            produced_files.append(csv_path)

            # ── Per-iteration Access Score summary ────────────────────────────
            stats = _access_score_stats(csv_path)
            if stats:
                logger.info(
                    "RUN %d/%d Access Score → avg=%.1f  min=%d  max=%d  "
                    "(%d/%d rows with valid score)",
                    run_num, args.runs,
                    stats["avg"], stats["min"], stats["max"],
                    stats["valid"], stats["total"],
                )
            else:
                logger.info(
                    "RUN %d/%d Access Score → column absent or all-NA in %s",
                    run_num, args.runs, csv_path.name,
                )
        else:
            logger.warning("Run %d did not produce a detectable output file", run_num)

    logger.info("All %d runs complete.", args.runs)
    logger.info("Produced: %s", [p.name for p in produced_files])

    if args.no_consensus:
        logger.info("--no-consensus set, skipping consensus step.")
        return

    # ── Build consensus on the 12 extracted parameters ───────────────────────
    consensus_script = PROJECT_ROOT / "consensus.py"
    if not consensus_script.exists():
        logger.error(
            "consensus.py not found at %s — skipping consensus. "
            "Run manually: python consensus.py <csv1> <csv2> ...",
            consensus_script,
        )
        return

    # Collect all CSVs for voting: existing (user-specified) + newly produced.
    all_csv_paths: list[Path] = []
    for existing in args.include_existing:
        p = Path(existing)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            all_csv_paths.append(p)
        else:
            logger.warning("Existing CSV not found: %s — skipping", p)

    all_csv_paths.extend(produced_files)

    all_csvs_str = [str(p) for p in all_csv_paths]

    if len(all_csvs_str) < 2:
        logger.warning(
            "Only %d CSV(s) available for consensus — need at least 2 for meaningful voting. "
            "Pass existing files with --include-existing.",
            len(all_csvs_str),
        )
        if not all_csvs_str:
            return

    output_path = PROJECT_ROOT / args.consensus_output
    logger.info(
        "Running consensus on %d files → %s",
        len(all_csvs_str), output_path.name,
    )

    subprocess.run(
        [
            sys.executable,
            str(consensus_script),
            *all_csvs_str,
            "--output", str(output_path),
            "--min-votes", str(args.min_votes),
        ],
        cwd=str(PROJECT_ROOT),
    )

    # ── Append averaged Access Score to the consensus CSV ────────────────────
    if output_path.exists():
        logger.info(
            "Computing averaged Access Score from %d CSV(s) → %s",
            len(all_csv_paths), output_path.name,
        )
        _append_averaged_access_score(output_path, all_csv_paths)
    else:
        logger.warning(
            "Consensus file not found at %s — cannot append Access Score average.",
            output_path,
        )


if __name__ == "__main__":
    main()
