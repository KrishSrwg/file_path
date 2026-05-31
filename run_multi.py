"""
run_multi.py — Run the pipeline N times with RETRY_ALL_NA_MODE, then build consensus.

Each run retries cells where the previous cached response was all-NA.
After all runs complete, consensus.py is called automatically on all result files.

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

# ── Locate project root (same directory as this script) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


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
        logger.warning("No new result CSV found after run — pipeline may have overwritten an existing file")
        return None

    produced = sorted(new_files)[0]
    logger.info("Run produced: %s", produced.name)
    return produced


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
        else:
            logger.warning("Run %d did not produce a detectable output file", run_num)

    logger.info("All %d runs complete.", args.runs)
    logger.info("Produced: %s", [p.name for p in produced_files])

    if args.no_consensus:
        logger.info("--no-consensus set, skipping consensus step.")
        return

    # Build consensus
    consensus_script = PROJECT_ROOT / "consensus.py"
    if not consensus_script.exists():
        logger.error(
            "consensus.py not found at %s — skipping consensus. "
            "Run manually: python consensus.py <csv1> <csv2> ...",
            consensus_script,
        )
        return

    # Collect all CSVs: existing (user-specified) + newly produced
    all_csvs: list[str] = []
    for existing in args.include_existing:
        p = Path(existing)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            all_csvs.append(str(p))
        else:
            logger.warning("Existing CSV not found: %s — skipping", p)

    all_csvs.extend(str(p) for p in produced_files)

    if len(all_csvs) < 2:
        logger.warning(
            "Only %d CSV(s) available for consensus — need at least 2 for meaningful voting. "
            "Pass existing files with --include-existing.",
            len(all_csvs),
        )
        if len(all_csvs) == 0:
            return

    output_path = PROJECT_ROOT / args.consensus_output
    logger.info("Running consensus on %d files → %s", len(all_csvs), output_path.name)

    subprocess.run(
        [
            sys.executable,
            str(consensus_script),
            *all_csvs,
            "--output", str(output_path),
            "--min-votes", str(args.min_votes),
        ],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()