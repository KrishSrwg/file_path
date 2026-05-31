"""
pdf_reader.py — Extract text from PA policy PDFs with page-level granularity.

Results are cached to EXTRACTED_TEXT_DIR so the pipeline can iterate without
re-parsing. Call extract_text() for a single PDF; it returns a dict of
1-indexed page numbers to extracted text strings.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

from src.config import EXTRACTED_TEXT_DIR

logger = logging.getLogger(__name__)


def extract_text(pdf_path: Path, use_cache: bool = False) -> dict[int, str]:
    """
    Extract text from a PDF, returning a mapping of 1-indexed page number to text.

    Blank pages and scanned-image pages are included as empty strings — callers
    decide what to do with them. Results are cached to EXTRACTED_TEXT_DIR as JSON
    so subsequent calls with the same file skip re-parsing.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        use_cache: If True, return cached result when available; write cache on miss.

    Returns:
        Dict mapping 1-indexed page numbers (int) to extracted text (str).

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EXTRACTED_TEXT_DIR / f"{pdf_path.stem}.json"

    if use_cache and cache_path.exists():
        logger.info("Serving from cache: %s", cache_path.name)
        return _load_cache(cache_path)

    return _extract_and_cache(pdf_path, cache_path)


def _load_cache(cache_path: Path) -> dict[int, str]:
    """Read a cache file and return pages with int keys."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["pages"].items()}


def _extract_and_cache(pdf_path: Path, cache_path: Path) -> dict[int, str]:
    """Run pdfplumber extraction, write cache, and return pages dict."""
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", pdf_path.name, exc)
        _write_cache(cache_path, pages={})
        return {}

    pages: dict[int, str] = {}
    with pdf:
        page_count = len(pdf.pages)
        logger.info("Extracting %s (%d pages)", pdf_path.name, page_count)

        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Page %d extraction failed in %s: %s", i, pdf_path.name, exc)
                text = ""
            pages[i] = text

    _write_cache(cache_path, pages=pages)
    return pages


def _write_cache(cache_path: Path, pages: dict[int, str]) -> None:
    """Serialise extraction results to disk."""
    payload = {
        "page_count": len(pages),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pages": {str(k): v for k, v in pages.items()},
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
