#!/usr/bin/env python3
"""Citation evaluation: F1-based citation key matching between GT and Pred LaTeX."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from paper_recon.common.log import get_logger

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

CITATION_COMMAND_PATTERN = re.compile(
    r"""
    \\(?P<command>[A-Za-z]*cite[A-Za-z]*\*?)
    (?:\s*\[[^\]]*\]){0,2}
    \s*\{(?P<keys>[^}]*)\}
    """,
    flags=re.VERBOSE | re.DOTALL,
)

FILECONTENTS_BIB_PATTERN = re.compile(
    r"\\begin\{filecontents\*?\}\{(?P<name>[^}]+\.bib)\}(?P<body>.*?)\\end\{filecontents\*?\}",
    flags=re.DOTALL,
)


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------


class CitationSectionEvaluation(TypedDict, total=True):
    gt_section_name: str
    pred_section_name: str
    citation_quality: int
    citation_hallucination: int
    citation_coverage: int
    citation_relevance: int
    key_issues: list[str]
    reasoning: str
    not_applicable: bool


class CitationF1Result(TypedDict, total=True):
    gt_keys: list[str]
    pred_keys: list[str]
    common_keys: list[str]
    missing_keys: list[str]  # In GT but not in Pred
    extra_keys: list[str]  # In Pred but not in GT
    hallucinated_keys: list[str]  # In Pred but not in bib
    precision: float
    recall: float
    f1: float


class CitationSummary(TypedDict, total=True):
    total_sections: int
    evaluated_sections: int
    average_citation_quality: float
    average_citation_hallucination: float
    average_citation_coverage: float
    average_citation_relevance: float
    # F1-based metrics
    citation_f1: CitationF1Result | None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def default_citation_summary(total_sections: int) -> CitationSummary:
    return {
        "total_sections": total_sections,
        "evaluated_sections": 0,
        "average_citation_quality": 0.0,
        "average_citation_hallucination": 0.0,
        "average_citation_coverage": 0.0,
        "average_citation_relevance": 0.0,
        "citation_f1": None,
    }


def _extract_all_citation_keys(latex_path: Path) -> set[str]:
    """Extract all citation keys from a LaTeX file."""
    text = latex_path.read_text(encoding="utf-8", errors="replace")
    keys: set[str] = set()
    for match in CITATION_COMMAND_PATTERN.finditer(text):
        command = match.group("command")
        if command.lower().startswith("nocite"):
            continue
        for raw_key in match.group("keys").split(","):
            stripped = raw_key.strip()
            if stripped:
                keys.add(stripped)
    return keys


def _extract_bib_keys(latex_path: Path) -> set[str]:
    """Extract entry keys from bib files associated with a LaTeX file."""
    bib_pattern = re.compile(r"@\w+\{([^,]+),")
    keys: set[str] = set()

    # Search for bib entries in filecontents
    text = latex_path.read_text(encoding="utf-8", errors="replace")
    for match in FILECONTENTS_BIB_PATTERN.finditer(text):
        for bib_match in bib_pattern.finditer(match.group("body")):
            keys.add(bib_match.group(1).strip())

    # Search for references.bib in the same directory
    bib_file = latex_path.parent / "references.bib"
    if bib_file.exists():
        bib_text = bib_file.read_text(encoding="utf-8", errors="replace")
        for match in bib_pattern.finditer(bib_text):
            keys.add(match.group(1).strip())

    return keys


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_citation_f1(
    gt_latex_path: Path,
    pred_latex_path: Path,
) -> CitationF1Result:
    """Compare GT and Pred citation keys and compute F1 score."""
    gt_keys = _extract_all_citation_keys(gt_latex_path)
    pred_keys = _extract_all_citation_keys(pred_latex_path)
    bib_keys = _extract_bib_keys(pred_latex_path)

    common = gt_keys & pred_keys
    missing = gt_keys - pred_keys
    extra = pred_keys - gt_keys
    hallucinated = pred_keys - bib_keys

    precision = len(common) / len(pred_keys) if pred_keys else 0.0
    recall = len(common) / len(gt_keys) if gt_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info("Citation F1: precision=%.2f, recall=%.2f, f1=%.2f", precision, recall, f1)
    logger.info(
        "  GT keys: %d, Pred keys: %d, Common: %d", len(gt_keys), len(pred_keys), len(common)
    )
    logger.info(
        "  Missing: %d, Extra: %d, Hallucinated: %d", len(missing), len(extra), len(hallucinated)
    )
    if hallucinated:
        logger.warning("  Hallucinated citation keys: %s", sorted(hallucinated))

    return {
        "gt_keys": sorted(gt_keys),
        "pred_keys": sorted(pred_keys),
        "common_keys": sorted(common),
        "missing_keys": sorted(missing),
        "extra_keys": sorted(extra),
        "hallucinated_keys": sorted(hallucinated),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
