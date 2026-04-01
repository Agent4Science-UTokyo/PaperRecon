"""Table extraction, matching, and evaluation between GT and Pred LaTeX files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field

from paper_recon.common.config import LLMConfig
from paper_recon.common.llm import get_response_from_llm
from paper_recon.common.log import get_logger

logger = get_logger(__file__)


def _consume_balanced_braces(text: str, start_idx: int) -> tuple[str, int]:
    """
    Extract text between balanced braces starting at start_idx.

    This is also used by citation parsing, so it is re-exported from here
    and imported by evaluate_per_section.
    """
    if text[start_idx] != "{":
        return "", start_idx
    depth = 0
    i = start_idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1 : i], i + 1
        i += 1
    return text[start_idx + 1 :], i


@dataclass
class Table:
    caption: str
    label: str | None
    content: str
    tabular: str


class TableMatchResponse(BaseModel):
    matched_index: int
    confidence: float
    reasoning: str


class TableComparisonResponse(BaseModel):
    reasoning: str
    match_score: int = Field(ge=1, le=5, description="1=completely different, 5=exact match")
    numerical_match: bool
    structure_match: bool
    differences: list[str]


class TableComparison(TypedDict, total=True):
    gt_caption: str
    gt_label: str | None
    pred_caption: str
    pred_label: str | None
    match_score: float
    numerical_match: float
    structure_match: float
    differences: list[str]
    reasoning: str


class TableSummary(TypedDict, total=True):
    total_gt_tables: int
    total_pred_tables: int
    compared_tables: int
    average_match_score: float


def extract_tables_from_latex(latex_path: Path) -> list[Table]:
    content = latex_path.read_text(encoding="utf-8")

    tables = []
    table_pattern = r"\\begin\s*\{(table\*?|wraptable)\}"
    table_matches = list(re.finditer(table_pattern, content))

    for i, match in enumerate(table_matches):
        start_pos = match.start()
        env_name = match.group(1)
        end_pattern = rf"\\end\s*\{{{re.escape(env_name)}\}}"
        end_match = re.search(end_pattern, content[start_pos:])
        if end_match:
            end_pos = start_pos + end_match.end()
            table_content = content[start_pos:end_pos]

            caption_cmd = re.search(r"\\caption\s*\{", table_content)
            if caption_cmd:
                brace_start = caption_cmd.end() - 1
                caption, _ = _consume_balanced_braces(table_content, brace_start)
                caption = re.sub(r"\\label\s*\{[^}]+\}", "", caption).strip()
            else:
                caption = f"Table {i + 1}"

            label_match = re.search(r"\\label\{([^}]+)\}", table_content)
            label = label_match.group(1).strip() if label_match else None

            tabular_pattern = r"\\begin\{(tabular[xy]?|tblr)\}.*?\\end\{(\1)\}"
            tabular_match = re.search(tabular_pattern, table_content, re.DOTALL)
            tabular_content = tabular_match.group(0) if tabular_match else ""

            tables.append(
                Table(caption=caption, label=label, content=table_content, tabular=tabular_content)
            )

    return tables


def match_table_labels(
    gt_table: Table, pred_tables: list[Table], used_pred_indices: set[int]
) -> int | None:
    gt_label = gt_table.label
    for pred_i, pred_table in enumerate(pred_tables):
        if pred_i in used_pred_indices:
            continue
        pred_label = pred_table.label
        if gt_label and pred_label and gt_label == pred_label:
            return pred_i
    return None


def match_table_captions(
    gt_table: Table, pred_tables: list[Table], used_pred_indices: set[int]
) -> int | None:
    gt_caption = gt_table.caption
    for pred_i, pred_table in enumerate(pred_tables):
        if pred_i in used_pred_indices:
            continue
        pred_caption = pred_table.caption
        if gt_caption and pred_caption and gt_caption[:50].lower() == pred_caption[:50].lower():
            return pred_i
    return None


def match_tables_with_llm(
    gt_table: Table,
    pred_tables: list[Table],
    used_pred_indices: set[int],
    config: LLMConfig,
) -> int | None:
    """Use an LLM to find the Pred table corresponding to a GT table."""
    if not pred_tables:
        logger.warning("No predicted tables available for matching")
        return None

    system_prompt = (
        "You are an expert paper reviewer. Your task is to find which "
        "predicted table corresponds to a given ground truth table.\n\n"
        "Compare the ground truth table with all predicted tables and "
        "determine which one (if any) represents the same information.\n\n"
        "Respond in JSON format:\n"
        "```json\n"
        "{\n"
        '  "matched_index": <0-based index of matched table, or -1 if none>,\n'
        '  "confidence": <0-100>,\n'
        '  "reasoning": "<explanation>"\n'
        "}\n"
        "```"
    )
    pred_candidate_tables = [t for i, t in enumerate(pred_tables) if i not in used_pred_indices]

    candidate_index_to_original_index: dict[int, int] = {}
    candidate_i = 0
    for i in range(len(pred_tables)):
        if i in used_pred_indices:
            continue
        candidate_index_to_original_index[candidate_i] = i
        candidate_i += 1

    pred_tables_desc = []
    for i, pred_table in enumerate(pred_candidate_tables):
        pred_tables_desc.append(
            f"Table {i}:\n"
            f"  Caption: {pred_table.caption}\n"
            f"  Label: {pred_table.label}\n"
            f"  Content preview: {pred_table.tabular[:500]}\n"
        )

    user_prompt = (
        f"Find which predicted table matches this ground truth table:\n\n"
        f"**Ground Truth Table:**\n"
        f"Caption: {gt_table.caption}\n"
        f"Label: {gt_table.label}\n"
        f"Content preview: {gt_table.tabular[:500]}\n\n"
        f"**Available Predicted Tables:**\n"
        f"{''.join(pred_tables_desc)}\n\n"
        f"Return the index (0-based) of the matched table, or -1 if none match."
    )

    response, _ = get_response_from_llm(
        model=config.model,
        user_message=user_prompt,
        system_message=system_prompt,
        temperature=config.temp,
        response_format=TableMatchResponse,
    )

    matched_idx = response.matched_index
    confidence = response.confidence

    if matched_idx < 0:
        logger.info("No matching table found according to LLM response")
        return None

    if matched_idx >= len(pred_candidate_tables):
        logger.warning("Matched index %d is out of range for predicted tables", matched_idx)
        return None

    if confidence < 50:
        logger.warning(
            "LLM found potential match at index %d but confidence (%f) is too low (< 50)",
            matched_idx,
            confidence,
        )
        return None

    logger.info("Match found! Index %d with confidence %f", matched_idx, confidence)
    return candidate_index_to_original_index[matched_idx]


def compare_tables(
    config: LLMConfig, gt_table: Table, pred_table: Table
) -> TableComparisonResponse:
    """Compare two tables and evaluate their degree of match."""
    system_prompt = (
        "You are an expert paper reviewer evaluating whether two LaTeX "
        "tables contain the same information.\n\n"
        "Your task is to evaluate:\n"
        "1. **Numerical values**: Are all numerical values identical?\n"
        "2. **Table structure**: Are the rows and columns the same?\n"
        "3. **Content consistency**: Are the labels, descriptions, and "
        "other text content consistent?\n\n"
        "Score the match on a 1-5 scale:\n"
        "5: Tables are identical or have only trivial formatting differences.\n"
        "4: Tables have the same structure and most values match, minor differences.\n"
        "3: Tables share the same topic but have notable differences in values or structure.\n"
        "2: Tables are loosely related but have major differences in content or structure.\n"
        "1: Tables are completely different or unrelated.\n\n"
        "Respond in JSON format:\n"
        "```json\n"
        "{\n"
        '  "match_score": <1-5>,\n'
        '  "numerical_match": <true/false>,\n'
        '  "structure_match": <true/false>,\n'
        '  "differences": ["<difference1>", "<difference2>", ...],\n'
        '  "reasoning": "<explanation>"\n'
        "}\n"
        "```"
    )

    user_prompt = (
        f"Compare the following two tables:\n\n"
        f"**Ground Truth Table:**\n"
        f"Caption: {gt_table.caption}\n"
        f"Label: {gt_table.label}\n"
        f"Content:\n{gt_table.tabular[:3000]}\n\n"
        f"**Predicted Table:**\n"
        f"Caption: {pred_table.caption}\n"
        f"Label: {pred_table.label}\n"
        f"Content:\n{pred_table.tabular[:3000]}\n\n"
        f"Evaluate if these tables match."
    )

    response, _ = get_response_from_llm(
        model=config.model,
        user_message=user_prompt,
        system_message=system_prompt,
        temperature=config.temp,
        response_format=TableComparisonResponse,
    )

    return response


def matching_and_evaluate_tables(
    gt_latex_path: Path, pred_latex_path: Path, config: LLMConfig
) -> tuple[list[TableComparison], TableSummary]:
    logger.info("=" * 80)
    logger.info("Extracting tables...")
    gt_tables = extract_tables_from_latex(gt_latex_path)
    pred_tables = extract_tables_from_latex(pred_latex_path)

    logger.info("Found %d tables in GT", len(gt_tables))
    for gt_i, table in enumerate(gt_tables):
        logger.debug("\t%d: caption='%s...', label='%s'", gt_i + 1, table.caption[:50], table.label)

    logger.info("Found %d tables in Pred", len(pred_tables))
    for pred_i, table in enumerate(pred_tables):
        logger.debug(
            "\t%d: caption='%s...', label='%s'", pred_i + 1, table.caption[:50], table.label
        )

    comparison_results: list[TableComparison] = []
    total_match_score = 0.0
    compared_count = 0
    used_pred_indices: set[int] = set()

    for gt_i, gt_table in enumerate(gt_tables):
        gt_label = gt_table.label
        gt_caption = gt_table.caption

        logger.info("=" * 80)
        logger.info("Processing GT table %d/%d", gt_i + 1, len(gt_tables))
        logger.info("GT caption: '%s'", gt_caption[:100])
        logger.info("GT label: '%s'", gt_label)
        logger.info("=" * 80)

        matched_idx = match_table_labels(gt_table, pred_tables, used_pred_indices)

        if matched_idx is None:
            logger.info("Failed to match by label. Trying to match by caption...")
            matched_idx = match_table_captions(gt_table, pred_tables, used_pred_indices)

        if matched_idx is None:
            logger.info("Failed to match by caption. Trying to match with LLM...")
            matched_idx = match_tables_with_llm(
                config=config,
                gt_table=gt_table,
                pred_tables=pred_tables,
                used_pred_indices=used_pred_indices,
            )

        if matched_idx is not None:
            used_pred_indices.add(matched_idx)
            matched_pred = pred_tables[matched_idx]
            logger.info("Table matched:")
            logger.info("\tGT %d: caption='%s'", gt_i + 1, gt_caption[:50])
            logger.info("\tPred %d: caption='%s'", matched_idx + 1, matched_pred.caption[:50])
            comparison = compare_tables(config=config, gt_table=gt_table, pred_table=matched_pred)

            logger.info(
                "Table %d comparison result: match_score=%s, numerical_match=%s, structure_match=%s",
                gt_i + 1,
                comparison.match_score,
                comparison.numerical_match,
                comparison.structure_match,
            )

            table_result: TableComparison = {
                "gt_caption": gt_caption,
                "gt_label": gt_label,
                "pred_caption": matched_pred.caption,
                "pred_label": matched_pred.label,
                "match_score": comparison.match_score,
                "numerical_match": comparison.numerical_match,
                "structure_match": comparison.structure_match,
                "differences": comparison.differences,
                "reasoning": comparison.reasoning,
            }

            comparison_results.append(table_result)
            total_match_score += comparison.match_score
            compared_count += 1
        else:
            logger.warning("NO MATCHING TABLE FOUND for GT table %d", gt_i + 1)
            comparison_results.append(
                {
                    "gt_caption": gt_caption,
                    "gt_label": gt_label,
                    "pred_caption": "",
                    "pred_label": None,
                    "match_score": 1,
                    "numerical_match": False,
                    "structure_match": False,
                    "differences": ["No matching table found in prediction"],
                    "reasoning": "No matching table found",
                }
            )

    table_summary: TableSummary = {
        "total_gt_tables": len(gt_tables),
        "total_pred_tables": len(pred_tables),
        "average_match_score": total_match_score / len(gt_tables) if gt_tables else 0,
        "compared_tables": compared_count,
    }
    return (comparison_results, table_summary)
