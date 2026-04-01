#!/usr/bin/env python3
from __future__ import annotations

"""Compare GT and Pred LaTeX files section-by-section, evaluating quality and hallucination."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, Field

from paper_recon.common.coding_agent import run_agent
from paper_recon.common.config import AgentConfig, LLMConfig
from paper_recon.common.llm import get_response_from_llm
from paper_recon.common.log import get_logger
from paper_recon.evaluation.evaluate_citation import (
    CitationSectionEvaluation,
    CitationSummary,
    default_citation_summary,
    evaluate_citation_f1,
)

logger = get_logger(__file__)


@dataclass
class Section:
    name: str
    content: str


SECTION_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Introduction", ["introduction"]),
    (
        "Related Work",
        [
            "related work",
            "relatedwork",
            "related works",
            "relatedworks",
            "background and related work",
            "background",
        ],
    ),
    (
        "Method",
        [
            "method",
            "methods",
            "methodology",
            "our method",
            "our approach",
            "approach",
            "proposed method",
            "proposed framework",
            "proposed attack",
            "preliminary",
            "preliminaries",
            "model",
            "models",
        ],
    ),
    (
        "Benchmark Construction",
        [
            "benchmark",
            "dataset",
            "datasets",
            "data analysis",
            "benchmark construction",
            "dataset construction",
            "benchmark tasks and experimental setup",
            "fairness benchmark settings",
        ],
    ),
    (
        "Experiment",
        [
            "experiment",
            "experiments",
            "experimental results",
            "experimental evaluation",
            "experimental setup",
            "experiments and results",
            "evaluation",
            "evaluation results",
            "evaluation setup",
            "results",
            "results and analysis",
            "analysis",
            "ablation studies",
            "ablation and analysis",
            "empirical studies and key findings",
            "empirical analysis",
        ],
    ),
    (
        "Conclusion",
        [
            "conclusion",
            "conclusions",
            "conclusion and discussion",
            "conclusion and discussions",
            "conclusion and outlook",
            "discussion",
            "discussions",
            "discussion and conclusion",
        ],
    ),
]


VALID_CATEGORIES = [
    "Introduction",
    "Related Work",
    "Method",
    "Benchmark Construction",
    "Experiment",
    "Conclusion",
]


class SectionClassificationResponse(BaseModel):
    category: str = Field(
        description="One of: Introduction, Related Work, Method, Benchmark Construction, Experiment, Conclusion, or Skip",
    )
    reasoning: str


def classify_section_name_by_rule(normalized_name: str) -> str | None:
    """Classify section name by rule-based matching. Returns None if no match."""
    for category, keywords in SECTION_CATEGORY_RULES:
        if normalized_name in keywords:
            return category
        for kw in keywords:
            if kw in normalized_name:
                return category
    return None


def classify_section_name_by_llm(
    section_name: str,
    section_content_preview: str,
    config: LLMConfig,
) -> str | None:
    """Classify section name into a category using LLM."""
    categories_str = ", ".join(VALID_CATEGORIES)
    system_prompt = (
        "You are a paper structure classifier. Given a LaTeX section name and a short preview of its content, "
        "classify it into exactly one of the following categories:\n"
        f"{categories_str}\n\n"
        "If the section does not fit any category (e.g., Acknowledgements, Ethics Statement, Limitations), "
        "return 'Skip'.\n\n"
        "Respond in JSON:\n"
        '{"category": "<category>", "reasoning": "<brief explanation>"}'
    )
    user_prompt = (
        f"Section name: {section_name}\n\n"
        f"Content preview (first 500 chars):\n{section_content_preview[:500]}"
    )

    response, _ = get_response_from_llm(
        model=config.model,
        user_message=user_prompt,
        system_message=system_prompt,
        response_format=SectionClassificationResponse,
    )

    category = response.category.strip()
    if category == "Skip" or category not in VALID_CATEGORIES:
        logger.info("LLM classified '%s' -> Skip (reasoning: %s)", section_name, response.reasoning)
        return None
    logger.info(
        "LLM classified '%s' -> '%s' (reasoning: %s)", section_name, category, response.reasoning
    )
    return category


def _merge_sections_by_category(sections: list[Section]) -> list[Section]:
    """Merge sections classified into the same category, preserving order."""
    seen_categories: dict[str, int] = {}  # category -> index in merged list
    merged: list[Section] = []

    for section in sections:
        category = section.name  # already classified
        if category in seen_categories:
            idx = seen_categories[category]
            merged[idx] = Section(
                name=category,
                content=merged[idx].content + "\n\n" + section.content,
            )
            logger.info(
                "Merged duplicate section '%s' (%d chars total)", category, len(merged[idx].content)
            )
        else:
            seen_categories[category] = len(merged)
            merged.append(Section(name=category, content=section.content))

    return merged


def extract_sections_from_text(text: str) -> list[Section]:
    r"""
    Extract all sections defined by \\section{} commands from LaTeX text.
    Section names are kept as-is (classification is done separately via classify_and_merge_sections).

    Args:
        text (str): latex content as string

    Returns:
        list[Section]: A list of Section objects with original section names.

    """
    sections: list[Section] = []

    # Extract Abstract (between \begin{abstract} and \end{abstract})
    abstract_pattern = r"\\begin\{abstract\}"
    abstract_match = re.search(abstract_pattern, text)
    if abstract_match:
        abstract_start = abstract_match.end()
        abstract_end_pattern = r"\\end\{abstract\}"
        abstract_end_match = re.search(abstract_end_pattern, text[abstract_start:])
        if abstract_end_match:
            abstract_end = abstract_start + abstract_end_match.start()
            abstract_content = text[abstract_start:abstract_end]
            sections.append(Section(name="Abstract", content=abstract_content))

    # Find all \section{...} commands
    section_pattern = r"\\section\{([^}]+)\}"
    section_matches = list(re.finditer(section_pattern, text))

    for i, match in enumerate(section_matches):
        section_name = match.group(1).strip()
        start_pos = match.end()

        if i + 1 < len(section_matches):
            end_pos = section_matches[i + 1].start()
        else:
            end_pos = len(text)

        section_content = text[start_pos:end_pos]
        sections.append(Section(name=section_name, content=section_content))

    return sections


def classify_and_merge_sections(
    sections: list[Section],
    llm_config: LLMConfig | None = None,
) -> list[Section]:
    """
    Classify sections into 7 categories and merge same-category sections.

    1. Keep Abstract as-is
    2. Try rule-based classification
    3. Fall back to LLM classification if llm_config is provided
    4. Merge sections in the same category
    """
    classified: list[Section] = []

    for section in sections:
        if section.name == "Abstract":
            classified.append(section)
            continue

        normalized = normalize_section_name(section.name)
        category = classify_section_name_by_rule(normalized)

        if category is None and llm_config is not None:
            logger.info(
                "Rule-based classification failed for '%s', falling back to LLM", section.name
            )
            category = classify_section_name_by_llm(
                section_name=section.name,
                section_content_preview=section.content,
                config=llm_config,
            )

        if category is not None:
            classified.append(Section(name=category, content=section.content))
            logger.debug(
                "Classified '%s' -> '%s' (%d chars)", section.name, category, len(section.content)
            )
        else:
            logger.debug("Skipping unclassified section: '%s'", section.name)

    # Merge sections in the same category
    classified = _merge_sections_by_category(classified)

    return classified


def clean_latex_content(content: str) -> str:
    """Strip LaTeX commands and comments, extracting plain text only."""
    lines = content.split("\n")
    cleaned_lines = []

    for _line in lines:
        line = _line
        # Skip comment lines
        if line.strip().startswith("%"):
            continue

        # Remove inline comments
        if "%" in line:
            # Exclude escaped \%
            comment_pos = line.find("%")
            if comment_pos > 0 and line[comment_pos - 1] != "\\":
                line = line[:comment_pos]

        # Remove basic LaTeX commands
        # Remove \label{...}
        line = re.sub(r"\\label\{[^}]+\}", "", line)
        # Keep \citep{...} and \cite{...} (citations are important)
        # Remove formatting commands like \emph{...}, \textbf{...}
        line = re.sub(r"\\(?:emph|textbf|textit|texttt)\{([^}]+)\}", r"\1", line)
        # Remove \begin{...} and \end{...} blocks
        line = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", "", line, flags=re.DOTALL)
        # Remove other simple commands
        line = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})*", "", line)

        # Only add non-empty lines
        if line.strip():
            cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)


from paper_recon.evaluation.evaluate_figure import (  # noqa: E402
    FigureContextComparison,
    FigureCoverage,
    FigureSummary,
    SectionFigureInfo,
    matching_and_evaluate_figures,
)
from paper_recon.evaluation.evaluate_table import (  # noqa: E402
    TableComparison,
    TableSummary,
    matching_and_evaluate_tables,
)


def extract_sections_from_latex(latex_path: Path) -> list[Section]:
    """Extract sections from a LaTeX file."""
    content = latex_path.read_text(encoding="utf-8")

    # Extract content between \begin{document} and \end{document}
    doc_start = content.find(r"\begin{document}")
    doc_end = content.find(r"\end{document}")

    if doc_start != -1 and doc_end != -1:
        content = content[doc_start:doc_end]
        logger.debug(
            r"Extracted content between \begin{document} and \end{document}: %d characters",
            len(content),
        )
    elif doc_start != -1:
        content = content[doc_start:]
        logger.debug(r"Extracted content from \begin{document} to end: %d characters", len(content))
    # If doc_start is -1, use entire content
    else:
        logger.warning(r"No \begin{document} found, using entire file content")
    return extract_sections_from_text(content)


def normalize_section_name(section_name: str) -> str:
    """Normalize section name for comparison."""
    # Convert to lowercase
    normalized = section_name.lower()
    # Remove special characters
    normalized = re.sub(r"[^\w\s]", "", normalized)
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


SECTION_CITATION_WEIGHTS = {
    "abstract": 0.05,
    "introduction": 0.20,
    "related work": 0.35,
    "method": 0.20,
    "benchmark construction": 0.10,
    "experiment": 0.20,
    "conclusion": 0.05,
}

FIGURE_TABLE_ENV_PATTERN = re.compile(
    r"\\begin\{(?:figure\*?|table\*?)\}.*?\\end\{(?:figure\*?|table\*?)\}",
    flags=re.DOTALL,
)


def match_sections(
    gt_sections: list[Section], pred_sections: list[Section]
) -> list[tuple[Section | None, Section | None]]:
    """
    Match GT and Pred sections by normalized name.

    Args:
        gt_sections (list[Section]): List of GT sections.
        pred_sections (list[Section]): List of predicted sections.

    Returns:
        list[tuple[Section, Section]]: [(gt_section, pred_section), ...]

    """

    def _merged_section_names(
        gt_sections: list[Section],
        pred_sections: list[Section],
    ) -> list[str]:
        ordered_lists = [
            [normalize_section_name(section.name) for section in gt_sections],
            [normalize_section_name(section.name) for section in pred_sections],
        ]

        first_seen: dict[str, int] = {}
        next_index = 0
        adjacency: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}

        for names in ordered_lists:
            filtered_names = [name for name in names if name]
            for name in filtered_names:
                if name not in first_seen:
                    first_seen[name] = next_index
                    next_index += 1
                adjacency.setdefault(name, set())
                indegree.setdefault(name, 0)
            for prev_name, next_name in zip(filtered_names, filtered_names[1:], strict=False):
                if next_name not in adjacency[prev_name]:
                    adjacency[prev_name].add(next_name)
                    indegree[next_name] = indegree.get(next_name, 0) + 1

        queue = sorted(
            [name for name, degree in indegree.items() if degree == 0],
            key=lambda name: first_seen[name],
        )
        ordered_names: list[str] = []
        while queue:
            current = queue.pop(0)
            ordered_names.append(current)
            for neighbor in sorted(adjacency[current], key=lambda name: first_seen[name]):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort(key=lambda name: first_seen[name])

        if len(ordered_names) < len(first_seen):
            remaining = sorted(
                [name for name in first_seen if name not in ordered_names],
                key=lambda name: first_seen[name],
            )
            ordered_names.extend(remaining)

        return ordered_names

    gt_normalized = {normalize_section_name(section.name): section for section in gt_sections}
    pred_normalized = {normalize_section_name(section.name): section for section in pred_sections}
    ordered_names = _merged_section_names(gt_sections, pred_sections)
    return [(gt_normalized.get(name), pred_normalized.get(name)) for name in ordered_names]


# ---------------------------------------------------------------------------
# Rubric-based evaluation (eval_points)
# ---------------------------------------------------------------------------


class RubricItemResult(BaseModel):
    element: str
    score: int = Field(
        ge=1, le=5, description="1=not described at all, 5=completely and accurately described"
    )
    reasoning: str


class RubricSectionResult(BaseModel):
    section_name: str
    results: list[RubricItemResult]
    average_score: float
    total_count: int


RUBRIC_EVAL_PROMPT = """\
You are an expert paper reviewer. You are given:
1. A list of key elements (rubric) that should appear in a specific section of a paper.
2. The predicted section content to evaluate.
3. (Optional) Figure/Table context showing which visual assets are present or missing.

For each element, score how well the predicted section covers it on a 1-5 scale:
5: Fully and accurately described. The element is present with correct details.
4: Mostly described. The core idea is present but some details are missing or slightly imprecise.
3: Partially described. The element is mentioned but with significant gaps or vagueness.
2: Barely mentioned. Only a superficial or indirect reference exists.
1: Not described at all. The element is completely absent from the predicted section.


Respond in JSON format. For each element, provide:
- "element": the element name (copy exactly from input)
- "score": 1-5
- "reasoning": brief explanation
"""


class RubricEvalResponse(BaseModel):
    results: list[RubricItemResult]


def evaluate_section_by_rubric(
    config: LLMConfig,
    section_name: str,
    pred_content: str,
    eval_points: list[dict],
    figure_table_context: str = "",
) -> RubricSectionResult:
    """Evaluate a Pred section by rubric using eval_points and figure/table context."""
    elements_text = "\n".join(
        f"- **{p['element']}** ({p['importance']}): {p['description']}" for p in eval_points
    )

    asset_info = ""
    if figure_table_context:
        asset_info = (
            "\n\n### Figure/Table Context for this Section\n"
            "The following analysis describes the presence or absence of visual assets "
            "in this section compared to the Ground Truth:\n"
            f"{figure_table_context}\n"
        )

    user_prompt = (
        f"**Section: {section_name}**\n\n"
        f"**Rubric (key elements to check):**\n{elements_text}\n\n"
        f"**Predicted section content:**\n{pred_content}\n"
        f"{asset_info}\n"
        "**Instructions:**\n"
        "1. Evaluate each rubric element based on the text content.\n"
        "2. CRITICAL: If a rubric element requires or refers to data/visuals, and the "
        "corresponding Figure/Table is reported as MISSING or has low match score in the "
        "context above, you MUST penalize the score for that element.\n"
        "Evaluate each element on a 1-5 scale."
    )

    response, _ = get_response_from_llm(
        model=config.model,
        user_message=user_prompt,
        system_message=RUBRIC_EVAL_PROMPT,
        temperature=config.temp,
        response_format=RubricEvalResponse,
    )

    if isinstance(response, RubricEvalResponse):
        result = response
    else:
        result = RubricEvalResponse.model_validate_json(response)

    total = len(result.results)
    avg = sum(r.score for r in result.results) / total if total > 0 else 0.0

    return RubricSectionResult(
        section_name=section_name,
        results=result.results,
        average_score=avg,
        total_count=total,
    )


# ---------------------------------------------------------------------------
# Hallucination claim analysis (count-based)
# ---------------------------------------------------------------------------


class ClaimClassification(BaseModel):
    claim: str = Field(description="Specific claim from the predicted section")
    classification: str = Field(description="supported / neutral / contradictory")
    evidence: str = Field(description="Evidence for the classification")
    severity: str | None = Field(
        default=None,
        description="Only for contradictory: major / minor",
    )


class SectionClaimAnalysis(BaseModel):
    claims: list[ClaimClassification]


class HallucinationSectionResult(BaseModel):
    section_name: str
    claims: list[dict]
    supported: int
    neutral: int
    contradictory_major: int
    contradictory_minor: int


HALLUCINATION_CLAIM_PROMPT = """\
You are an expert paper reviewer detecting factual errors in a predicted paper section.

You are given:
1. The predicted section content.
2. The ground truth (GT) full paper content for reference.

Your task is to identify all **concrete, verifiable claims** in the predicted section \
(e.g., specific numbers, method descriptions, experimental setups, results) \
and classify each into one of three categories:

- **supported**: The claim is directly stated in or logically derivable from the GT paper.
- **neutral**: The claim is NOT in the GT, but is a reasonable general statement, \
common knowledge, or supplementary detail that does not contradict the GT. \
This is NOT an error.
- **contradictory**: The claim **directly contradicts** specific information in the GT paper. \
This is a factual error / hallucination.

For contradictory claims, also assign severity:
- **major**: Incorrect numbers, fabricated results, wrong method descriptions, \
misattributed findings — errors that would mislead a reader.
- **minor**: Overly strong generalizations, imprecise wording that slightly distorts meaning, \
minor numerical rounding issues.

IMPORTANT:
- Do NOT classify claims as contradictory simply because they are absent from the GT. \
Absence ≠ contradiction.
- Focus on claims that can be verified against the GT. Skip purely stylistic or structural observations.
- Be thorough: extract ALL verifiable claims, not just a few.

Respond in JSON with a list of claims. Each claim has:
- "claim": the specific statement from Pred
- "classification": "supported" | "neutral" | "contradictory"
- "evidence": brief explanation
- "severity": "major" | "minor" (only for contradictory, null otherwise)
"""


HALLUCINATION_VERIFY_PROMPT = """\
You are a rigorous fact-checker performing a second-pass verification.

A previous reviewer flagged the following claim as **contradictory** (factual error / hallucination) \
in a predicted paper section. Your job is to carefully re-examine whether this is truly a contradiction \
with the Ground Truth paper, or a false positive.

Classify the claim into one of:
- **contradictory**: Confirmed. The claim genuinely contradicts specific information in the GT paper.
- **neutral**: False positive. The claim is absent from the GT but does NOT contradict it. \
Absence is not contradiction.
- **supported**: False positive. The claim is actually supported by the GT paper.

For confirmed contradictory claims, assign severity:
- **major**: Incorrect numbers, fabricated results, wrong method descriptions.
- **minor**: Overly strong generalizations, imprecise wording.

Respond in JSON:
- "classification": "supported" | "neutral" | "contradictory"
- "severity": "major" | "minor" for contradictory, "none" for supported/neutral
- "evidence": brief explanation of why you changed or kept the classification
"""


class VerifyClaimResponse(BaseModel):
    classification: str
    severity: str = Field(description="'major' or 'minor' for contradictory, 'none' otherwise")
    evidence: str


class BatchVerifyItem(BaseModel):
    claim: str
    classification: str
    severity: str = Field(description="'major' or 'minor' for contradictory, 'none' otherwise")
    evidence: str


class BatchVerifyResponse(BaseModel):
    results: list[BatchVerifyItem]


def _verify_contradictory_claims_batch(
    agent_config: AgentConfig,
    claims: list[ClaimClassification],
    gt_resources_dir: Path,
) -> list[ClaimClassification]:
    """Re-verify multiple contradictory claims in batch using coding agent."""
    claims_text = "\n".join(
        f"### Claim {i + 1}\n"
        f"- Claim: {c.claim}\n"
        f"- Original evidence: {c.evidence}\n"
        f"- Original severity: {c.severity}\n"
        for i, c in enumerate(claims)
    )

    prompt = f"""{HALLUCINATION_VERIFY_PROMPT}

The following {len(claims)} claims were flagged as **contradictory** by a previous reviewer.
Re-examine EACH claim and classify it independently.

{claims_text}

Your current working directory contains the Ground Truth paper resources:
- The GT paper's LaTeX source (main.tex, gt_main.tex, or similar .tex files)
- code/ directory with the original codebase (if exists)
- figures/ directory with figure images
- tables/ directory with table data (.tex files)

Please read the relevant GT files to verify each claim. For claims about the implementation details or methods, check the code/ directory. For other claims, you can check main.tex/gt_main.tex and tables/.
Return a JSON with a "results" array containing one entry per claim, in the same order.
"""

    response = run_agent(
        agent=agent_config.agent,
        user_prompt=prompt,
        working_dir=gt_resources_dir,
        model=agent_config.model,
        max_turns=agent_config.max_turns,
        mode="READONLY",
        response_format=BatchVerifyResponse,
    )

    try:
        batch_result = BatchVerifyResponse.model_validate_json(response)
    except Exception:
        logger.warning(
            "Failed to parse batch verification response, keeping original classifications"
        )
        return claims

    verified_claims: list[ClaimClassification] = []
    for original, verified in zip(claims, batch_result.results):
        verified_claims.append(
            ClaimClassification(
                claim=original.claim,
                classification=verified.classification,
                severity=verified.severity if verified.severity != "none" else None,
                evidence=f"[Agent Verified] {verified.evidence}",
            )
        )

    return verified_claims


def _extract_claims_for_section(
    config: LLMConfig,
    section_name: str,
    pred_content: str,
    gt_full_content: str,
) -> SectionClaimAnalysis:
    """Pass 1: Extract and classify claims for one section using LLM."""
    user_prompt = (
        f"**Section being evaluated: {section_name}**\n\n"
        f"**Predicted section content:**\n{pred_content}\n\n"
        f"**Ground Truth full paper (for reference):**\n{gt_full_content}\n\n"
        "Extract and classify all verifiable claims from the predicted section."
    )

    response, _ = get_response_from_llm(
        model=config.model,
        user_message=user_prompt,
        system_message=HALLUCINATION_CLAIM_PROMPT,
        temperature=config.temp,
        response_format=SectionClaimAnalysis,
    )

    if isinstance(response, SectionClaimAnalysis):
        return response
    return SectionClaimAnalysis.model_validate_json(response)


def _build_hallucination_result(
    section_name: str,
    claims: list[ClaimClassification],
    verified_map: dict[int, ClaimClassification],
) -> HallucinationSectionResult:
    """Merge Pass 1 claims with Pass 2 verification results into final output."""
    final_claims: list[ClaimClassification] = []
    num_verified = 0
    num_overturned = 0

    for i, c in enumerate(claims):
        if i in verified_map:
            verified = verified_map[i]
            num_verified += 1
            if verified.classification != "contradictory":
                num_overturned += 1
                logger.info(
                    "  Overturned: '%s' -> %s",
                    c.claim[:60],
                    verified.classification,
                )
            final_claims.append(verified)
        else:
            final_claims.append(c)

    if num_verified > 0:
        logger.info(
            "  Verification [%s]: %d contradictory claims checked, %d overturned",
            section_name,
            num_verified,
            num_overturned,
        )

    supported = 0
    neutral = 0
    contradictory_major = 0
    contradictory_minor = 0
    for c in final_claims:
        if c.classification == "supported":
            supported += 1
        elif c.classification == "neutral":
            neutral += 1
        elif c.classification == "contradictory":
            if c.severity == "major":
                contradictory_major += 1
            else:
                contradictory_minor += 1

    return HallucinationSectionResult(
        section_name=section_name,
        claims=[c.model_dump() for c in final_claims],
        supported=supported,
        neutral=neutral,
        contradictory_major=contradictory_major,
        contradictory_minor=contradictory_minor,
    )


def _format_table_comparisons_for_prompt(table_comparisons: list[TableComparison]) -> str:
    """Format table comparison results for LLM prompt."""
    if not table_comparisons:
        return "(no table comparison results available for this section)"
    lines: list[str] = []
    for i, tc in enumerate(table_comparisons):
        lines.append(
            f"Table {i + 1}:\n"
            f"  GT caption: {tc['gt_caption']}\n"
            f"  GT label: {tc['gt_label']}\n"
            f"  Pred caption: {tc['pred_caption']}\n"
            f"  Pred label: {tc['pred_label']}\n"
            f"  Match score: {tc['match_score']}/100\n"
            f"  Numerical match: {tc['numerical_match']}\n"
            f"  Structure match: {tc['structure_match']}\n"
            f"  Differences: {tc['differences']}\n"
            f"  Reasoning: {tc['reasoning']}"
        )
    return "\n\n".join(lines)


def _find_tables_in_section(
    section_content: str,
    table_comparisons: list[TableComparison],
    gt_section_content: str = "",
) -> list[TableComparison]:
    """
    Extract table comparison results relevant to a section.

    For matched tables, search pred section content by pred label/caption.
    For missing tables, search gt section content by gt label.
    """
    result: list[TableComparison] = []
    for tc in table_comparisons:
        pred_label = tc.get("pred_label")
        pred_caption = tc.get("pred_caption", "")
        gt_label = tc.get("gt_label")
        is_missing = not pred_caption and not pred_label
        if is_missing:
            # Missing table: search GT section content by GT label
            if gt_section_content and gt_label and gt_label in gt_section_content:
                result.append(tc)
        else:
            # Matched table: search Pred section content by Pred label/caption
            if (pred_label and pred_label in section_content) or (pred_caption and pred_caption[:40] in section_content):
                result.append(tc)
    return result


def classify_sections_for_paper(
    gt_latex_path: Path,
    pred_latex_path: Path,
    llm_config: LLMConfig,
) -> tuple[list[Section], list[Section], list[tuple[Section | None, Section | None]]]:
    """
    Extract, classify, and match sections. Called at the start of evaluate_paper.

    Returns:
        (gt_sections, pred_sections, matched_sections)

    """
    gt_sections_raw = extract_sections_from_latex(gt_latex_path)
    logger.info("Extracted %d raw sections from GT", len(gt_sections_raw))
    for s in gt_sections_raw:
        logger.info("\t'%s':\t%d characters", s.name, len(s.content))

    pred_sections_raw = extract_sections_from_latex(pred_latex_path)
    logger.info("Extracted %d raw sections from Pred", len(pred_sections_raw))
    for s in pred_sections_raw:
        logger.info("\t'%s':\t%d characters", s.name, len(s.content))

    gt_sections = classify_and_merge_sections(gt_sections_raw, llm_config=llm_config)
    logger.info(
        "Classified GT into %d sections: %s", len(gt_sections), [s.name for s in gt_sections]
    )

    pred_sections = classify_and_merge_sections(pred_sections_raw, llm_config=llm_config)
    logger.info(
        "Classified Pred into %d sections: %s", len(pred_sections), [s.name for s in pred_sections]
    )

    matched_sections = match_sections(gt_sections, pred_sections)
    logger.info("Matched %d section pairs:", len(matched_sections))
    for gt_section, pred_section in matched_sections:
        logger.debug(
            "\t GT '%s'\t<-> Pred '%s'",
            gt_section.name if gt_section else "None",
            pred_section.name if pred_section else "None",
        )

    return gt_sections, pred_sections, matched_sections


class SectionSummary(TypedDict, total=True):
    total_sections: int
    average_quality: float
    average_hallucination: float


class SectionEvaluation(TypedDict, total=True):
    gt_section_name: str
    pred_section_name: str
    quality: float
    hallucination: float
    quality_reasoning: str
    hallucination_reasoning: str
    key_differences: list[str]
    key_differences_unknown: list[str]
    citation: CitationSectionEvaluation | None
    figures: list[SectionFigureInfo]


class EvalResult(TypedDict, total=False):
    gt_latex: str
    pred_latex: str
    sections: list[SectionEvaluation]
    overall: dict
    section_summary: SectionSummary
    citation_summary: CitationSummary
    tables: list[TableComparison]
    table_summary: TableSummary
    figure_coverage: list[FigureCoverage]
    figure_context: list[FigureContextComparison]
    figure_summary: FigureSummary
    rubric: list[dict]
    rubric_summary: dict
    hallucination_analysis: list[dict]
    hallucination_claim_summary: dict


def evaluate_paper(
    gt_latex_path: Path,
    pred_latex_path: Path,
    gt_codebase_dir: Path,
    llm_config: LLMConfig,
    agent_config: AgentConfig,
    output_path: Path | None = None,
    eval_points_path: Path | None = None,
    figure_summary_path: Path | None = None,
    eval_mode: str = "rubric",
    hal_verification_dir: Path | None = None,
) -> EvalResult:
    """
    Compare GT and Pred LaTeX files and run evaluation.

    Args:
        gt_latex_path: Path to the GT LaTeX file.
        pred_latex_path: Path to the predicted LaTeX file.
        gt_codebase_dir: Path to the GT codebase directory.
        llm_config: LLM configuration.
        agent_config: Agent configuration.
        output_path: Optional path to save results.

    Returns:
        EvalResult dict.

    """
    logger.info("=" * 80)
    logger.info("Starting paper evaluation")
    logger.info("GT LaTeX: %s", gt_latex_path)
    logger.info("Pred LaTeX: %s", pred_latex_path)
    logger.info("LLM: %s", llm_config)
    logger.info("=" * 80)

    # Step 1: Section classification (shared across all eval modes)
    gt_sections, pred_sections, matched_sections = classify_sections_for_paper(
        gt_latex_path, pred_latex_path, llm_config
    )

    # Citation mode: compute citation F1 only and return early
    if eval_mode == "citation":
        citation_f1 = evaluate_citation_f1(gt_latex_path, pred_latex_path)
        results: EvalResult = {
            "gt_latex": str(gt_latex_path),
            "pred_latex": str(pred_latex_path),
            "sections": [],
            "overall": {},
            "section_summary": {"average_quality": 0.0, "average_hallucination": 0.0},
            "citation_summary": {
                "citation_f1": citation_f1,
            },
            "table_summary": {},
            "tables": [],
            "figure_coverage": [],
            "figure_context": [],
            "figure_summary": {},
            "rubric": [],
            "rubric_summary": {},
            "hallucination_analysis": [],
            "hallucination_claim_summary": {},
        }
        if output_path:
            output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Citation results saved to: %s", output_path)
        return results

    # Hallucination mode: run hallucination analysis only and return early
    if eval_mode == "hallucination":
        logger.info("Running hallucination-only analysis...")
        gt_full_content = gt_latex_path.read_text(encoding="utf-8")
        hal_sections: list[tuple[str, str]] = []
        for gt_section, pred_section in matched_sections:
            sec_name = (
                gt_section.name if gt_section else (pred_section.name if pred_section else "N/A")
            )
            if sec_name == "Conclusion":
                continue
            pred_content = pred_section.content if pred_section else ""
            if not pred_content or len(pred_content.strip()) < 50:
                continue
            hal_sections.append((sec_name, pred_content))

        # Pass 1: claim extraction (sequential)
        pass1_results: dict[str, SectionClaimAnalysis] = {}
        for sec_name, pred_content in hal_sections:
            try:
                claim_result = _extract_claims_for_section(
                    config=llm_config,
                    section_name=sec_name,
                    pred_content=pred_content,
                    gt_full_content=gt_full_content,
                )
                pass1_results[sec_name] = claim_result
            except Exception:
                logger.exception("Hallucination Pass 1 failed for section '%s'", sec_name)

        # Pass 2: Agent verification (same as normal flow)
        all_contradictory: list[tuple[str, int, ClaimClassification]] = []
        for sec_name, claim_result in pass1_results.items():
            for i, claim in enumerate(claim_result.claims):
                if claim.classification == "contradictory":
                    all_contradictory.append((sec_name, i, claim))

        all_verified: dict[tuple[str, int], ClaimClassification] = {}
        if all_contradictory and agent_config:
            try:
                all_claims_for_verify: list[ClaimClassification] = []
                all_sections_for_verify: list[str] = []
                for sec_name, _, c in all_contradictory:
                    all_claims_for_verify.append(c)
                    all_sections_for_verify.append(sec_name)
                for claim, sec_name in zip(all_claims_for_verify, all_sections_for_verify):
                    claim.claim = f"[Section: {sec_name}] {claim.claim}"
                verified_list = _verify_contradictory_claims_batch(
                    agent_config,
                    all_claims_for_verify,
                    hal_verification_dir or gt_codebase_dir.parent,
                )
                for claim, sec_name in zip(all_claims_for_verify, all_sections_for_verify):
                    prefix = f"[Section: {sec_name}] "
                    claim.claim = claim.claim.removeprefix(prefix)
                for (sec_name, idx, _), verified in zip(all_contradictory, verified_list):
                    all_verified[(sec_name, idx)] = verified
            except Exception:
                logger.exception("Hallucination Pass 2 (Agent verification) failed")

        hallucination_results: list[dict] = []
        total_supported = total_neutral = total_contradictory_major = total_contradictory_minor = 0
        for sec_name, claim_result in pass1_results.items():
            sec_verified = {idx: v for (sn, idx), v in all_verified.items() if sn == sec_name}
            hal_result = _build_hallucination_result(sec_name, claim_result.claims, sec_verified)
            hallucination_results.append({"section_name": sec_name, **hal_result.model_dump()})
            total_supported += hal_result.supported
            total_neutral += hal_result.neutral
            total_contradictory_major += hal_result.contradictory_major
            total_contradictory_minor += hal_result.contradictory_minor

        hallucination_summary = {
            "total_supported": total_supported,
            "total_neutral": total_neutral,
            "total_contradictory_major": total_contradictory_major,
            "total_contradictory_minor": total_contradictory_minor,
            "total_claims": total_supported
            + total_neutral
            + total_contradictory_major
            + total_contradictory_minor,
        }
        results: EvalResult = {
            "gt_latex": str(gt_latex_path),
            "pred_latex": str(pred_latex_path),
            "sections": [],
            "overall": {},
            "section_summary": {"average_quality": 0.0, "average_hallucination": 0.0},
            "citation_summary": {},
            "table_summary": {},
            "tables": [],
            "figure_coverage": [],
            "figure_context": [],
            "figure_summary": {},
            "rubric": [],
            "rubric_summary": {},
            "hallucination_analysis": hallucination_results,
            "hallucination_claim_summary": hallucination_summary,
        }
        if output_path:
            output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Hallucination results saved to: %s", output_path)
        return results

    # Step 2: Table evaluation
    table_comparisons, table_summary = matching_and_evaluate_tables(
        gt_latex_path, pred_latex_path, llm_config
    )

    # Step 3: Figure evaluation
    figure_coverage: list[FigureCoverage] = []
    figure_context: list[FigureContextComparison] = []
    figure_summary: FigureSummary = {
        "total_provided_figures": 0,
        "total_included_figures": 0,
        "total_referenced_figures": 0,
        "coverage_score": 0.0,
        "reference_score": 0.0,
        "average_context_score": 0.0,
        "total_matched_figures": 0,
    }
    if figure_summary_path:
        figure_coverage, figure_context, figure_summary = matching_and_evaluate_figures(
            gt_latex_path,
            pred_latex_path,
            figure_summary_path,
            llm_config.model,
            llm_config.temp,
            gt_classified_sections=gt_sections,
            pred_classified_sections=pred_sections,
        )

    # Step 4: Citation F1 evaluation (all mode only)
    citation_summary: CitationSummary = default_citation_summary(
        total_sections=len(matched_sections)
    )
    if eval_mode == "all":
        citation_f1 = evaluate_citation_f1(gt_latex_path, pred_latex_path)
        citation_summary["citation_f1"] = citation_f1

    # Step 5: Section evaluation (section-wise)
    section_evals: list[SectionEvaluation] = []
    section_summary: SectionSummary = {"average_quality": 0.0, "average_hallucination": 0.0}

    # Step 6: Rubric evaluation (if eval_points.json exists)
    rubric_results: list[dict] = []
    rubric_summary: dict = {}
    if eval_mode in ("rubric", "all") and eval_points_path and eval_points_path.exists():
        logger.info("Running rubric evaluation with %s", eval_points_path)
        eval_points_data = json.loads(eval_points_path.read_text(encoding="utf-8"))
        pred_section_map = {s.name: s.content for s in pred_sections}
        gt_section_map = {s.name: s.content for s in gt_sections}

        def _rubric_one_section(section_data: dict) -> tuple[dict, float, int]:
            """Evaluate one section by rubric. Returns (result_dict, score_sum, point_count)."""
            sec_name = section_data["section_name"]
            points = section_data["eval_points"]
            pred_content = pred_section_map.get(sec_name, "")

            if not pred_content:
                logger.warning(
                    "Rubric: Pred has no '%s' section, marking all %d points as score=1",
                    sec_name,
                    len(points),
                )
                res = RubricSectionResult(
                    section_name=sec_name,
                    results=[
                        RubricItemResult(
                            element=p["element"],
                            score=1,
                            reasoning="This section does not exist in Pred",
                        )
                        for p in points
                    ],
                    average_score=1.0,
                    total_count=len(points),
                )
                return res.model_dump(), float(len(points)), len(points)

            # Build figure/table context
            ft_context_lines: list[str] = []
            if figure_context and sec_name:
                missing_figs = []
                present_figs = []
                for fc in figure_context:
                    gt_refs = fc.get("gt_reference_sections", [])
                    pred_refs = fc.get("pred_reference_sections", [])
                    fig_name = fc.get("filename", "unknown")
                    if sec_name in gt_refs and sec_name not in pred_refs:
                        missing_figs.append(fig_name)
                    elif sec_name in gt_refs and sec_name in pred_refs:
                        score = fc.get("context_score", "N/A")
                        score_desc = {
                            5: "perfectly appropriate usage matching GT",
                            4: "mostly appropriate, minor differences",
                            3: "partially relevant usage",
                            2: "mostly inappropriate usage",
                            1: "completely inappropriate or irrelevant usage",
                        }.get(score, "unknown")
                        present_figs.append(f"{fig_name} (context_score={score}: {score_desc})")
                if missing_figs:
                    ft_context_lines.append(
                        f"[MISSING Figures] GT references these figures in this section, but Pred does not: {', '.join(missing_figs)}"
                    )
                if present_figs:
                    ft_context_lines.append(f"[Present Figures] {', '.join(present_figs)}")

            gt_content = gt_section_map.get(sec_name, "")

            if table_comparisons and sec_name:
                section_tables = _find_tables_in_section(
                    pred_content, table_comparisons, gt_section_content=gt_content
                )
                if section_tables:
                    ft_context_lines.append(
                        "[Table comparison results (GT vs Pred)]\n"
                        + _format_table_comparisons_for_prompt(section_tables)
                    )

            ft_context = "\n".join(ft_context_lines)

            result = evaluate_section_by_rubric(
                config=llm_config,
                section_name=sec_name,
                pred_content=pred_content,
                eval_points=points,
                figure_table_context=ft_context,
            )

            # Add figure context_score as rubric items
            # Score figures referenced by GT in this section
            # Use context_score even if Pred references figure in a different section (no penalty for section mismatch)
            if figure_context and sec_name:
                for fc in figure_context:
                    gt_refs = fc.get("gt_reference_sections", [])
                    fig_name = fc.get("filename", "unknown")
                    if sec_name not in gt_refs:
                        continue
                    pred_refs = fc.get("pred_reference_sections", [])
                    if pred_refs:
                        # Referenced somewhere in Pred -> use context_score
                        mapped_score = fc.get("context_score", 1)
                        reasoning = fc.get("reasoning", "")
                    else:
                        # Not referenced anywhere in Pred
                        mapped_score = 1
                        reasoning = (
                            f"GT references {fig_name} but Pred does not reference it anywhere"
                        )
                    result.results.append(
                        RubricItemResult(
                            element=f"Figure: {fig_name}", score=mapped_score, reasoning=reasoning
                        )
                    )
                    result.total_count += 1

            # Add table match_score as rubric items (including missing tables)
            if table_comparisons and sec_name:
                section_tables = _find_tables_in_section(
                    pred_content, table_comparisons, gt_section_content=gt_content
                )
                for tc in section_tables:
                    gt_caption = tc.get("gt_caption", "unknown")[:50]
                    is_missing = not tc.get("pred_caption") and not tc.get("pred_label")
                    if is_missing:
                        result.results.append(
                            RubricItemResult(
                                element=f"Table (missing): {gt_caption}",
                                score=1,
                                reasoning="GT table exists but no matching Pred table found",
                            )
                        )
                    else:
                        result.results.append(
                            RubricItemResult(
                                element=f"Table: {gt_caption}",
                                score=int(tc.get("match_score", 1)),
                                reasoning=f"match_score={tc.get('match_score', 1)}/5, numerical={tc.get('numerical_match')}, structure={tc.get('structure_match')}",
                            )
                        )
                    result.total_count += 1

            result.average_score = (
                sum(r.score for r in result.results) / result.total_count
                if result.total_count > 0
                else 0.0
            )

            logger.info(
                "Rubric [%s]: avg=%.2f/5 (%d points, incl. %d fig/table items)",
                sec_name,
                result.average_score,
                result.total_count,
                result.total_count - len(points),
            )
            return result.model_dump(), sum(r.score for r in result.results), result.total_count

        # Run sections sequentially
        sections_data = eval_points_data.get("sections", [])
        total_score_sum = 0.0
        total_points = 0
        for sd in sections_data:
            sec_name = sd["section_name"]
            try:
                result_dict, score_sum, point_count = _rubric_one_section(sd)
                rubric_results.append(result_dict)
                total_score_sum += score_sum
                total_points += point_count
            except Exception:
                logger.exception("Rubric evaluation failed for section '%s'", sec_name)

        # Sort by section order (preserve eval_points.json ordering)
        section_order = {sd["section_name"]: i for i, sd in enumerate(sections_data)}
        rubric_results.sort(key=lambda r: section_order.get(r.get("section_name", ""), 999))

        rubric_summary = {
            "total_points": total_points,
            "average_score": total_score_sum / total_points if total_points > 0 else 0.0,
        }
        logger.info(
            "Rubric overall: avg=%.2f/5 (%d points)", rubric_summary["average_score"], total_points
        )

    # Step 7: Hallucination analysis (count-based)
    # Pass 1 (LLM claim extraction) per section, Pass 2 (Agent verification) in one batch
    hallucination_results: list[dict] = []
    hallucination_summary: dict = {}
    if eval_mode == "all":
        logger.info("Running hallucination claim analysis...")
        gt_full_content = gt_latex_path.read_text(encoding="utf-8")

        # Collect sections to evaluate
        gt_section_map = {s.name: s.content for s in gt_sections}
        hal_sections: list[tuple[str, str]] = []
        for gt_section, pred_section in matched_sections:
            sec_name = (
                gt_section.name if gt_section else (pred_section.name if pred_section else "N/A")
            )
            if sec_name == "Conclusion":
                continue
            pred_content = pred_section.content if pred_section else ""
            if not pred_content or len(pred_content.strip()) < 50:
                continue
            hal_sections.append((sec_name, pred_content))

        # --- Pass 1: Extract claims via LLM (sequential) ---
        pass1_results: dict[str, SectionClaimAnalysis] = {}
        pass1_pred_contents: dict[str, str] = {}

        for sec_name, pred_content in hal_sections:
            try:
                logger.info("Hallucination Pass 1 (claim extraction) for section: %s", sec_name)
                claim_result = _extract_claims_for_section(
                    config=llm_config,
                    section_name=sec_name,
                    pred_content=pred_content,
                    gt_full_content=gt_full_content,
                )
                pass1_results[sec_name] = claim_result
                pass1_pred_contents[sec_name] = pred_content
            except Exception:
                logger.exception("Hallucination Pass 1 failed for section '%s'", sec_name)

        # --- Pass 2: Aggregate contradictory claims across sections and verify in one agent call ---
        all_contradictory: list[tuple[str, int, ClaimClassification]] = []
        for sec_name, claim_result in pass1_results.items():
            for i, c in enumerate(claim_result.claims):
                if c.classification == "contradictory":
                    all_contradictory.append((sec_name, i, c))

        # Store verification results: {(sec_name, claim_index): ClaimClassification}
        all_verified: dict[tuple[str, int], ClaimClassification] = {}
        if all_contradictory:
            logger.info(
                "Hallucination Pass 2: verifying %d contradictory claims across %d sections in 1 agent call...",
                len(all_contradictory),
                len({sec for sec, _, _ in all_contradictory}),
            )
            # Prepend section name to claims (only claim + evidence is passed, not full pred_content)
            all_claims: list[ClaimClassification] = []
            all_sections: list[str] = []
            for sec_name, _, c in all_contradictory:
                all_claims.append(c)
                all_sections.append(sec_name)
            # Temporarily prepend section name to claim text for context
            for claim, sec_name in zip(all_claims, all_sections):
                claim.claim = f"[Section: {sec_name}] {claim.claim}"
            verified_list = _verify_contradictory_claims_batch(
                agent_config,
                all_claims,
                hal_verification_dir or gt_codebase_dir.parent,
            )
            # Remove section name prefix from claim text
            for claim, sec_name in zip(all_claims, all_sections):
                prefix = f"[Section: {sec_name}] "
                claim.claim = claim.claim.removeprefix(prefix)
            for (sec_name, idx, _), verified in zip(all_contradictory, verified_list):
                all_verified[(sec_name, idx)] = verified
        else:
            logger.info(
                "Hallucination Pass 2: no contradictory claims to verify, skipping agent call"
            )

        # --- Build results per section ---
        total_supported = 0
        total_neutral = 0
        total_contradictory_major = 0
        total_contradictory_minor = 0

        for sec_name, claim_result in pass1_results.items():
            verified_map: dict[int, ClaimClassification] = {
                idx: all_verified[(sec_name, idx)]
                for idx in range(len(claim_result.claims))
                if (sec_name, idx) in all_verified
            }
            hal_result = _build_hallucination_result(
                sec_name,
                list(claim_result.claims),
                verified_map,
            )
            hallucination_results.append(hal_result.model_dump())
            total_supported += hal_result.supported
            total_neutral += hal_result.neutral
            total_contradictory_major += hal_result.contradictory_major
            total_contradictory_minor += hal_result.contradictory_minor
            logger.info(
                "Hallucination [%s]: supported=%d, neutral=%d, contradictory(major=%d, minor=%d)",
                sec_name,
                hal_result.supported,
                hal_result.neutral,
                hal_result.contradictory_major,
                hal_result.contradictory_minor,
            )

        # Preserve matched_sections ordering
        section_order = {name: i for i, (name, _) in enumerate(hal_sections)}
        hallucination_results.sort(key=lambda r: section_order.get(r.get("section_name", ""), 999))

        hallucination_summary = {
            "total_supported": total_supported,
            "total_neutral": total_neutral,
            "total_contradictory_major": total_contradictory_major,
            "total_contradictory_minor": total_contradictory_minor,
            "total_claims": total_supported
            + total_neutral
            + total_contradictory_major
            + total_contradictory_minor,
        }
        logger.info(
            "Hallucination overall: supported=%d, neutral=%d, contradictory(major=%d, minor=%d)",
            total_supported,
            total_neutral,
            total_contradictory_major,
            total_contradictory_minor,
        )
    else:
        gt_full_content = gt_latex_path.read_text(encoding="utf-8")

    overall_result = {}

    results: EvalResult = {
        "gt_latex": str(gt_latex_path),
        "pred_latex": str(pred_latex_path),
        "sections": section_evals,
        "overall": overall_result,
        "section_summary": section_summary,
        "citation_summary": citation_summary,
        "table_summary": table_summary,
        "tables": table_comparisons,
        "figure_coverage": figure_coverage,
        "figure_context": figure_context,
        "figure_summary": figure_summary,
        "rubric": rubric_results,
        "rubric_summary": rubric_summary,
        "hallucination_analysis": hallucination_results,
        "hallucination_claim_summary": hallucination_summary,
    }

    # Save results
    logger.info("=" * 80)
    logger.info("Evaluation completed")
    logger.info("Average quality: %.2f", results["section_summary"]["average_quality"])
    logger.info("Average hallucination: %.2f", results["section_summary"]["average_hallucination"])
    logger.info("Average table match score: %.2f", results["table_summary"]["average_match_score"])
    if citation_summary.get("citation_f1"):
        cf1 = citation_summary["citation_f1"]
        logger.info(
            "Citation F1: precision=%.2f recall=%.2f f1=%.2f (hallucinated=%d)",
            cf1["precision"],
            cf1["recall"],
            cf1["f1"],
            len(cf1["hallucinated_keys"]),
        )
    logger.info("=" * 80)

    if output_path:
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Results saved to: %s", output_path)
    else:
        logger.info("EVALUATION RESULTS")
        logger.info(json.dumps(results, indent=2, ensure_ascii=False))

    return results
