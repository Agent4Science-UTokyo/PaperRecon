"""Figure extraction, matching, and evaluation between GT and Pred LaTeX files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ValidationError

from paper_recon.common.llm import get_response_from_llm
from paper_recon.common.log import get_logger

logger = get_logger(__file__)


def _normalize_section_name(section_name: str) -> str:
    """Normalize a section name for easier comparison."""
    normalized = section_name.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass
class Figure:
    """Figure information extracted from a LaTeX document."""

    filename: str  # basename (e.g., "method_overview.pdf")
    caption: str
    label: str | None  # e.g., "fig:method"
    content: str  # entire \begin{figure}...\end{figure} block


@dataclass
class FigureReference:
    r"""Occurrence of \ref{fig:...} in the document."""

    label: str  # e.g., "fig:method"
    section: str  # section name where the reference occurs
    context_snippet: str  # surrounding ~200 characters of context


class FigureCoverage(TypedDict, total=True):
    filename: str
    included_in_paper: bool
    referenced_in_text: bool
    pred_label: str | None
    pred_caption: str | None


class FigureContextComparison(TypedDict, total=True):
    filename: str
    gt_label: str | None
    pred_label: str | None
    gt_reference_sections: list[str]
    pred_reference_sections: list[str]
    sections_match: bool
    context_score: int  # 1-5
    reasoning: str
    context_appropriate: bool


class FigureSummary(TypedDict, total=True):
    total_provided_figures: int
    total_included_figures: int
    total_referenced_figures: int
    coverage_score: float  # included/provided (0-1)
    reference_score: float  # referenced/provided (0-1)
    average_context_score: float
    total_matched_figures: int


class FigureContextEvaluationResponse(BaseModel):
    """Response from LLM-based figure context evaluation."""

    context_score: int
    reasoning: str
    gt_context_summary: str
    pred_context_summary: str
    context_appropriate: bool


def parse_figure_summary(figure_summary_path: Path) -> list[str]:
    """Parse figure_summary.txt and return a list of filename basenames."""
    if not figure_summary_path.exists():
        logger.warning("figure_summary.txt not found: %s", figure_summary_path)
        return []

    filenames: list[str] = []
    content = figure_summary_path.read_text(encoding="utf-8")
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        filepath = line.split(":", 1)[0].strip()
        basename = Path(filepath).name
        filenames.append(basename)

    logger.info("Parsed %d figures from %s", len(filenames), figure_summary_path)
    return filenames


def normalize_figure_filename(filename: str) -> str:
    """Normalize a figure filename (strip directory, strip extension, lowercase)."""
    return Path(filename).stem.lower()


def extract_figures_from_latex(latex_path: Path) -> list[Figure]:
    r"""Extract all \begin{figure} / \begin{figure*} environments from a LaTeX file."""
    content = latex_path.read_text(encoding="utf-8")

    doc_start = content.find(r"\begin{document}")
    doc_end = content.find(r"\end{document}")
    if doc_start != -1 and doc_end != -1:
        content = content[doc_start:doc_end]
    elif doc_start != -1:
        content = content[doc_start:]

    figures: list[Figure] = []
    figure_pattern = r"\\begin\{(figure\*?|wrapfigure|teaserfigure)\}"
    figure_matches = list(re.finditer(figure_pattern, content))

    for match in figure_matches:
        start_pos = match.start()
        env_name = match.group(1)
        end_pattern = rf"\\end\{{{re.escape(env_name)}\}}"
        end_match = re.search(end_pattern, content[start_pos:])
        if not end_match:
            continue

        end_pos = start_pos + end_match.end()
        figure_content = content[start_pos:end_pos]

        incl_match = re.search(
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}",
            figure_content,
            re.DOTALL,
        )
        if incl_match:
            raw_filename = incl_match.group(1).strip()
            basename = Path(raw_filename).name
            if not Path(basename).suffix:
                basename += ".jpg"
        else:
            continue

        caption_match = re.search(r"\\caption\{(.+?)\}", figure_content, re.DOTALL)
        caption = caption_match.group(1).strip() if caption_match else ""

        label_match = re.search(r"\\label\{([^}]+)\}", figure_content)
        label = label_match.group(1).strip() if label_match else None

        figures.append(
            Figure(filename=basename, caption=caption, label=label, content=figure_content)
        )

    # Also find \includegraphics with \captionof{figure} or \captionsetup{type=figure}+\caption
    # outside figure environments. Limit search range to avoid matching distant icons.
    text_outside_figures = re.sub(
        r"\\begin\{(figure\*?|wrapfigure|teaserfigure)\}.*?\\end\{\1\}",
        "",
        content,
        flags=re.DOTALL,
    )
    seen_paths: set[str] = {f.filename.lower() for f in figures}

    # Pattern: \includegraphics{...} followed by \captionof{figure}{...} within ~500 chars
    captionof_patterns = [
        # includegraphics then captionof
        re.compile(
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{(?P<path>[^}]+)\}"
            r"(?P<between>.{0,500}?)"
            r"\\captionof\{figure\}\{(?P<caption>[^}]+)\}",
            re.DOTALL,
        ),
        # captionof then includegraphics
        re.compile(
            r"\\captionof\{figure\}\{(?P<caption>[^}]+)\}"
            r"(?P<between>.{0,500}?)"
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{(?P<path>[^}]+)\}",
            re.DOTALL,
        ),
        # includegraphics then \captionsetup{type=figure} + \caption{...}
        re.compile(
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{(?P<path>[^}]+)\}"
            r"(?P<between>.{0,500}?)"
            r"\\captionsetup\{type=figure[^}]*\}\s*\\caption\{(?P<caption>[^}]+)\}",
            re.DOTALL,
        ),
        # \captionsetup{type=figure} + \includegraphics then \caption{...}
        re.compile(
            r"\\captionsetup\{type=figure[^}]*\}\s*"
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{(?P<path>[^}]+)\}"
            r"(?P<between>.{0,500}?)"
            r"\\caption\{(?P<caption>[^}]+)\}",
            re.DOTALL,
        ),
    ]

    for pattern in captionof_patterns:
        for m in pattern.finditer(text_outside_figures):
            raw_path = m.group("path").strip()
            basename = Path(raw_path).name
            if not Path(basename).suffix:
                basename += ".jpg"
            if basename.lower() in seen_paths:
                continue
            seen_paths.add(basename.lower())
            caption = m.group("caption").strip()
            label_match = re.search(r"\\label\{([^}]+)\}", m.group(0))
            label = label_match.group(1).strip() if label_match else None
            figures.append(
                Figure(filename=basename, caption=caption, label=label, content=m.group(0))
            )

    logger.info("Extracted %d figures from %s", len(figures), latex_path)
    return figures


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


def _classify_section_name(section_name: str) -> str:
    """Classify a section name into a category. Returns the original name if no match."""
    normalized = _normalize_section_name(section_name)
    for category, keywords in SECTION_CATEGORY_RULES:
        if normalized in keywords:
            return category
        for kw in keywords:
            if kw in normalized:
                return category
    return section_name


def _find_section_at_position(position: int, section_map: list[tuple[int, str]]) -> str:
    """Return the section name that the given position belongs to."""
    current_section = "Preamble"
    for sec_pos, sec_name in section_map:
        if sec_pos > position:
            break
        current_section = sec_name
    return current_section


def build_section_position_map_from_classified(
    content: str,
    classified_sections: list | None = None,
) -> list[tuple[int, str]]:
    """
    Build a section position map from a list of classified sections.

    When classified_sections is provided, positions are identified by matching
    section names and classified category names are used. If not provided,
    falls back to rule-based classification.
    """
    if classified_sections is None:
        raise ValueError("classified_sections is required. Call classify_sections_for_paper first.")

    section_map: list[tuple[int, str]] = []

    abstract_match = re.search(r"\\begin\{abstract\}", content)
    if abstract_match:
        section_map.append((abstract_match.start(), "Abstract"))

    # name -> category name map from classified_sections
    # classified_sections is a list of Section dataclasses (name, content)
    classified_names = {s.name for s in classified_sections}

    section_pattern = r"\\section\{([^}]+)\}"
    for match in re.finditer(section_pattern, content):
        raw_name = match.group(1).strip()
        # Look up category name from classified sections.
        # Since classify_and_merge_sections merges sections of the same category,
        # determine which category the raw_name content belongs to.
        category = _classify_section_name(raw_name)
        if category in classified_names:
            section_map.append((match.start(), category))
        else:
            # Fallback: use the rule-based classification result as-is
            section_map.append((match.start(), category))

    section_map.sort(key=lambda x: x[0])
    return section_map


def extract_figure_references(
    latex_path: Path,
    classified_sections: list | None = None,
) -> list[FigureReference]:
    r"""
    Extract all occurrences of \ref{fig:...} from a LaTeX file.

    When classified_sections is provided, uses section names that match
    the classification results from classify_sections_for_paper.
    """
    content = latex_path.read_text(encoding="utf-8")

    doc_start = content.find(r"\begin{document}")
    doc_end = content.find(r"\end{document}")
    if doc_start != -1 and doc_end != -1:
        content = content[doc_start:doc_end]
    elif doc_start != -1:
        content = content[doc_start:]

    section_map = build_section_position_map_from_classified(content, classified_sections)
    references: list[FigureReference] = []

    ref_pattern = r"\\(?:auto)?[cC]?ref\{(fig:[^}]+)\}"
    for match in re.finditer(ref_pattern, content):
        label = match.group(1).strip()
        position = match.start()
        section = _find_section_at_position(position, section_map)
        ctx_start = max(0, position - 100)
        ctx_end = min(len(content), position + 100)
        context_snippet = content[ctx_start:ctx_end].replace("\n", " ").strip()
        references.append(
            FigureReference(label=label, section=section, context_snippet=context_snippet)
        )

    logger.info("Extracted %d figure references from %s", len(references), latex_path)
    return references


def match_gt_to_resource_filename(gt_filename: str, resource_filenames: list[str]) -> str | None:
    """Find the resource filename corresponding to a GT figure filename."""
    gt_stem = normalize_figure_filename(gt_filename)

    for res_fn in resource_filenames:
        if normalize_figure_filename(res_fn) == gt_stem:
            return res_fn

    for res_fn in resource_filenames:
        res_stem = normalize_figure_filename(res_fn)
        if res_stem.startswith(gt_stem) or gt_stem.startswith(res_stem):
            return res_fn

    return None


def match_figures_by_resource(
    gt_figures: list[Figure],
    pred_figures: list[Figure],
    resource_filenames: list[str],
) -> list[tuple[Figure | None, Figure | None, str]]:
    """Match GT and Pred figures using the resource filename as the canonical key."""
    pred_by_filename: dict[str, Figure] = {}
    for fig in pred_figures:
        pred_by_filename[fig.filename.lower()] = fig

    gt_by_resource: dict[str, Figure] = {}
    for fig in gt_figures:
        matched_resource = match_gt_to_resource_filename(fig.filename, resource_filenames)
        if matched_resource:
            gt_by_resource[matched_resource.lower()] = fig

    results: list[tuple[Figure | None, Figure | None, str]] = []
    for res_fn in resource_filenames:
        gt_fig = gt_by_resource.get(res_fn.lower())
        pred_fig = pred_by_filename.get(res_fn.lower())
        results.append((gt_fig, pred_fig, res_fn))

    return results


def evaluate_figure_coverage(
    resource_filenames: list[str],
    pred_figures: list[Figure],
    pred_references: list[FigureReference],
) -> list[FigureCoverage]:
    """Evaluate the usage status of each resource figure in the predicted paper."""
    pred_by_filename: dict[str, Figure] = {}
    for fig in pred_figures:
        pred_by_filename[fig.filename.lower()] = fig

    pred_ref_labels: set[str] = {ref.label for ref in pred_references}

    results: list[FigureCoverage] = []
    for res_fn in resource_filenames:
        pred_fig = pred_by_filename.get(res_fn.lower())
        included = pred_fig is not None
        referenced = False
        pred_label = None
        pred_caption = None

        if pred_fig:
            pred_label = pred_fig.label
            pred_caption = pred_fig.caption
            if pred_fig.label and pred_fig.label in pred_ref_labels:
                referenced = True

        results.append(
            FigureCoverage(
                filename=res_fn,
                included_in_paper=included,
                referenced_in_text=referenced,
                pred_label=pred_label,
                pred_caption=pred_caption,
            )
        )

    return results


def evaluate_figure_context_by_llm(
    model: str,
    filename: str,
    gt_figure: Figure,
    pred_figure: Figure,
    gt_refs: list[FigureReference],
    pred_refs: list[FigureReference],
    temperature: float = 0.3,
) -> FigureContextEvaluationResponse:
    """Evaluate the contextual appropriateness of a single figure using an LLM."""
    system_prompt = (
        "You are an expert paper reviewer evaluating whether a figure is referenced "
        "in appropriate context in a predicted paper compared to a ground truth paper.\n\n"
        "Both papers discuss the same research. The predicted paper may have a different "
        "section structure than the ground truth, so differences in section names are "
        "NOT automatically wrong. Focus on whether the figure is discussed in a "
        "contextually appropriate manner.\n\n"
        "Evaluate on a 1-5 scale:\n"
        "1 = Figure is referenced in completely wrong context\n"
        "2 = Figure is referenced in mostly inappropriate context\n"
        "3 = Figure is referenced in somewhat appropriate context\n"
        "4 = Figure is referenced in mostly appropriate context\n"
        "5 = Figure is referenced in perfectly appropriate context\n\n"
        "Respond in JSON format:\n"
        "```json\n"
        "{\n"
        '  "context_score": <1-5>,\n'
        '  "reasoning": "<explanation>",\n'
        '  "gt_context_summary": "<how GT uses this figure>",\n'
        '  "pred_context_summary": "<how Pred uses this figure>",\n'
        '  "context_appropriate": <true/false>\n'
        "}\n"
        "```"
    )

    gt_ref_desc = [f"  - Section '{ref.section}': ...{ref.context_snippet}..." for ref in gt_refs]
    gt_refs_text = (
        "\n".join(gt_ref_desc) if gt_ref_desc else "  (No explicit \\ref found in body text)"
    )

    pred_ref_desc = [
        f"  - Section '{ref.section}': ...{ref.context_snippet}..." for ref in pred_refs
    ]
    pred_refs_text = (
        "\n".join(pred_ref_desc) if pred_ref_desc else "  (No explicit \\ref found in body text)"
    )

    user_prompt = (
        f"Evaluate the context appropriateness of the following figure:\n\n"
        f"**Figure filename**: {filename}\n\n"
        f"**Ground Truth Usage:**\n"
        f"- Caption: {gt_figure.caption[:500]}\n"
        f"- Referenced in sections: {[ref.section for ref in gt_refs]}\n"
        f"- Reference contexts:\n{gt_refs_text}\n\n"
        f"**Predicted Paper Usage:**\n"
        f"- Caption: {pred_figure.caption[:500]}\n"
        f"- Referenced in sections: {[ref.section for ref in pred_refs]}\n"
        f"- Reference contexts:\n{pred_refs_text}\n\n"
        f"Is the predicted paper's usage of this figure contextually appropriate?"
    )

    response, _ = get_response_from_llm(
        model=model,
        user_message=user_prompt,
        system_message=system_prompt,
        temperature=temperature,
        response_format=FigureContextEvaluationResponse,
    )

    try:
        if isinstance(response, FigureContextEvaluationResponse):
            result = response
        else:
            result = FigureContextEvaluationResponse.model_validate_json(response)
    except ValidationError:
        logger.exception("Failed to parse figure context evaluation response")
        result = FigureContextEvaluationResponse(
            context_score=3,
            reasoning="Failed to parse evaluation",
            gt_context_summary="",
            pred_context_summary="",
            context_appropriate=True,
        )

    return result


def evaluate_figure_context(
    model: str,
    gt_figures: list[Figure],
    pred_figures: list[Figure],
    gt_references: list[FigureReference],
    pred_references: list[FigureReference],
    resource_filenames: list[str],
    temperature: float = 0.3,
) -> list[FigureContextComparison]:
    """Compare and evaluate figure reference contexts between GT and Pred."""
    matched_triples = match_figures_by_resource(gt_figures, pred_figures, resource_filenames)

    gt_refs_by_label: dict[str, list[FigureReference]] = {}
    for ref in gt_references:
        gt_refs_by_label.setdefault(ref.label, []).append(ref)

    pred_refs_by_label: dict[str, list[FigureReference]] = {}
    for ref in pred_references:
        pred_refs_by_label.setdefault(ref.label, []).append(ref)

    results: list[FigureContextComparison] = []

    for gt_fig, pred_fig, res_fn in matched_triples:
        if gt_fig is None:
            continue

        gt_refs = gt_refs_by_label.get(gt_fig.label, []) if gt_fig.label else []
        gt_sections = sorted({ref.section for ref in gt_refs})

        # Figure does not exist in Pred
        if pred_fig is None:
            results.append(
                FigureContextComparison(
                    filename=res_fn,
                    gt_label=gt_fig.label,
                    pred_label=None,
                    gt_reference_sections=gt_sections,
                    pred_reference_sections=[],
                    sections_match=False,
                    context_score=1,
                    reasoning="This figure does not exist in Pred",
                    context_appropriate=False,
                )
            )
            continue

        pred_refs = pred_refs_by_label.get(pred_fig.label, []) if pred_fig.label else []
        pred_sections = sorted({ref.section for ref in pred_refs})

        gt_normalized = sorted({_normalize_section_name(s) for s in gt_sections})
        pred_normalized = sorted({_normalize_section_name(s) for s in pred_sections})

        sections_match = gt_normalized == pred_normalized

        if sections_match:
            context_score = 5
            reasoning = "Referencing sections match GT"
            context_appropriate = True
        else:
            llm_result = evaluate_figure_context_by_llm(
                model=model,
                filename=res_fn,
                gt_figure=gt_fig,
                pred_figure=pred_fig,
                gt_refs=gt_refs,
                pred_refs=pred_refs,
                temperature=temperature,
            )
            context_score = llm_result.context_score
            reasoning = llm_result.reasoning
            context_appropriate = llm_result.context_appropriate

        results.append(
            FigureContextComparison(
                filename=res_fn,
                gt_label=gt_fig.label,
                pred_label=pred_fig.label if pred_fig else None,
                gt_reference_sections=gt_sections,
                pred_reference_sections=pred_sections,
                sections_match=sections_match,
                context_score=context_score,
                reasoning=reasoning,
                context_appropriate=context_appropriate,
            )
        )

    return results


def matching_and_evaluate_figures(
    gt_latex_path: Path,
    pred_latex_path: Path,
    figure_summary_path: Path,
    model: str,
    temperature: float = 0.3,
    gt_classified_sections: list | None = None,
    pred_classified_sections: list | None = None,
) -> tuple[list[FigureCoverage], list[FigureContextComparison], FigureSummary]:
    """
    Orchestration function for figure evaluation.

    When gt_classified_sections / pred_classified_sections are provided,
    uses section names that match the classification results from
    classify_sections_for_paper.
    """
    logger.info("=" * 80)
    logger.info("Starting figure evaluation")

    resource_filenames = parse_figure_summary(figure_summary_path)
    if not resource_filenames:
        logger.warning("No figures found in figure_summary.txt")
        empty_summary: FigureSummary = {
            "total_provided_figures": 0,
            "total_included_figures": 0,
            "total_referenced_figures": 0,
            "coverage_score": 0.0,
            "reference_score": 0.0,
            "average_context_score": 0.0,
            "total_matched_figures": 0,
        }
        return ([], [], empty_summary)

    gt_figures = extract_figures_from_latex(gt_latex_path)
    pred_figures = extract_figures_from_latex(pred_latex_path)

    gt_references = extract_figure_references(gt_latex_path, gt_classified_sections)
    pred_references = extract_figure_references(pred_latex_path, pred_classified_sections)

    coverage_results = evaluate_figure_coverage(resource_filenames, pred_figures, pred_references)

    total_included = sum(1 for c in coverage_results if c["included_in_paper"])
    total_referenced = sum(1 for c in coverage_results if c["referenced_in_text"])

    logger.info(
        "Figure coverage: %d/%d included, %d/%d referenced",
        total_included,
        len(resource_filenames),
        total_referenced,
        len(resource_filenames),
    )

    context_results = evaluate_figure_context(
        model,
        gt_figures,
        pred_figures,
        gt_references,
        pred_references,
        resource_filenames,
        temperature,
    )

    total_matched = len(context_results)
    total_context_score = sum(c["context_score"] for c in context_results)
    avg_context_score = total_context_score / total_matched if total_matched > 0 else 0.0

    figure_summary: FigureSummary = {
        "total_provided_figures": len(resource_filenames),
        "total_included_figures": total_included,
        "total_referenced_figures": total_referenced,
        "coverage_score": total_included / len(resource_filenames) if resource_filenames else 0.0,
        "reference_score": total_referenced / len(resource_filenames)
        if resource_filenames
        else 0.0,
        "average_context_score": avg_context_score,
        "total_matched_figures": total_matched,
    }

    logger.info(
        "Figure evaluation completed: coverage=%.2f, reference=%.2f, avg_context=%.2f",
        figure_summary["coverage_score"],
        figure_summary["reference_score"],
        figure_summary["average_context_score"],
    )
    logger.info("=" * 80)

    return (coverage_results, context_results, figure_summary)


class SectionFigureInfo(TypedDict, total=True):
    filename: str
    label: str | None
    caption: str
