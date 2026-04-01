#!/usr/bin/env python3
"""
Standalone script to run evaluate_paper only.

Usage:
    # Evaluate a specific paper (auto-detects the latest experiment directory)
    python run_evaluation.py --config-path configs/cc_sonnet4.yaml --paper paper_1

    # Evaluate all papers
    python run_evaluation.py --config-path configs/cc_sonnet4.yaml --all

    # Explicitly specify the target directory
    python run_evaluation.py --eval-target-dir experiments/short/ClaudeCode/.../latex
"""

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from paper_recon.common.config import Config, build_config_for_paper, load_base_config
from paper_recon.common.log import get_logger, set_log_dir
from paper_recon.evaluation.evaluate_per_section import evaluate_paper

logger = get_logger(__file__)


def resolve_latex_inputs(tex_path: Path) -> str:
    """Resolve all \\input{} directives in a LaTeX file by inlining referenced file contents."""
    tex_dir = tex_path.parent
    content = tex_path.read_text(encoding="utf-8")

    pattern = re.compile(r"\\input\{([^}]+)\}")

    def _replace(match: re.Match) -> str:
        target = match.group(1)
        target_path = tex_dir / target
        if not target_path.suffix:
            target_path = target_path.with_suffix(".tex")
        if target_path.is_file():
            logger.debug("Inlining \\input{%s} from %s", target, target_path)
            return target_path.read_text(encoding="utf-8")
        logger.warning("\\input{%s} referenced file not found: %s", target, target_path)
        return match.group(0)

    return pattern.sub(_replace, content)


def find_latest_experiment_dir(config: Config, paper_name: str) -> Path | None:
    """Find the latest timestamped directory for the specified paper under experiments/."""
    agent_name = config.writeup.agent
    model_name = config.writeup.model.split("/")[-1]
    overview_type = config.research_overview_type
    paper_dir = Path("experiments") / overview_type / agent_name / model_name / paper_name

    if not paper_dir.exists():
        return None

    # Find timestamp directories (YYYYMMDD_HHMMSS) and return the latest one
    timestamp_dirs = sorted(
        (d for d in paper_dir.iterdir() if d.is_dir() and d.name[:8].isdigit()),
        key=lambda d: d.name,
        reverse=True,
    )
    if not timestamp_dirs:
        return None

    latest = timestamp_dirs[0]
    # Return the latex/ subdirectory if it exists
    latex_dir = latest / "latex"
    if latex_dir.is_dir():
        return latex_dir
    return latest


def get_paper_names(args: argparse.Namespace) -> list[str]:
    """Resolve which papers to process from CLI arguments."""
    papers_root = Path("PaperWrite-Bench")

    if args.all:
        return sorted(
            d.name for d in papers_root.iterdir() if d.is_dir() and (d / "resources").is_dir()
        )
    if args.paper:
        return args.paper
    return []


def _build_eval_params(
    args: argparse.Namespace,
    eval_target_dir: Path,
    timestamp_str: str,
    eval_points_path: Path,
    figure_summary_path: Path,
) -> dict:
    """Return evaluate_paper parameters based on the evaluation mode."""
    mode = args.eval_mode
    prefix = {
        "citation": "citation_results",
        "hallucination": "hallucination_results",
        "rubric": "rubric_results",
        "all": "evaluation_results",
    }[mode]
    return {
        "output_path": eval_target_dir / f"{prefix}_{timestamp_str}.json",
        "eval_points_path": eval_points_path if eval_points_path.exists() else None,
        "figure_summary_path": figure_summary_path if figure_summary_path.exists() else None,
        "eval_mode": mode,
    }


def _needs_hallucination(args: argparse.Namespace) -> bool:
    """Determine whether hallucination verification is required."""
    return args.eval_mode in ("hallucination", "all")


def run_evaluation_for_paper(config: Config, paper_name: str, args: argparse.Namespace) -> None:
    """Run evaluation for a single paper."""
    eval_target_dir = find_latest_experiment_dir(config, paper_name)
    if eval_target_dir is None:
        logger.warning("No experiment directory found for %s. Skipping.", paper_name)
        return

    timestamp_dir = eval_target_dir.parent if eval_target_dir.name == "latex" else eval_target_dir
    eval_dir = timestamp_dir / "eval"

    # Skip check: determine by existence of result files for the given mode
    if not args.force:
        skip_patterns = {
            "citation": "citation_results_*.json",
            "hallucination": "hallucination_results_*.json",
            "rubric": "rubric_results_*.json",
            "all": "evaluation_results_*.json",
        }
        pattern = skip_patterns.get(args.eval_mode)
        if pattern and list(eval_target_dir.glob(pattern)):
            logger.info("Skipping %s (%s already exists)", paper_name, pattern)
            return

    logger.info("=== Evaluating paper: %s ===", paper_name)
    logger.info("Eval target: %s", eval_target_dir)

    # Set up the log directory
    agent_name = config.writeup.agent
    model_name = config.writeup.model.split("/")[-1]
    overview_type = config.research_overview_type
    timestamp = (
        eval_target_dir.parent.name if eval_target_dir.name == "latex" else eval_target_dir.name
    )
    log_dir = config.log_dir / overview_type / agent_name / model_name / paper_name / timestamp
    set_log_dir(log_dir)

    # Expand \input{} directives and create template_eval.tex
    template_path = eval_target_dir / "template.tex"
    if not template_path.exists():
        logger.error("template.tex not found in %s. Skipping.", eval_target_dir)
        return

    resolved_content = resolve_latex_inputs(template_path)
    pred_latex_path = eval_target_dir / "template_eval.tex"
    pred_latex_path.write_text(resolved_content, encoding="utf-8")
    logger.info("Created resolved LaTeX for evaluation: %s", pred_latex_path)

    # Create eval/ directory (only needed for hallucination verification)
    if _needs_hallucination(args):
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        shutil.copytree(
            eval_target_dir,
            eval_dir,
            ignore=shutil.ignore_patterns(
                "evaluation_results_*.json", "citation_results_*.json", "rubric_results_*.json"
            ),
        )
        gt_main = config.paper_dir.original_tex
        shutil.copy2(gt_main, eval_dir / "gt_main.tex")
        logger.info("Created eval directory: %s", eval_dir)
        hal_dir = eval_dir
    else:
        hal_dir = None

    figure_summary_path = config.paper_dir.resources / "figure_summary.txt"
    eval_points_path = config.paper_dir.resources / "eval_points.json"

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_params = _build_eval_params(
        args, eval_target_dir, timestamp_str, eval_points_path, figure_summary_path
    )

    evaluate_paper(
        gt_latex_path=config.paper_dir.original_tex,
        pred_latex_path=pred_latex_path,
        gt_codebase_dir=config.base_codebase_dir,
        llm_config=config.evaluation_llm,
        agent_config=config.evaluation_agent,
        hal_verification_dir=hal_dir,
        **mode_params,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation only")

    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/cc_sonnet4.yaml",
        help="Path to the config file",
    )

    # Paper selection
    paper_group = parser.add_mutually_exclusive_group()
    paper_group.add_argument(
        "--all",
        action="store_true",
        help="Run for all papers in PaperWrite-Bench/ directory",
    )
    paper_group.add_argument(
        "--paper",
        type=str,
        nargs="+",
        default=None,
        help="Run for specific paper(s) (e.g. paper_1 locoop)",
    )
    paper_group.add_argument(
        "--eval-target-dir",
        type=str,
        default=None,
        help="Explicitly specify the evaluation target directory (backward compatible)",
    )

    parser.add_argument(
        "--from-hf",
        type=str,
        nargs="?",
        const="your-org/PaperWrite-Bench",
        default=None,
        metavar="REPO_ID",
        help="Download PaperWrite-Bench from HuggingFace (default repo: your-org/PaperWrite-Bench)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval-mode",
        choices=["rubric", "hallucination", "citation", "all"],
        default="rubric",
        help="Evaluation mode: rubric, hallucination, citation, all (default: rubric)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-evaluate even if eval/ already exists (do not skip)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["PAPER_RECON_ROOT"] = str(Path(__file__).parent)

    # Backward compatible: when --eval-target-dir is directly specified
    if args.eval_target_dir:
        eval_target_dir = Path(args.eval_target_dir)
        set_log_dir(eval_target_dir / "logs")

        template_path = eval_target_dir / "template.tex"
        resolved_content = resolve_latex_inputs(template_path)
        pred_latex_path = eval_target_dir / "template_eval.tex"
        pred_latex_path.write_text(resolved_content, encoding="utf-8")

        base_config = load_base_config(Path(args.config_path))
        # Infer the paper name from eval-target-dir
        # experiments/short/ClaudeCode/model/paper_name/timestamp/latex
        parts = eval_target_dir.parts
        paper_name = None
        for i, part in enumerate(parts):
            if part.startswith("20") and len(part) == 15:  # timestamp
                paper_name = parts[i - 1]
                break
        # If no timestamp is found, search by directory names that exist in papers/
        if not paper_name:
            papers_root = Path("PaperWrite-Bench")
            known_papers = (
                {d.name for d in papers_root.iterdir() if d.is_dir()}
                if papers_root.exists()
                else set()
            )
            for part in parts:
                if part in known_papers:
                    paper_name = part
                    break
        if paper_name:
            config = build_config_for_paper(paper_name, base_config)
        else:
            logger.error("Could not determine paper name from %s", eval_target_dir)
            raise SystemExit(1)

        figure_summary_path = config.paper_dir.resources / "figure_summary.txt"
        eval_points_path = config.paper_dir.resources / "eval_points.json"

        # Create eval/ directory (only needed for hallucination verification)
        timestamp_dir = (
            eval_target_dir.parent if eval_target_dir.name == "latex" else eval_target_dir
        )
        eval_dir = timestamp_dir / "eval"
        if _needs_hallucination(args):
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
            shutil.copytree(
                eval_target_dir,
                eval_dir,
                ignore=shutil.ignore_patterns(
                    "evaluation_results_*.json", "citation_results_*.json", "rubric_results_*.json"
                ),
            )
            shutil.copy2(config.paper_dir.original_tex, eval_dir / "gt_main.tex")
            hal_dir = eval_dir
        else:
            hal_dir = None

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_params = _build_eval_params(
            args, eval_target_dir, timestamp_str, eval_points_path, figure_summary_path
        )

        evaluate_paper(
            gt_latex_path=config.paper_dir.original_tex,
            pred_latex_path=pred_latex_path,
            gt_codebase_dir=config.base_codebase_dir,
            llm_config=config.evaluation_llm,
            agent_config=config.evaluation_agent,
            hal_verification_dir=hal_dir,
            **mode_params,
        )
    else:
        # New format: --paper / --all
        config_path = Path(args.config_path)
        paper_names = get_paper_names(args)

        if args.from_hf:
            from paper_recon.common.hf_download import download_from_hf

            download_from_hf(
                repo_id=args.from_hf,
                paper_names=paper_names if paper_names else None,
            )

        if not paper_names:
            logger.error("No papers specified. Use --paper or --all.")
            raise SystemExit(1)

        base_config = load_base_config(config_path)
        for paper_name in paper_names:
            try:
                config = build_config_for_paper(paper_name, base_config)
                run_evaluation_for_paper(config, paper_name, args)
            except Exception:
                logger.exception("Failed to evaluate paper: %s", paper_name)
