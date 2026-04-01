import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paper_recon.common.config import Config, build_config_for_paper, load_base_config, load_config
from paper_recon.common.log import get_logger, set_log_dir
from paper_recon.writing.perform_writeup import perform_writeup_with_agent

logger = get_logger(__file__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")

    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/cc_sonnet4.yaml",
        help="Path to the config file",
    )

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

    parser.add_argument(
        "--from-hf",
        type=str,
        nargs="?",
        const="hal-utokyo/PaperWrite-Bench",
        default="hal-utokyo/PaperWrite-Bench",
        metavar="REPO_ID",
        help="Download PaperWrite-Bench from HuggingFace (default repo: your-org/PaperWrite-Bench)",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="If set, skip writeup",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )

    return parser.parse_args()


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


def run_single_paper(config: Config, args: argparse.Namespace, paper_name: str) -> None:
    """Run writeup for a single paper."""
    agent_name = config.writeup.agent
    model_name = config.writeup.model.split("/")[-1]
    overview_type = config.research_overview_type
    paper_parent_dir = Path("experiments") / overview_type / agent_name / model_name / paper_name

    if paper_parent_dir.exists() and any(paper_parent_dir.iterdir()):
        logger.info("Skipping %s: already exists at %s", paper_name, paper_parent_dir)
        return

    timestamp = datetime.now(tz=timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    output_dir = paper_parent_dir / timestamp
    latex_dir = output_dir / "latex"
    pdf_dir = output_dir / "pdf"
    latex_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(config.log_dir / overview_type / agent_name / model_name / paper_name / timestamp)

    logger.info("=== Processing paper: %s ===", paper_name)

    if not args.skip_writeup:
        idea_dir = config.paper_dir.resources

        writeup_success = False

        references_bib = idea_dir / "references.bib"
        if references_bib.exists():
            citations_text = references_bib.read_text(encoding="utf-8")
            logger.info("Loaded references.bib (%d chars)", len(citations_text))
        else:
            logger.error("references.bib not found in %s. Run get_papers.py first.", idea_dir)
            return

        for attempt in range(args.writeup_retries):
            logger.info("Writeup attempt %d of %d", attempt + 1, args.writeup_retries)
            writeup_success = perform_writeup_with_agent(
                agents_md=config.paper_dir.agents_md,
                agent_config=config.writeup,
                resources_folder=config.paper_dir.resources,
                citations_text=citations_text,
                latex_folder=latex_dir,
                pdf_folder=pdf_dir,
                research_overview_type=config.research_overview_type,
                num_page=config.paper_dir.num_page,
                column_type=config.paper_dir.column_type,
            )
            if writeup_success:
                break

        if not writeup_success:
            logger.warning("Writeup process did not complete successfully after all retries.")


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["PAPER_RECON_ROOT"] = str(Path(__file__).parent)
    logger.debug("Set PAPER_RECON_ROOT to %s", os.environ["PAPER_RECON_ROOT"])

    config_path = Path(args.config_path)
    paper_names = get_paper_names(args)

    if args.from_hf:
        from paper_recon.common.hf_download import download_from_hf

        download_from_hf(
            repo_id=args.from_hf,
            paper_names=paper_names if paper_names else None,
        )

    if paper_names:
        base_config = load_base_config(config_path)
        for paper_name in paper_names:
            try:
                config = build_config_for_paper(paper_name, base_config)
                run_single_paper(config, args, paper_name)
            except Exception:
                logger.exception("Failed to process paper: %s", paper_name)
    else:
        config = load_config(config_path)
        paper_name = config.output_dir.name
        run_single_paper(config, args, paper_name)
