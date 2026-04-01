"""Download PaperWrite-Bench from HuggingFace and reconstruct local directory structure."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from paper_recon.common.log import get_logger

logger = get_logger(__file__)

DEFAULT_REPO_ID = "your-org/PaperWrite-Bench"
DEFAULT_CACHE_DIR = Path("PaperWrite-Bench")


def download_from_hf(
    repo_id: str = DEFAULT_REPO_ID,
    output_dir: Path = DEFAULT_CACHE_DIR,
    paper_names: list[str] | None = None,
) -> Path:
    """Download PaperWrite-Bench from HuggingFace and reconstruct the directory structure.

    Args:
        repo_id: HuggingFace dataset repository ID.
        output_dir: Local directory to write the reconstructed benchmark.
        paper_names: If given, only download these papers. None means all.

    Returns:
        Path to the output directory (same structure as PaperWrite-Bench/).
    """
    from datasets import load_dataset

    logger.info("Loading dataset from HuggingFace: %s", repo_id)
    ds = load_dataset(repo_id, split="test")

    for sample in ds:
        paper_id = sample["paper_id"]
        if paper_names and paper_id not in paper_names:
            continue

        paper_dir = output_dir / paper_id
        if paper_dir.exists() and (paper_dir / "resources" / "template.tex").exists():
            logger.info("Skipping %s: already exists", paper_id)
            continue

        logger.info("Reconstructing %s ...", paper_id)
        _reconstruct_paper(paper_dir, sample)

    return output_dir


def _reconstruct_paper(paper_dir: Path, sample: dict) -> None:
    """Reconstruct a single paper's directory structure from a dataset sample."""
    original = paper_dir / "original"
    resources = paper_dir / "resources"
    original.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    # original/config.yaml
    config_lines = [
        f"type: {sample['type']}",
        f"num_page: {sample['num_page']}",
        f"column: {sample['column']}",
        f"conference: {sample['conference']}",
    ]
    (original / "config.yaml").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    # original/main.tex
    _write_text(original / "main.tex", sample["gt_tex"])

    # original/main.pdf
    if sample.get("gt_pdf"):
        (original / "main.pdf").write_bytes(sample["gt_pdf"])

    # resources/ text files
    _write_text(resources / "template.tex", sample["template_tex"])
    _write_text(resources / "research_overview_short.md", sample["research_overview_short"])
    _write_text(resources / "research_overview_long.md", sample["research_overview_long"])
    _write_text(resources / "references.bib", sample["references_bib"])
    _write_text(resources / "figure_summary.txt", sample["figure_summary"])
    _write_text(resources / "table_summary.txt", sample["table_summary"])
    if sample.get("eval_points"):
        _write_text(resources / "eval_points.json", sample["eval_points"])

    # resources/figures/
    fig_dir = resources / "figures"
    fig_dir.mkdir(exist_ok=True)
    for filename, image in zip(sample["figure_filenames"], sample["figure_images"]):
        ext = Path(filename).suffix.lower()
        fmt = "PNG" if ext == ".png" else "JPEG"
        image.save(fig_dir / filename, format=fmt)

    # resources/tables/
    tab_dir = resources / "tables"
    tab_dir.mkdir(exist_ok=True)
    for filename, content in zip(sample["table_filenames"], sample["table_contents"]):
        _write_text(tab_dir / filename, content)

    # resources/code/ (extract tar.gz)
    if sample.get("has_code") and sample.get("code_tar_gz"):
        code_bytes = sample["code_tar_gz"]
        with tarfile.open(fileobj=io.BytesIO(code_bytes), mode="r:gz") as tar:
            tar.extractall(path=str(resources), filter="data")


def _write_text(path: Path, content: str) -> None:
    if content:
        path.write_text(content, encoding="utf-8")
