from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from paper_recon.common.log import get_logger

logger = get_logger(__file__)

agents_mdS_DIR = Path(__file__).parent.parent / "writing" / "agents_md"


class AgentConfig(BaseModel):
    agent: Literal["ClaudeCode", "ClaudeCode_Teams", "Codex"]
    model: str
    max_turns: int


class LLMConfig(BaseModel):
    model: str
    temp: float | None = Field(default=None)
    max_tokens: int | None = Field(default=None)


class PaperDirConfig(BaseModel):
    original: Path
    original_tex: Path
    resources: Path
    agents_md: Path | None = None
    num_page: int = 8
    column_type: str = "single-column"


class Config(BaseModel):
    log_dir: Path
    base_codebase_dir: Path

    paper_dir: PaperDirConfig

    output_dir: Path

    research_overview_type: Literal["short", "long"] = "long"

    writeup: AgentConfig
    evaluation_llm: LLMConfig
    evaluation_agent: AgentConfig


def get_agents_md(paper_type: str) -> Path:
    """Get the AGENTS_*.md path based on paper type (method/benchmark/both)."""
    prompt_md = agents_mdS_DIR / f"AGENTS_{paper_type}.md"
    if not prompt_md.exists():
        logger.warning("AGENTS_%s.md not found, falling back to AGENTS_method.md", paper_type)
        prompt_md = agents_mdS_DIR / "AGENTS_method.md"
    return prompt_md


def load_paper_config(paper_dir: Path) -> dict:
    """Read original/config.yaml and return as dict."""
    config_path = paper_dir / "original" / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


def get_paper_type(paper_dir: Path) -> str:
    """Read paper type from original/config.yaml. Defaults to 'method'."""
    return load_paper_config(paper_dir).get("type", "method")


def find_tex(directory: Path) -> Path:
    """Find the main .tex file in the directory."""
    for name in ["main.tex", "paper.tex", "acl_latex.tex", "arXiv.tex"]:
        p = directory / name
        if p.exists():
            return p
    tex_files = list(directory.glob("*.tex"))
    if not tex_files:
        raise FileNotFoundError(f"No .tex files found in {directory}")
    return tex_files[0]


def build_config_for_paper(paper_name: str, base_config: dict) -> Config:
    """
    Build a full Config for a given paper name using shared agent settings.

    Args:
        paper_name: Name of the paper directory under PaperWrite-Bench/ (e.g. "paper_1")
        base_config: Shared config dict (agent settings, etc.)

    Returns:
        Config object with paper-specific paths filled in.

    """
    paper_root = Path("PaperWrite-Bench") / paper_name
    original_dir = paper_root / "original"
    resources_dir = paper_root / "resources"

    paper_config = load_paper_config(paper_root)
    paper_type = paper_config.get("type", "method")
    agents_md = get_agents_md(paper_type)
    original_tex = find_tex(original_dir)

    # column: "1column" -> "single-column", "2column" -> "double-column"
    raw_column = paper_config.get("column", "1column")
    column_type = "double-column" if raw_column.startswith("2") else "single-column"

    merged = {
        **base_config,
        "paper_dir": {
            "original": str(original_dir),
            "original_tex": str(original_tex),
            "resources": str(resources_dir),
            "agents_md": str(agents_md),
            "num_page": paper_config.get("num_page", 8),
            "column_type": column_type,
        },
        "output_dir": f"experiments/{paper_name}",
        "base_codebase_dir": str(resources_dir / "code"),
    }
    return Config.model_validate(merged)


def load_config(path: Path) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    assert path.exists()
    assert path.suffix in {".yaml", ".yml"}

    return Config.model_validate(yaml.load(path.read_text(), Loader=yaml.FullLoader))


def load_base_config(path: Path) -> dict:
    """Load a base config YAML (without paper-specific paths) as a dict."""
    assert path.exists()
    assert path.suffix in {".yaml", ".yml"}

    return yaml.load(path.read_text(), Loader=yaml.FullLoader)
