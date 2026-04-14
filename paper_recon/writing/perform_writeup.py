import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from subprocess import CalledProcessError
from typing import TypedDict

from paper_recon.common.coding_agent import run_agent
from paper_recon.common.config import AgentConfig
from paper_recon.common.log import get_logger

# review
from paper_recon.writing.writeup_prompt import (
    page_limit_prompt_template,
    reflection_prompt_template,
    writeup_agent_prompt_template,
)

logger = get_logger(__file__)


WATERMARK_BLOCK = (
    "\\usepackage{draftwatermark}\n"
    "\\SetWatermarkText{AI-Generated}\n"
    "\\SetWatermarkScale{0.5}\n"
    "\\SetWatermarkColor[gray]{0.60}\n"
)


def ensure_watermark(latex_file: Path) -> None:
    """Ensure the draftwatermark block exists in the LaTeX file; add it if missing."""
    content = latex_file.read_text(encoding="utf-8")
    has_package = re.search(r"\\usepackage(?:\[[^\]]*\])?\{draftwatermark\}", content)
    has_text = re.search(r"\\SetWatermarkText\{", content)
    has_scale = re.search(r"\\SetWatermarkScale\{", content)
    has_color = re.search(r"\\SetWatermarkColor", content)
    if has_package and has_text and has_scale and has_color:
        return

    doc_match = re.search(r"\\documentclass(?:\[[^\]]*\])?\{[^}]+\}\s*\n?", content)
    if doc_match:
        insert_pos = doc_match.end()
        content = content[:insert_pos] + "\n" + WATERMARK_BLOCK + content[insert_pos:]
    else:
        content = WATERMARK_BLOCK + content
    latex_file.write_text(content, encoding="utf-8")
    logger.info("Restored watermark block in %s", latex_file)


def compile_latex(
    cwd: Path, latex_file: Path, pdf_file: Path, timeout: int = 30
) -> tuple[bool, str]:
    """
    Compile latex file.

    Args:
        cwd (Path): working directory containing the LaTeX files
        latex_file (Path): LaTeX file to compile
        pdf_file (Path): output PDF file path
        timeout (int, optional): timeout for each LaTeX command in seconds. Defaults to 30.

    Returns:
        tuple[bool, str]: (success, error_output)

    """
    logger.info("GENERATING LATEX")

    error_output = ""
    tex_name = latex_file.stem
    commands = [
        ["pdflatex", "-interaction=nonstopmode", f"{tex_name}.tex"],
        ["bibtex", tex_name],
        ["pdflatex", "-interaction=nonstopmode", f"{tex_name}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{tex_name}.tex"],
    ]

    for command in commands:
        is_bibtex = command[0] == "bibtex"
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                if is_bibtex:
                    # bibtex returns non-zero even for warnings; detect only fatal errors
                    fatal_keywords = [
                        "I couldn't open",
                        "illegal end of database",
                        "repeated entry",
                    ]
                    has_fatal = any(kw in result.stdout for kw in fatal_keywords)
                    if has_fatal:
                        logger.info("BibTeX fatal error:\n%s", result.stdout)
                        error_output = result.stdout
                    else:
                        logger.info("BibTeX warnings (ignored).")
                else:
                    logger.info("Standard Output:\n%s", result.stdout)
                    logger.info("Standard Error:\n%s", result.stderr)
                    error_output = result.stderr + result.stdout
            else:
                logger.info("LaTeX command completed successfully.")

        except subprocess.TimeoutExpired:
            logger.exception(
                "EXCEPTION in compile_latex: LaTeX timed out after %d seconds.", timeout
            )
            error_output = f"LaTeX timed out after {timeout} seconds."
        except subprocess.CalledProcessError:
            logger.exception(
                "EXCEPTION in compile_latex: Error running command %s", " ".join(command)
            )
            error_output = f"Error running command {' '.join(command)}"

    logger.info("FINISHED GENERATING LATEX")

    if not error_output:
        try:
            shutil.move(cwd / (latex_file.stem + ".pdf"), pdf_file)
        except FileNotFoundError:
            logger.exception("Failed to rename PDF.")
            return False, "PDF file not found after compilation."
        return True, ""

    # Remove stale PDF left in cwd after compilation failure
    stale_pdf = cwd / (latex_file.stem + ".pdf")
    if stale_pdf.exists():
        stale_pdf.unlink()
        logger.info("Removed stale PDF after compilation failure: %s", stale_pdf)

    return False, error_output


def reflect_until_compiles(
    latex_file: Path,
    latex_folder: Path,
    pdf_file: Path,
    agent_config: AgentConfig,
    max_reflection: int = 5,
) -> bool:
    """
    Run compile-reflect loop until LaTeX compiles without errors.

    Args:
        latex_file: Path to the LaTeX file.
        latex_folder: Working directory for the agent.
        pdf_file: Output PDF path.
        agent_config: Agent configuration.
        max_reflection: Maximum number of reflection rounds.

    Returns:
        True if compilation succeeded, False otherwise.

    """
    for i in range(max_reflection):
        logger.info("Reflection round %d of %d", i + 1, max_reflection)

        check_output = os.popen(f"chktex {latex_file} -q -n2 -n24 -n13 -n1").read()

        compile_success, compile_output = compile_latex(
            cwd=latex_folder, latex_file=latex_file, pdf_file=pdf_file
        )

        if compile_success:
            if check_output.strip():
                logger.info(
                    "PDF generated successfully. Remaining chktex warnings (ignored):\n%s",
                    check_output.strip(),
                )
            else:
                logger.info("No errors found.")
            logger.info("Skipping further reflection.")
            return True

        logger.info("Compilation failed, running reflection agent...")
        reflection_prompt = reflection_prompt_template.format(
            check_output=check_output,
            compile_output=compile_output,
        )

        reflection_agent = (
            "ClaudeCode" if agent_config.agent == "ClaudeCode_Teams" else agent_config.agent
        )
        run_agent(
            agent=reflection_agent,
            user_prompt=reflection_prompt,
            working_dir=latex_folder,
            model=agent_config.model,
            max_turns=agent_config.max_turns,
            mode="READWRITE",
        )

    # Final compile after last reflection
    compile_success, compile_output = compile_latex(
        cwd=latex_folder, latex_file=latex_file, pdf_file=pdf_file
    )
    if not compile_success:
        logger.warning(
            "LaTeX still has errors after %d reflections: %s",
            max_reflection,
            compile_output,
        )
    return compile_success


def adjust_page_limit(
    latex_file: Path,
    latex_folder: Path,
    pdf_file: Path,
    agent_config: AgentConfig,
    page_limit: int = 8,
    max_rounds: int = 10,
) -> bool:
    """
    Adjust the paper to fit within the page limit.

    Compiles the PDF, checks if the main text exceeds the page limit,
    and runs the agent to shorten the text if needed.

    Args:
        latex_file: Path to the LaTeX file.
        latex_folder: Working directory for the agent.
        pdf_file: Output PDF path.
        agent_config: Agent configuration.
        page_limit: Maximum pages for main text (before References).
        max_rounds: Maximum number of adjustment rounds.

    Returns:
        True if the paper fits within the page limit, False otherwise.

    """
    for i in range(max_rounds):
        logger.info("Page limit adjustment round %d of %d", i + 1, max_rounds)

        # Compile to get current PDF
        compile_success, _ = compile_latex(
            cwd=latex_folder, latex_file=latex_file, pdf_file=pdf_file
        )
        if not compile_success:
            logger.warning("Compilation failed during page limit adjustment. Attempting to fix...")
            fixed = reflect_until_compiles(
                latex_file=latex_file,
                latex_folder=latex_folder,
                pdf_file=pdf_file,
                agent_config=agent_config,
            )
            if not fixed:
                logger.warning("Could not fix compilation errors during page limit adjustment.")
                return False

        # Check page limit
        info = check_page_limit(pdf_file, page_limit)
        if info is None:
            logger.warning("Could not detect References position. Skipping page adjustment.")
            return True

        ref_page = info["ref_page"]
        deficit = page_limit - ref_page  # positive means under, negative means over

        if info["excess"] is None and deficit < 1:
            logger.info("Paper fits within %d-page limit.", page_limit)
            return True

        if info["excess"] is not None:
            # Over limit: reduce one page at a time (on the final round, go straight to the target)
            if i == max_rounds - 1:
                target_limit = page_limit
            else:
                target_limit = ref_page - 1
            # target_limit = page_limit
            over = ref_page - target_limit
            status = f"{over} page(s) long"
            action = "shorten it"
            logger.info(
                "Main text is %d pages, target is %d pages. Running agent to shorten...",
                ref_page,
                target_limit,
            )
        else:
            # Under limit: expand to the target page count
            target_limit = page_limit
            status = f"{deficit} page(s) short"
            action = "expand it with substantive content"
            logger.info(
                "Paper is %d pages, %d page(s) shorter than %d-page target. Running agent to expand...",
                ref_page,
                deficit,
                page_limit,
            )

        prompt = page_limit_prompt_template.format(
            main_pages=ref_page,
            page_limit=target_limit,
            status=status,
            action=action,
        )

        page_agent = (
            "ClaudeCode" if agent_config.agent == "ClaudeCode_Teams" else agent_config.agent
        )
        run_agent(
            agent=page_agent,
            user_prompt=prompt,
            working_dir=latex_folder,
            model=agent_config.model,
            max_turns=agent_config.max_turns,
            mode="READWRITE",
        )

    # Final check
    compile_success, _ = compile_latex(cwd=latex_folder, latex_file=latex_file, pdf_file=pdf_file)
    if not compile_success:
        return False
    info = check_page_limit(pdf_file, page_limit)
    if info is not None and info["excess"] is not None:
        logger.warning("Paper still exceeds %d-page limit after %d rounds.", page_limit, max_rounds)
        return False
    return True


def is_header_or_footer(line: str) -> bool:
    """
    Return True if the line is likely a header or footer.

    Filters out:
      - Lines that are too short (< 4 characters after stripping).
      - Lines that are only digits.
      - Lines starting with known phrases (e.g., "Under review").
      - Lines that consist solely of capital letters and spaces.

    Args:
        line (str): input line of text

    Returns:
        bool: True if the line is likely a header or footer, False otherwise

    """
    line_stripped = line.strip()
    if len(line_stripped) < 1:
        return True

    header_footer_patterns = [
        r"^\d+$",  # Only digits (e.g., page numbers like "000", "001", etc.)
        r"^Under review",  # Lines starting with "Under review"
    ]
    return any(re.match(pattern, line_stripped) for pattern in header_footer_patterns)


def clean_lines(content: str) -> list[str]:
    """
    Clean lines by removing headers and footers.

    Args:
        content (str): raw text content extracted from PDF

    Returns:
        list[str]: cleaned list of lines without headers/footers

    """
    lines = content.splitlines()
    # Keep only lines that are not detected as headers/footers.
    return [line for line in lines if not is_header_or_footer(line)]


def detect_references_position_clean(pdf_file: Path) -> tuple[int, int] | None:
    """
    Detect the position of the "References" section in the PDF.

    Locate the last occurrence of the word "References" (or variations like "R EFERENCES") within
    the cleaned content extracted from the PDF.
    Uses pdftotext with layout preservation and cleans the extracted text.

    Returns a tuple (ref_page, ref_line) if found (with ref_line counting only
    the cleaned lines), otherwise None.

    Args:
        pdf_file (Path): path to the PDF file

    Returns:
        tuple[int, int] | None: (page number, cleaned line number) of "References" or \
            None if not found

    """
    if not pdf_file.exists():
        return None

    # Compile a regex pattern to match "REFERENCES" even if there are extra spaces
    # between letters (and do a case-insensitive match).
    pattern = re.compile(r"\bR\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S\b", re.IGNORECASE)

    # Loop through pages (limit to 50 pages by default)
    # Keep track of the last occurrence found
    last_found: tuple[int, int] | None = None
    for page in range(1, 51):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            temp_dir = Path(tempfile.mkdtemp())
            page_txt = temp_dir / f"page_{page}.txt"
            try:
                subprocess.run(
                    [
                        "pdftotext",
                        "-layout",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-q",
                        pdf_file,
                        page_txt,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if not page_txt.exists():
                    break
                content = page_txt.read_text(encoding="utf-8", errors="ignore")
            except CalledProcessError:
                # logger.exception("Error running pdftotext for page %d", page)
                continue

        # Clean the lines before searching for "References"
        cleaned = clean_lines(content)
        for idx, line in enumerate(cleaned):
            if pattern.search(line):
                last_found = (page, idx + 1)
    return last_found


def extract_page_line_counts(pdf_file: Path, first_page: int, last_page: int) -> dict[int, int]:
    """
    Extract the number of cleaned text lines for each page from first_page to last_page.

    This uses pdftotext with layout preservation and the clean_lines helper.
    Returns a dictionary {page_number: number_of_cleaned_lines}.
    Pages for which extraction fails are omitted.

    Args:
        pdf_file (Path): path to the PDF file
        first_page (int): first page number to extract
        last_page (int): last page number to extract

    Returns:
        dict[int, int]: dictionary mapping page numbers to number of cleaned lines

    """
    page_lines = {}
    for page in range(first_page, last_page + 1):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            page_txt = temp_dir / f"page_{page}.txt"
            try:
                subprocess.run(
                    [
                        "pdftotext",
                        "-layout",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-q",
                        pdf_file,
                        page_txt,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if not page_txt.exists():
                    break
                content = page_txt.read_text(encoding="utf-8", errors="ignore")
            except CalledProcessError:
                logger.exception("Error running pdftotext for page %d", page)
                continue
        # Clean the extracted text and count the number of remaining lines.
        cleaned = clean_lines(content)
        page_lines[page] = len(cleaned)
    return page_lines


class PageLimitInfo(TypedDict):
    ref_page: int
    ref_line: int
    used_lines: int
    allowed_lines: int
    excess: int | None
    available: int | None


def check_page_limit(pdf_file: Path, page_limit: int = 4) -> PageLimitInfo | None:
    """
    Check if the main text (before "References") exceeds the allowed number of pages.

    Compile the LaTeX project in a temporary folder, then determine where the
    "References" section begins using cleaned text extraction.
    Next, count the number of cleaned text lines used before the word "References" and compare that
    to the total number of cleaned lines available in the allowed number of pages (page_limit).

    Args:
        pdf_file (Path): path to the compiled PDF file
        page_limit (int, optional): allowed number of pages for main text. Defaults to 4.

    Returns:
        PageLimitInfo | None: dictionary with the following keys if "References" is found:
            - 'ref_page': page number where "References" was found (or None)
            - 'ref_line': cleaned line number within that page (or None)
            - 'used_lines': number of cleaned lines used for main content (before "References")
            - 'allowed_lines': total number of cleaned text lines available in pages 1..page_limit
            - 'excess': if used_lines > allowed_lines (number of lines over the limit),
            - 'available': if used_lines < allowed_lines (number of lines still available)
            If compilation or extraction fails, returns None.

    """
    try:
        # Ensure the PDF was produced
        if not pdf_file.exists():
            return None

        # Locate the first occurrence of "References" using the cleaned extraction
        ref_pos = detect_references_position_clean(pdf_file)
        if ref_pos is None:
            # If "References" isn't found, assume no reference section exists.
            return None
        ref_page, ref_line = ref_pos

        # Determine up to which page we need to extract cleaned line counts:
        max_page_to_extract = max(page_limit, ref_page)
        page_line_counts = extract_page_line_counts(pdf_file, 1, max_page_to_extract)
        if not page_line_counts:
            return None

        # Compute total cleaned lines available in the allowed pages (pages 1 to page_limit)
        allowed_lines = sum(page_line_counts.get(page, 0) for page in range(1, page_limit + 1))

        # Compute cleaned lines used before "References":
        used_lines = 0
        # Sum full pages before the reference page
        for page in range(1, ref_page):
            used_lines += page_line_counts.get(page, 0)
        # Add lines from the reference page up to (but not including) the line
        # where "References" appears
        used_lines += ref_line - 1

        result: PageLimitInfo = {
            "ref_page": ref_page,
            "ref_line": ref_line,
            "used_lines": used_lines,
            "allowed_lines": allowed_lines,
            "excess": None,
            "available": None,
        }
        if used_lines > allowed_lines:
            result["excess"] = used_lines - allowed_lines
        else:
            result["available"] = allowed_lines - used_lines

    except Exception:
        logger.exception("Error checking page limit")
        return None

    else:
        return result


def load_research_overview_text(folder: Path) -> str:
    """
    Load the idea text from the base folder.

    Args:
        base_folder (Path): Path to base folder which contains research_overview.md

    Returns:
        str: The research overview text

    """
    research_overview_text = ""
    research_overview_path = folder / "research_overview.md"
    if research_overview_path.exists():
        research_overview_text = research_overview_path.read_text()
        logger.info("Loaded research overview text from %s", research_overview_path)
    else:
        logger.warning(
            "research_overview.md not found at %s. Continuing without research overview text.",
            research_overview_path,
        )
    return research_overview_text


def load_table_description(folder: Path) -> str:
    """
    Load the figure descriptions from the base folder.

    Args:
        base_folder (Path): Path to base folder which contains table_summary.txt

    Returns:
        str: The table descriptions text

    """
    figure_descriptions_text = ""
    figure_descriptions_path = folder / "table_summary.txt"
    if figure_descriptions_path.exists():
        figure_descriptions_text = figure_descriptions_path.read_text()
    else:
        logger.warning(
            "table_summary.txt not found at %s. Continuing without table descriptions.",
            figure_descriptions_path,
        )
    return figure_descriptions_text


def load_figure_descriptions(folder: Path) -> str:
    """
    Load the figure descriptions from the base folder.

    Args:
        base_folder (Path): Path to base folder which contains figure_summary.txt

    Returns:
        str: The figure descriptions text

    """
    figure_descriptions_text = ""
    figure_descriptions_path = folder / "figure_summary.txt"
    if figure_descriptions_path.exists():
        figure_descriptions_text = figure_descriptions_path.read_text()
    else:
        logger.warning(
            "figure_summary.txt not found at %s. Continuing without figure descriptions.",
            figure_descriptions_path,
        )
    return figure_descriptions_text


def perform_writeup_with_agent(
    agents_md: Path,
    agent_config: AgentConfig,
    resources_folder: Path,
    citations_text: str,
    latex_folder: Path,
    pdf_folder: Path,
    research_overview_type: str = "long",
    num_page: int = 8,
    column_type: str = "single-column",
) -> bool:
    try:
        # Copy resources_folder to latex_folder, excluding unnecessary files
        exclude_patterns = {
            "reproduction_judgment_",
            "research_overview_long",
            "research_overview_short",
            "AGENTS_",
            "overview_sufficiency_evaluation",
            "eval_points",
        }

        def _ignore_files(_directory: str, filenames: list[str]) -> set[str]:
            ignored = set()
            for f in filenames:
                if any(f.startswith(pat) for pat in exclude_patterns):
                    ignored.add(f)
            return ignored

        shutil.copytree(resources_folder, latex_folder, dirs_exist_ok=True, ignore=_ignore_files)

        # Copy the prompt file with a filename appropriate for the agent type
        agent_type = agent_config.agent
        if agent_type in ("ClaudeCode", "ClaudeCode_Teams"):
            shutil.copy2(agents_md, latex_folder / "CLAUDE.md")
        else:
            shutil.copy2(agents_md, latex_folder / "AGENTS.md")

        # Copy research_overview_{type}.md as research_overview.md
        overview_src = resources_folder / f"research_overview_{research_overview_type}.md"
        if overview_src.exists():
            shutil.copy2(overview_src, latex_folder / "research_overview.md")
            logger.info("Copied %s as research_overview.md", overview_src.name)
        else:
            logger.warning(
                "research_overview_%s.md not found in %s", research_overview_type, resources_folder
            )

        shutil.copy2(resources_folder / "template.tex", latex_folder / "template.tex")

        research_overview_text = load_research_overview_text(latex_folder)

        (latex_folder / "references.bib").write_text(citations_text)

        # TODO
        table_description_str = load_table_description(latex_folder)
        plot_descriptions_str = load_figure_descriptions(latex_folder)
        combined_prompt = writeup_agent_prompt_template.format(
            research_overview_text=research_overview_text,
            plot_descriptions=plot_descriptions_str,
            table_descriptions=table_description_str,
            num_page=num_page,
            column_type=column_type,
        )

        logger.info("Step 3: Running Agent for full paper writing...")
        run_agent(
            agent=agent_config.agent,
            user_prompt=combined_prompt,
            working_dir=latex_folder,
            model=agent_config.model,
            max_turns=agent_config.max_turns,
            mode="READWRITE",
        )
        latex_file = latex_folder / "template.tex"
        if latex_file.exists():
            logger.info("template.tex successfully generated.")
        else:
            logger.info("Agent writeup generation failed")
            return False

        # Step 1: Compile after initial writeup
        writeup_pdf = pdf_folder / "writeup.pdf"
        compile_latex(cwd=latex_folder, latex_file=latex_file, pdf_file=writeup_pdf)

        # Step 2: Reflect until compiles
        reflection_pdf = pdf_folder / "reflection.pdf"
        reflect_until_compiles(
            latex_file=latex_file,
            latex_folder=latex_folder,
            pdf_file=reflection_pdf,
            agent_config=agent_config,
        )

        # Step 3: Adjust page limit
        page_adjusted_pdf = pdf_folder / "page_adjusted.pdf"
        adjust_page_limit(
            latex_file=latex_file,
            latex_folder=latex_folder,
            pdf_file=page_adjusted_pdf,
            agent_config=agent_config,
            page_limit=num_page,
        )

        # Final compile
        ensure_watermark(latex_file)
        final_pdf = pdf_folder / "final.pdf"
        compile_success, _ = compile_latex(
            cwd=latex_folder, latex_file=latex_file, pdf_file=final_pdf
        )
        if not compile_success:
            logger.warning("Final compile failed. Attempting to fix...")
            reflect_until_compiles(
                latex_file=latex_file,
                latex_folder=latex_folder,
                pdf_file=final_pdf,
                agent_config=agent_config,
            )

    except Exception:
        logger.exception("EXCEPTION in perform_writeup_with_agent")
        return False

    else:
        return True
