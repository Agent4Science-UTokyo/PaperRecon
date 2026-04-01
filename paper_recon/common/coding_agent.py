import asyncio
import json
import os
import subprocess as sp
import uuid
from pathlib import Path
from typing import Literal

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from claude_agent_sdk import query as query_claude_code
from claude_agent_sdk._errors import MessageParseError
from pydantic import BaseModel

from paper_recon.common.log import get_logger

logger = get_logger(__file__)


def run_agent(
    agent: Literal["ClaudeCode", "ClaudeCode_Teams", "Codex"],
    user_prompt: str,
    working_dir: Path,
    model: str,
    max_turns: int,
    mode: Literal["READONLY", "READWRITE"],
    response_format: type[BaseModel] | None = None,
) -> str:
    """
    Run Codex to generate the write-up in LaTeX format.

    Args:
        agent (Literal["ClaudeCode", "Codex"]): The agent to use for generating the write-up.
        user_prompt (str): The prompt to provide to the agent.
        working_dir (Path): Path to the working directory.
        model (str): The agent model to use.
        max_turns (int): Maximum number of turns for the agent (only applicable for Claude Code, ignored for Codex).
        mode (Literal["READONLY", "READWRITE"]): Mode for the agent, determining the level of access to the file system.
        response_format (type[BaseModel] | None): Optional Pydantic model for response validation.

    Returns:
        str: Result message from Codex.

    """
    if agent == "ClaudeCode":
        return run_claudecode(
            user_prompt=user_prompt,
            working_dir=working_dir,
            model=model,
            max_turns=max_turns,
            mode=mode,
            response_format=response_format,
        )
    if agent == "ClaudeCode_Teams":
        return run_claudecode_agent_teams(
            user_prompt=user_prompt,
            working_dir=working_dir,
            model=model,
            max_turns=max_turns,
            mode=mode,
            response_format=response_format,
        )
    if agent == "Codex":
        return run_codex(
            user_prompt=user_prompt,
            working_dir=working_dir,
            model=model,
            max_turns=max_turns,
            mode=mode,
            response_format=response_format,
        )
    raise ValueError(f"Unsupported agent: {agent}")


def run_claudecode(
    user_prompt: str,
    working_dir: Path,
    model: str,
    max_turns: int,
    mode: Literal["READONLY", "READWRITE"],
    response_format: type[BaseModel] | None = None,
) -> str:
    """
    Run Claude Code to generate the write-up in LaTeX format.

    Args:
        user_prompt (str): The prompt to provide to Claude Code.
        working_dir (Path): Path to the working directory.
        model (str): The Claude Code model to use.
        max_turns (int): Maximum number of turns for Claude Code.
        mode (Literal["READONLY", "READWRITE"]): Mode for Claude Code.
        response_format (type[BaseModel] | None): Optional Pydantic model for response validation.

    Returns:
        str: Result message from Claude Code.

    """
    # See: https://code.claude.com/docs/en/settings#tools-available-to-claude
    allowed_tools = ["Read", "Glob", "Grep"]
    permission_mode = "default"
    if mode == "READWRITE":
        allowed_tools.extend(["Write", "Edit"])
        permission_mode = "acceptEdits"

    options = ClaudeAgentOptions(
        max_turns=max_turns,
        model=model,
        cwd=working_dir,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
    )
    if response_format:
        options.output_format = {
            "type": "json_schema",
            "schema": response_format.model_json_schema(),
        }

    async def _arun_claudecode() -> None | ResultMessage:
        cc_logger = get_logger("claudecode", enable_stdout=True)
        result_message = None
        try:
            async for message in query_claude_code(prompt=user_prompt, options=options):
                cc_logger.info(message)
                if isinstance(message, ResultMessage):
                    result_message = message
        except MessageParseError:
            logger.exception("Claude Code session interrupted by MessageParseError")
            raise
        return result_message

    logger.info("[Claude Code] Running...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_message = loop.run_until_complete(_arun_claudecode())
    finally:
        loop.close()

    if result_message:
        if response_format:
            return json.dumps(result_message.structured_output)
        return result_message.result or ""
    return ""


def run_claudecode_agent_teams(
    user_prompt: str,
    working_dir: Path,
    model: str,
    max_turns: int,
    mode: Literal["READONLY", "READWRITE"],
    response_format: type[BaseModel] | None = None,
) -> str:
    """
    Run Claude Code with Agent Teams enabled.

    Args:
        user_prompt (str): The prompt to provide to Claude Code.
        working_dir (Path): Path to the working directory.
        model (str): The Claude Code model to use.
        max_turns (int): Maximum number of turns for Claude Code.
        mode (Literal["READONLY", "READWRITE"]): Mode for Claude Code.
        response_format (type[BaseModel] | None): Optional Pydantic model for response validation.

    Returns:
        str: Result message from Claude Code.

    """
    team_prompt = (
        "You are the lead agent. You MUST use the TeamCreate tool to create "
        "a team of specialized agents before starting any work. This is "
        "mandatory — do NOT attempt to do everything alone.\n\n"
        "Steps:\n"
        "1. Analyze the task below.\n"
        "2. Use TeamCreate to create 2-4 specialized teammates with clear "
        "roles (e.g., one for reading/analyzing data, one for writing "
        "sections, one for formatting tables/figures, etc.).\n"
        "3. Coordinate the team using SendMessage to delegate subtasks.\n"
        "4. As team lead, review and integrate the results.\n\n"
        "Do NOT write a Python script or SDK code to orchestrate this. "
        "Use the built-in TeamCreate and SendMessage tools directly.\n\n"
        "IMPORTANT:\n"
        "- Do NOT compile LaTeX (pdflatex/latexmk/etc.). "
        "Compilation will be handled automatically after you finish.\n"
        "Focus only on writing the content in template.tex.\n\n"
        f"Task: {user_prompt}"
    )

    allowed_tools = ["Read", "Glob", "Grep"]
    permission_mode = "default"
    if mode == "READWRITE":
        allowed_tools.extend(["Write", "Edit"])
        permission_mode = "acceptEdits"

    options = ClaudeAgentOptions(
        max_turns=max_turns,
        model=model,
        cwd=working_dir,
        allowed_tools=allowed_tools,
        permission_mode=permission_mode,
        env={"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"},
    )
    if response_format:
        options.output_format = {
            "type": "json_schema",
            "schema": response_format.model_json_schema(),
        }

    async def _arun() -> None | ResultMessage:
        cc_logger = get_logger("claudecode_teams", enable_stdout=False)
        result_message = None
        agent_done = asyncio.Event()
        template_path = working_dir / "template.tex"

        async def _run_agent():
            nonlocal result_message
            try:
                async for message in query_claude_code(prompt=team_prompt, options=options):
                    cc_logger.info(message)
                    if isinstance(message, ResultMessage):
                        result_message = message
            except asyncio.CancelledError:
                logger.info("[Claude Code Agent Teams] Agent cancelled")
            except MessageParseError as e:
                logger.exception("Claude Code Teams session interrupted: %s", e)
                raise
            finally:
                agent_done.set()

        async def _monitor():
            """Stop the agent if template.tex has been sufficiently written and no updates occur for 3 minutes."""
            poll_interval = 10
            stale_timeout = 180  # 3 minutes
            last_mtime = None
            last_change_time = None
            min_content_bytes = 15000

            # Required sections (Abstract is handled separately via \begin{abstract})
            required_markers = [
                "\\begin{abstract}",
                "\\section",  # At least one section must exist
            ]

            def _is_sufficiently_written() -> bool:
                if not template_path.exists():
                    return False
                content = template_path.read_text(encoding="utf-8")
                if len(content) < min_content_bytes:
                    return False
                for marker in required_markers:
                    if marker not in content:
                        return False
                # Check whether Abstract has actual content
                if "\\begin{abstract}" in content and "\\end{abstract}" in content:
                    abstract = (
                        content.split("\\begin{abstract}")[1].split("\\end{abstract}")[0].strip()
                    )
                    if len(abstract) < 100:
                        return False
                else:
                    return False
                return True

            # Condition 1: Record the mtime just before startup
            initial_mtime = template_path.stat().st_mtime if template_path.exists() else None
            has_been_modified = False

            while not agent_done.is_set():
                await asyncio.sleep(poll_interval)
                if not template_path.exists():
                    continue
                mtime = template_path.stat().st_mtime

                # Condition 1: Check if file has been modified since startup
                if not has_been_modified:
                    if initial_mtime is None or mtime != initial_mtime:
                        has_been_modified = True
                        last_mtime = mtime
                        last_change_time = asyncio.get_event_loop().time()
                        logger.info("[Claude Code Agent Teams] First modification detected")
                    continue

                # Track updates
                if mtime != last_mtime:
                    last_mtime = mtime
                    last_change_time = asyncio.get_event_loop().time()
                elif last_change_time is not None:
                    elapsed = asyncio.get_event_loop().time() - last_change_time
                    # Condition 2: 3 minutes since last update + Condition 3: sufficient content size
                    if elapsed >= stale_timeout and _is_sufficiently_written():
                        logger.info(
                            "[Claude Code Agent Teams] template.tex unchanged "
                            "for %.0fs and sufficiently written (%d bytes), "
                            "stopping agent...",
                            elapsed,
                            template_path.stat().st_size,
                        )
                        return True
            return False

        if mode == "READWRITE" and not response_format:
            agent_task = asyncio.create_task(_run_agent())
            monitor_task = asyncio.create_task(_monitor())
            done, _ = await asyncio.wait(
                [agent_task, monitor_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor_task in done and monitor_task.result():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            else:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
        else:
            await _run_agent()

        return result_message

    logger.info("[Claude Code Agent Teams] Running...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_message = loop.run_until_complete(_arun())
    finally:
        loop.close()

    if result_message:
        if response_format:
            return json.dumps(result_message.structured_output)
        return result_message.result or ""
    return ""


def run_codex(
    user_prompt: str,
    working_dir: Path,
    model: str,
    max_turns: int,
    mode: Literal["READONLY", "READWRITE"],
    response_format: type[BaseModel] | None = None,
) -> str:
    """
    Run Codex to generate the write-up in LaTeX format.

    Args:
        user_prompt (str): The prompt to provide to Codex.
        working_dir (Path): Path to the working directory.
        model (str): The Codex model to use.
        max_turns (int): NO EFFECT for Codex, included for interface consistency.
        mode (Literal["READONLY", "READWRITE"]): Mode for Codex.
        response_format (type[BaseModel] | None): Optional Pydantic model for response validation.

    Returns:
        str: Result message from Codex.

    """
    configs = []
    # Always use full-access because bwrap sandbox does not work inside Docker
    configs.extend(["-c", "sandbox_mode=danger-full-access"])
    configs.extend(["-c", "web_search=disabled"])

    if model.startswith("azure/"):
        model = model[len("azure/") :]
        configs.extend(["-c", "model_provider=azure"])
        configs.extend(["-c", "model_providers.azure.name=AzureOpenAI"])
        # GPT-5.4 uses a separate endpoint and API key
        if "gpt-5.4" in model or "gpt-54" in model:
            base_url = os.getenv("AZURE_GPT54_API_BASE", "") + "openai/v1"
            env_key = "AZURE_GPT54_API_KEY"
        else:
            base_url = os.getenv("AZURE_API_BASE", "") + "openai/v1"
            env_key = "AZURE_API_KEY"
        configs.extend(["-c", f"model_providers.azure.base_url={base_url}"])
        configs.extend(["-c", f"model_providers.azure.env_key={env_key}"])
        configs.extend(["-c", "model_providers.azure.wire_api=responses"])

    logger.info("[Codex] Running...")
    codex_logger = get_logger("codex", enable_stdout=False)
    if response_format is not None:
        uid = uuid.uuid4().hex[:8]
        schema_filename = f"response_format_{uid}.json"
        output_filename = f"response_{uid}.json"
        response_format_file = working_dir / schema_filename
        schema = response_format.model_json_schema()

        # OpenAI structured output requires additionalProperties: false on all objects
        def _add_additional_properties(obj: dict) -> dict:
            if isinstance(obj, dict):
                if obj.get("type") == "object" or "properties" in obj:
                    obj["additionalProperties"] = False
                for v in obj.values():
                    if isinstance(v, dict):
                        _add_additional_properties(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                _add_additional_properties(item)
            return obj

        schema = _add_additional_properties(schema)
        schema["additionalProperties"] = False
        response_format_file.write_text(json.dumps(schema, indent=2))
        p = sp.run(
            [
                "codex",
                "exec",
                "--model",
                model,
                *configs,
                "--output-schema",
                f"./{schema_filename}",
                "-o",
                f"./{output_filename}",
                user_prompt,
            ],
            cwd=working_dir,
            capture_output=True,
        )
        if p.returncode != 0:
            codex_logger.error("Codex stderr: %s", p.stderr.decode())
            codex_logger.error("Codex stdout: %s", p.stdout.decode())
            p.check_returncode()
        response_file = working_dir / output_filename
        response = response_file.read_text()
        response_file.unlink(missing_ok=True)
        response_format_file.unlink(missing_ok=True)
        codex_logger.info(p.stdout.decode())
        codex_logger.info(p.stderr.decode())
        return response
    else:  # noqa: RET505
        p = sp.run(
            [
                "codex",
                "exec",
                "--model",
                model,
                *configs,
                "-o",
                "./response.json",
                user_prompt,
            ],
            cwd=working_dir,
            capture_output=True,
        )
        codex_logger.info(p.stdout.decode())
        codex_logger.info(p.stderr.decode())
        if p.returncode != 0:
            logger.error("Codex failed with return code %d", p.returncode)
            logger.error("Codex stdout: %s", p.stdout.decode())
            logger.error("Codex stderr: %s", p.stderr.decode())
            raise sp.CalledProcessError(p.returncode, p.args)
        response_file = working_dir / "response.json"
        response = response_file.read_text()
        response_file.unlink()
        return response
