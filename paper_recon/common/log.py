import logging
import shutil
from pathlib import Path

from rich.logging import RichHandler

_LOGGER_MAP: dict[str, tuple[logging.Logger, bool]] = {}  # name -> (logger, enable_stdout)
_LOG_DIR: Path | None = None


def get_logger(name: str, enable_stdout: bool = True) -> logging.Logger:
    """
    Get logger by name.

    Args:
        name (str): logger name.
        enable_stdout (bool, optional): whether to enable stdout logging. Defaults to True.

    Returns:
        Logger: logger instance

    """
    name = name.removesuffix(".py")
    name = name.replace("/", "-").replace("\\", "_").replace(".", "_")
    if name in _LOGGER_MAP:
        return _LOGGER_MAP[name][0]
    logger = _build_logger(name, enable_stdout=enable_stdout)
    _LOGGER_MAP[name] = (logger, enable_stdout)
    return logger


def _build_logger(name: str, enable_stdout: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if _LOG_DIR is not None:
        file_handler = _file_handler(name)
        logger.addHandler(file_handler)
    if enable_stdout:
        stream_handler = _stream_handler()
        logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def _file_handler(name: str) -> logging.FileHandler:
    if _LOG_DIR is None:
        raise RuntimeError("Log dir is not set.")
    file_handler = logging.FileHandler(_LOG_DIR / f"{name}.log")
    file_handler.setFormatter(
        WrappingFormatter("%(asctime)s %(levelname)s %(message)s", datefmt="[%X]")
    )
    return file_handler


def _stream_handler() -> RichHandler:
    stream_handler = RichHandler()
    stream_handler.setFormatter(
        WrappingFormatter("%(message)s", datefmt="[%X]", max_lines=10, console=True)
    )
    return stream_handler


class WrappingFormatter(logging.Formatter):
    """Output raw text for file handler while truncating long messages for console handler."""

    max_lines: int | None
    console: bool

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        max_lines: int | None = None,
        console: bool = False,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.max_lines = max_lines
        self.console = console

    def truncate_message_for_console(self, message: str) -> str:
        if self.max_lines:
            width = shutil.get_terminal_size((80, 20)).columns - 20
            if len(message) > width * self.max_lines:
                return message[: width * self.max_lines] + " ...(truncated)"
            return message
        return message

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if self.console:
            return self.truncate_message_for_console(message)
        return message


def set_log_dir(log_dir: Path) -> None:
    """
    Set directory path of file handler for all logger instances.

    Only call this once during the lifetime of the program.
    The first call sets the log directory for all loggers.

    Args:
        log_dir (Path): directory path for log files.

    """
    global _LOG_DIR  # noqa: PLW0603

    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIR = log_dir

    for name, (logger, enable_stdout) in _LOGGER_MAP.items():
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

        logger.addHandler(_file_handler(name))
        if enable_stdout:
            logger.addHandler(_stream_handler())
