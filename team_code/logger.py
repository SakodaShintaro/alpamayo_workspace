import logging
import os
from pathlib import Path

_LOGGER: logging.Logger | None = None
_FORMATTER = logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S")


def _ensure_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("alpamayo15_agent")
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = False
    return _LOGGER


def get_logger() -> logging.Logger:
    logger = _ensure_logger()
    if not logger.handlers:
        configure_logger(Path(os.environ.get("SAVE_PATH", ".")))
    return logger


def configure_logger(log_dir: Path) -> None:
    logger = _ensure_logger()
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "agent.log", mode="w")
    handler.setFormatter(_FORMATTER)
    logger.addHandler(handler)
