import logging
import os
from pathlib import Path

_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    save_path = Path(os.environ.get("SAVE_PATH", "."))
    save_path.mkdir(parents=True, exist_ok=True)
    log_file = save_path / "agent.log"
    logger = logging.getLogger("alpamayo15_agent")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    _LOGGER = logger
    return logger
