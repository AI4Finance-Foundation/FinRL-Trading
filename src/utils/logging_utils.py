"""Shared logging helpers for the FinRL trading platform.

This module is referenced by ``src/main.py`` and ``src/web/app.py`` but was
missing from the repository distribution. It provides a small,
dependency-free ``setup_logging`` helper compatible with those call sites.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

Level = Union[int, str]


def setup_logging(
    level: Level = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    console: bool = True,
) -> logging.Logger:
    """Configure root logging with a console handler and optional file handler.

    Args:
        level: Logging level (int or name such as ``"INFO"``).
        log_file: Optional path to a log file; parent dirs are created.
        log_format: Format string for log records.
        console: Whether to attach a stream handler to stderr.

    Returns:
        The root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove handlers installed by earlier calls so re configuration is idempotent.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(log_format)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


__all__ = ["setup_logging", "DEFAULT_FORMAT"]
