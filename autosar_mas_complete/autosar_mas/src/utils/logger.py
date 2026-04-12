"""
logger.py — Centralized logging configuration for AUTOSAR MAS project.

Provides a pre-configured logger instance used across all modules.
Follows MNC standard: structured format with timestamps and log levels.
"""

import logging
import sys

# ---------------------------------------------------------------------------
# Logging Format: timestamp | level | module | message
# ---------------------------------------------------------------------------
_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Module-level logger — imported by all other modules
logger = logging.getLogger("autosar_mas")
