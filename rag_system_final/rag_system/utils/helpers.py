"""
RAG System Helper Functions
Utility functions for logging and display
"""

import logging
from typing import Optional


def setup_logging(name: str) -> logging.Logger:
    """Setup logging for a module"""
    logger = logging.getLogger(name)
    return logger


def print_header(text: str, char: str = "=", length: int = 60) -> None:
    """Print a formatted header"""
    print(f"\n{char * length}")
    print(f"{text.center(length)}")
    print(f"{char * length}\n")