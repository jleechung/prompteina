# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.

"""
Validation utilities for protein generation metrics and file management.
Extracted from Proteina class for separation of concerns.
"""

import os
import re
import shutil
import tempfile

import numpy as np



def clean_validation_files(paths: list[str], limit: int = 2) -> None:
    """
    Remove most generated validation files, keeping a subset for visualization.

    Keeps files where num < limit and rank <= 2.

    Args:
        paths: List of file paths to clean up
        limit: Keep files with num < limit (from filename pattern _num_N_)
    """
    from loguru import logger

    pattern = re.compile(r".*_num_(\d+)_rank_(\d+)")
    for fname in paths:
        match = pattern.match(fname)
        if match:
            i = int(match.group(1))
            rank = int(match.group(2))
            if i >= limit or rank > 2:
                try:
                    os.remove(fname)
                except Exception:
                    logger.info("Could not remove validation file")
