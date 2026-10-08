# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.
import math
import os
import re


def _last_sort_key(fname: str) -> float | None:
    """
    Sort key for last*.ckpt files where higher == newer.

    Lightning's versioning across sessions: each new session writes
    last-v{N+1}.ckpt where N is the current max on disk.  Higher N
    is always newer.  Bare last.ckpt is a legacy artifact from the
    first session and is treated as older than any versioned file.
    """
    match = re.match(r"^last-v(\d+)\.ckpt$", fname)
    if match:
        return int(match.group(1))
    if fname == "last.ckpt":
        return -math.inf
    return None


def fetch_last_ckpt(ckpt_dir: str) -> str | None:
    """
    Returns the filename of the most recent checkpoint in ckpt_dir.

    Search order (highest priority first):
    1. last-vN.ckpt files — pick the one with the largest N.
       Falls back to bare last.ckpt only if no versioned file exists.
    2. chk_epoch=*_step=*.ckpt files — only consulted if no last-family
       file is present.  Zero-padded filenames make lexicographic max()
       correct.

    Returns None if ckpt_dir does not exist or contains no matching files.
    """
    if not os.path.exists(ckpt_dir):
        return None

    all_files = os.listdir(ckpt_dir)

    last_ckpts = [f for f in all_files if _last_sort_key(f) is not None]
    if last_ckpts:
        return max(last_ckpts, key=_last_sort_key)

    named_ckpts = [
        f for f in all_files
        if re.match(r"^chk_epoch=\d+_step=\d+\.ckpt$", f)
    ]
    if named_ckpts:
        return max(named_ckpts)

    return None
