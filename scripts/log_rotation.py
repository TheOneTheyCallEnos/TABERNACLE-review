#!/usr/bin/env python3
"""
LOG ROTATION — Prevents log files from consuming all disk space.
Called nightly by gardener.

Strategy:
  - Rotate .log files > 10MB: compress to .1.gz, keep 2 generations
  - Truncate .err files > 50MB (typically benign warnings/noise)
"""

import gzip
import shutil
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("/Users/enos/TABERNACLE/logs")
MAX_LOG_SIZE = 10 * 1024 * 1024    # 10MB
MAX_ERR_SIZE = 50 * 1024 * 1024    # 50MB
EXEMPT_SUFFIXES = {".gz", ".bak"}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [LOG-ROTATION] {msg}")


def rotate_file(path: Path):
    """Rotate a single log file: .log -> .1.gz, delete old .2.gz"""
    gen1 = path.parent / (path.name + ".1.gz")
    gen2 = path.parent / (path.name + ".2.gz")

    if gen2.exists():
        gen2.unlink()
    if gen1.exists():
        gen1.rename(gen2)

    size = path.stat().st_size
    try:
        with open(path, 'rb') as f_in:
            with gzip.open(gen1, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        with open(path, 'w') as f:
            f.write("")
        log(f"Rotated {path.name} ({size // 1024}KB → {gen1.name})")
    except Exception as e:
        log(f"Error rotating {path.name}: {e}")


def truncate_err(path: Path):
    """Truncate oversized .err files."""
    size = path.stat().st_size
    try:
        with open(path, 'w') as f:
            f.write("")
        log(f"Truncated {path.name} ({size // (1024*1024)}MB)")
    except Exception as e:
        log(f"Error truncating {path.name}: {e}")


def run_rotation():
    """Main rotation pass. Returns dict with counts."""
    log("Starting log rotation...")
    rotated = 0
    truncated = 0

    for f in sorted(LOG_DIR.iterdir()):
        if not f.is_file():
            continue
        if f.suffix in EXEMPT_SUFFIXES:
            continue

        try:
            size = f.stat().st_size
        except OSError:
            continue

        if f.suffix == '.err' and size > MAX_ERR_SIZE:
            truncate_err(f)
            truncated += 1
        elif f.suffix == '.log' and size > MAX_LOG_SIZE:
            rotate_file(f)
            rotated += 1

    log(f"Complete: {rotated} rotated, {truncated} truncated")
    return {"rotated": rotated, "truncated": truncated}


if __name__ == "__main__":
    run_rotation()
