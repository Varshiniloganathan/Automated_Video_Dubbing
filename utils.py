"""Small shared helpers used across the pipeline."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path


def step(msg: str) -> None:
    """Print a progress banner so the user can see what's happening live."""
    print(f"\n\033[1;36m==> {msg}\033[0m", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\033[1;31mERROR: {msg}\033[0m", file=sys.stderr, flush=True)
    sys.exit(1)


def check_binary(name: str) -> None:
    """Fail fast with a clear message if a required CLI tool is missing."""
    if shutil.which(name) is None:
        die(
            f"Required command-line tool '{name}' was not found on PATH. "
            f"Install it before running this script."
        )


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command, raising with useful output on failure."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        text=True,
    )
    if result.returncode != 0:
        die(f"Command failed ({' '.join(cmd)}):\n{result.stdout}")
    return result


def ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    check_binary("ffprobe")
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


class Timer:
    """Tiny context manager to print how long a stage took."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        info(f"({self.label} took {time.time() - self._t0:.1f}s)")
