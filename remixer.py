"""
Step 4: Swap in the dubbed audio and produce the final deliverable.

`-c:v copy` tells ffmpeg to stream-copy the original video track instead
of re-encoding it, so this step is fast and there is zero quality loss
on the picture.
"""
from __future__ import annotations

from pathlib import Path

from .utils import check_binary, info, run, step


def remix(video_path: Path, dubbed_audio_path: Path, out_path: Path) -> Path:
    step("Muxing dubbed audio into the original video (no video re-encode)")
    check_binary("ffmpeg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(dubbed_audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    run(cmd)
    info(f"Saved final dubbed video: {out_path}")
    return out_path
