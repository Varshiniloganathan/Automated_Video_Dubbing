"""
Step 1 (part A): Fetch the source video.

yt-dlp is used instead of pytube because it is far more resilient to
YouTube's frequent player/format changes and handles age-gated / throttled
streams better.
"""
from __future__ import annotations

from pathlib import Path

from .utils import check_binary, info, run, step


def download_video(url: str, out_dir: Path, max_height: int = 1080) -> Path:
    """
    Download an mp4 (video+audio) for `url` into `out_dir`, capped at
    `max_height` (default 1080p). Since the video track is stream-copied
    unchanged in the final remux step, downloading 4K/8K buys this pipeline
    nothing - it just costs disk space, download time, and bandwidth.
    Returns the path to the downloaded file.
    """
    step("Downloading source video with yt-dlp")
    check_binary("yt-dlp")

    out_template = str(out_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        # prefer a single progressive mp4 stream; fall back to best video+audio merged to mp4
        f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/best[ext=mp4][height<={max_height}]/best[height<={max_height}]",
        "--merge-output-format",
        "mp4",
        "-o",
        out_template,
        "--no-playlist",
        url,
    ]
    run(cmd, quiet=False)

    downloaded = sorted(out_dir.glob("source.*"))
    downloaded = [p for p in downloaded if p.suffix != ".part"]
    if not downloaded:
        raise FileNotFoundError("yt-dlp did not produce an output file.")

    video_path = downloaded[0]
    info(f"Saved: {video_path}")
    return video_path


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Pull a mono 16kHz WAV track out of the video for Whisper to consume."""
    step("Extracting audio track for transcription")
    check_binary("ffmpeg")

    audio_path = out_dir / "source_audio.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    run(cmd)
    info(f"Saved: {audio_path}")
    return audio_path
