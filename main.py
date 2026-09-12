#!/usr/bin/env python3
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from .downloader import download_video, extract_audio
from .remixer import remix
from .synthesizer import synthesize_dubbed_track
from .transcriber import transcribe
from .translator import translate_segments
from .utils import die, ffprobe_duration, step, Timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output", default=None, help="Path for the final dubbed .mp4 (default: <video_id>_dubbed.mp4)"
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (bigger = more accurate, slower). Default: medium",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Cap downloaded video resolution in pixels of height (default: 1080). "
             "The video track is copied unchanged in the final output, so higher "
             "values just cost more disk/bandwidth for no quality benefit here.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for intermediate files (default: ~/.cache/video_dubber/tmp). "
             "IMPORTANT: on many Linux systems /tmp is RAM-backed (tmpfs) and much "
             "smaller than your actual disk - avoid pointing this at /tmp for anything "
             "but short clips.",
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep intermediate files (audio, tts clips) for debugging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = args.url or input("Enter a YouTube URL: ").strip()
    if not url:
        die("No URL provided.")

    base_dir = Path(args.work_dir) if args.work_dir else Path.home() / ".cache" / "video_dubber" / "tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="video_dub_", dir=str(base_dir)))
    step(f"Working directory: {work_dir}")

    try:
        # 1) Fetch & Transcribe --------------------------------------------------
        with Timer("download"):
            video_path = download_video(url, work_dir, max_height=args.max_height)
            source_audio = extract_audio(video_path, work_dir)

        total_duration = ffprobe_duration(video_path)

        with Timer("transcription"):
            language, segments = transcribe(source_audio, model_size=args.whisper_model)

        if not segments:
            die("No speech detected in the video - nothing to dub.")

        # 2) Translate -------------------------------------------------------------
        with Timer("translation"):
            english_segments = translate_segments(language, segments)

        # 3) Synthesize --------------------------------------------------------------
        with Timer("speech synthesis"):
            dubbed_audio = synthesize_dubbed_track(
                source_audio, english_segments, total_duration, work_dir
            )

        # 4) Remix & Output ------------------------------------------------------------
        out_path = Path(args.output) if args.output else Path.cwd() / f"{video_path.stem}_dubbed.mp4"
        with Timer("remux"):
            remix(video_path, dubbed_audio, out_path)

        step(f"Done! English-dubbed video saved to: {out_path}")

    finally:
        if args.keep_temp:
            step(f"Intermediate files kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
