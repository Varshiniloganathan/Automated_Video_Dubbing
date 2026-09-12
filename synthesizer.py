"""
Step 3: Turn the translated English text back into natural speech.

edge-tts wraps Microsoft Edge's free, high-quality neural voices - no API
key, no cost, and it sounds far less robotic than classic TTS engines.

Two things make the dub feel like it "matches" the original speaker:
  1. Voice gender matching - we sample pitch across several points in the
     source audio and take a majority vote to pick ONE male- or
     female-sounding English neural voice for the entire video. Per-segment
     detection was tried first but was too noisy - it could flip mid-video
     for a single continuous speaker.
  2. Time-alignment - each synthesized clip is time-stretched (ffmpeg
     `atempo`, which changes speed WITHOUT changing pitch) toward the
     original segment's start/end window, clamped to a range that still
     sounds natural (roughly 0.75x-1.35x). A clip that runs slightly long
     or short but sounds normal beats one crammed into an exact window
     that comes out distorted - exact sync is secondary to naturalness.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from pydub import AudioSegment

from .transcriber import Segment
from .utils import check_binary, ffprobe_duration, info, run, step

# A small pool of natural-sounding edge-tts voices to choose from.
MALE_VOICE = "en-US-GuyNeural"
FEMALE_VOICE = "en-US-JennyNeural"

# atempo only accepts 0.5-2.0 per filter instance; chain instances to go further.
_MIN_TEMPO, _MAX_TEMPO = 0.5, 2.0

# Beyond this range, atempo starts audibly distorting speech even though
# pitch is technically preserved. Better to let a clip run a bit long/short
# than to butcher how natural it sounds - exact sync is a secondary goal.
_NATURAL_TEMPO_MIN, _NATURAL_TEMPO_MAX = 0.75, 1.35


def _detect_gender_for_clip(y, sr) -> str | None:
    """Pitch-median heuristic on one audio clip. Returns None if no voiced pitch found."""
    import librosa
    import numpy as np

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced = f0[voiced_flag] if voiced_flag is not None else f0
    voiced = voiced[~np.isnan(voiced)] if voiced is not None else []
    if len(voiced) == 0:
        return None
    median_f0 = float(np.median(voiced))
    return "male" if median_f0 < 165 else "female"


def detect_dominant_voice(source_audio: Path, segments: list[Segment]) -> str:
    """
    Decide ONE voice gender for the whole video by sampling pitch across
    several segments and taking a majority vote, instead of re-deciding
    per segment.
    """
    import librosa

    try:
        sample_indices = range(0, len(segments), max(1, len(segments) // 15))
        votes = {"male": 0, "female": 0}
        for i in sample_indices:
            seg = segments[i]
            y, sr = librosa.load(
                str(source_audio), sr=16000, offset=seg.start, duration=max(seg.end - seg.start, 0.2)
            )
            vote = _detect_gender_for_clip(y, sr)
            if vote:
                votes[vote] += 1

        if votes["male"] == 0 and votes["female"] == 0:
            info("  Could not confidently detect voice pitch; defaulting to female voice")
            return "female"

        winner = "male" if votes["male"] >= votes["female"] else "female"
        info(f"  Voice sample votes: {votes} -> using '{winner}' voice for the whole video")
        return winner
    except Exception as e:
        info(f"  Voice detection failed ({type(e).__name__}: {e or 'no message'}); defaulting to female voice")
        return "female"


async def _edge_tts_save(text: str, voice: str, out_path: Path, max_attempts: int = 4) -> None:
    import edge_tts

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(out_path))
            return
        except Exception as e:  # covers aiohttp timeouts/connection resets, etc.
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt  # 2s, 4s, 8s
                info(f"    edge-tts call failed ({e!r}), retrying in {wait}s "
                     f"[{attempt}/{max_attempts}]")
                await asyncio.sleep(wait)
    raise RuntimeError(f"edge-tts failed after {max_attempts} attempts: {last_error}") from last_error


def _synthesize_clip(text: str, voice: str, out_path: Path) -> None:
    if not text.strip():
        # Write a near-silent placeholder so downstream duration math is simple.
        AudioSegment.silent(duration=200).export(out_path, format="mp3")
        return
    try:
        asyncio.run(_edge_tts_save(text, voice, out_path))
    except Exception as e:
        info(f"    Giving up on this segment after retries ({e!r}); using silence instead")
        AudioSegment.silent(duration=200).export(out_path, format="mp3")


def _time_stretch(in_path: Path, out_path: Path, tempo: float) -> None:
    """Chain ffmpeg atempo filters to hit any tempo factor, in-range or not."""
    tempo = max(0.35, min(tempo, 3.0))  # sane overall clamp
    factors = []
    remaining = tempo
    while remaining > _MAX_TEMPO:
        factors.append(_MAX_TEMPO)
        remaining /= _MAX_TEMPO
    while remaining < _MIN_TEMPO:
        factors.append(_MIN_TEMPO)
        remaining /= _MIN_TEMPO
    factors.append(remaining)
    filter_chain = ",".join(f"atempo={f:.4f}" for f in factors)

    run(["ffmpeg", "-y", "-i", str(in_path), "-filter:a", filter_chain, str(out_path)])


def synthesize_dubbed_track(
    source_audio: Path,
    segments: list[Segment],
    total_duration: float,
    work_dir: Path,
) -> Path:
    step("Synthesizing English speech with edge-tts (natural neural voices)")
    check_binary("ffmpeg")
    clips_dir = work_dir / "tts_clips"
    clips_dir.mkdir(exist_ok=True)

    timeline = AudioSegment.silent(duration=int(total_duration * 1000))

    gender = detect_dominant_voice(source_audio, segments)
    voice = MALE_VOICE if gender == "male" else FEMALE_VOICE

    for i, seg in enumerate(segments):
        target_ms = max(int((seg.end - seg.start) * 1000), 250)

        raw_path = clips_dir / f"seg_{i:04d}_raw.mp3"
        stretched_path = clips_dir / f"seg_{i:04d}_fit.mp3"

        _synthesize_clip(seg.text, voice, raw_path)
        raw_duration = ffprobe_duration(raw_path) or 0.001

        needed_tempo = raw_duration / (target_ms / 1000)
        applied_tempo = max(_NATURAL_TEMPO_MIN, min(needed_tempo, _NATURAL_TEMPO_MAX))
        if abs(applied_tempo - 1.0) > 0.03:
            _time_stretch(raw_path, stretched_path, applied_tempo)
        else:
            stretched_path = raw_path

        clip = AudioSegment.from_file(stretched_path)

        next_start_ms = int(segments[i + 1].start * 1000) if i + 1 < len(segments) else int(total_duration * 1000)
        max_allowed_ms = max(target_ms, int(next_start_ms - seg.start * 1000) - 50)  # 50ms safety gap
        if len(clip) > max_allowed_ms:
            clip = clip[:max_allowed_ms]

        timeline = timeline.overlay(clip, position=int(seg.start * 1000))
        info(f"[{i + 1}/{len(segments)}] {seg.start:6.1f}s-{seg.end:6.1f}s ({gender}): {seg.text[:60]}")

    out_path = work_dir / "dubbed_audio.wav"
    timeline.export(out_path, format="wav")
    info(f"Saved: {out_path}")
    return out_path
