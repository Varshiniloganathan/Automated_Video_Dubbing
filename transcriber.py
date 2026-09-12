"""
Step 1 (part B): Transcribe the source audio into timestamped text.

Whisper is used because it is free, open-source, works across dozens of
languages out of the box, and returns word/segment-level timestamps, which
we need later to time-align the dubbed audio to the original video.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import info, step


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(audio_path: Path, model_size: str = "medium") -> tuple[str, list[Segment]]:
    """
    Returns (detected_language_code, list_of_segments).

    model_size options (speed vs. accuracy trade-off):
      tiny / base / small / medium / large-v3
    "medium" is a good default: much more accurate than "small" on
    non-English speech, while still running on a single GPU or a patient CPU.
    """
    step("Transcribing speech with Whisper (this can take a while on CPU)")
    import whisper  # imported lazily so --help etc. don't require the heavy dep

    model = whisper.load_model(model_size)
    result = model.transcribe(
        str(audio_path),
        verbose=False,
        # Tighter hallucination guards than Whisper's defaults - important for
        # stretches of music, silence, or noise, which otherwise sometimes get
        # transcribed as confident-looking gibberish in a random language.
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    language = result.get("language", "unknown")
    info(f"Detected source language: {language}")

    segments = []
    dropped = 0
    for s in result["segments"]:
        text = s["text"].strip()
        if not text:
            continue
        # Only drop a segment if MULTIPLE signals agree it's a hallucination,
        # not just one - a single low signal (e.g. no_speech_prob alone) can
        # fire on genuine speech, especially code-switched/accented audio
        # where Whisper's confidence is naturally a bit lower even when the
        # transcription is correct. This mirrors Whisper's own internal
        # "should I retry this segment" heuristic rather than being stricter.
        no_speech_prob = s.get("no_speech_prob", 0.0)
        avg_logprob = s.get("avg_logprob", 0.0)
        compression_ratio = s.get("compression_ratio", 0.0)

        looks_silent_and_uncertain = no_speech_prob > 0.6 and avg_logprob < -1.0
        looks_repetitive_gibberish = compression_ratio > 2.6

        if looks_silent_and_uncertain or looks_repetitive_gibberish:
            dropped += 1
            continue
        segments.append(Segment(start=s["start"], end=s["end"], text=text))

    if dropped:
        info(f"Dropped {dropped} low-confidence segment(s) that looked like hallucinations")

    segments = _merge_short_segments(segments)
    info(f"Got {len(segments)} speech segments")
    return language, segments


def _merge_short_segments(segments: list[Segment], min_duration: float = 1.5, max_gap: float = 0.4) -> list[Segment]:
    """
    Merge a short segment into its neighbor when they're close together in
    time. Heavily over-segmented transcripts (many <1s fragments) force the
    synthesizer to time-stretch tiny TTS clips extremely hard to fit their
    tiny windows, which is the main source of unnatural-sounding dubbing -
    giving edge-tts fuller phrases to work with sounds much more natural.
    """
    if not segments:
        return segments

    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_duration = prev.end - prev.start
        gap = seg.start - prev.end
        if prev_duration < min_duration and gap <= max_gap:
            merged[-1] = Segment(start=prev.start, end=seg.end, text=f"{prev.text} {seg.text}".strip())
        else:
            merged.append(seg)
    return merged
