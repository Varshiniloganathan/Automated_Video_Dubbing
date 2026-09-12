"""
Step 2: Translate for meaning, not word-for-word.

Strategy: run Meta's NLLB-200 (facebook/nllb-200-distilled-600M) LOCALLY via
plain `transformers` for every source language it supports. This replaces
the remote Hugging Face Inference API approach, which turned out to not
host this model on their free tier at all ("Model not supported by
provider hf-inference") - a dead end, not a transient issue.

Deliberately NO fallback to another translator: if a segment can't be
translated (e.g. it's genuinely empty), it's kept in its original language
rather than substituting a different translation engine's output.

First run downloads the model (~2.5GB) from Hugging Face and caches it
locally - no token/account needed for this, since it's not going through
the paid Inference Providers routing that broke the remote path.
"""
from __future__ import annotations

from .transcriber import Segment
from .utils import info, step

# Whisper language codes -> FLORES-200 codes that NLLB-200 expects.
NLLB_LANG_MAP = {
    "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "ne": "npi_Deva",
    "or": "ory_Orya", "pa": "pan_Guru", "sa": "san_Deva", "sd": "snd_Arab",
    "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab",
}

_NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
_nllb_cache: dict = {}  # lazy-loaded model/tokenizer, kept warm across calls


def translate_segments(language: str, segments: list[Segment]) -> list[Segment]:
    step("Translating transcript into English")

    if language not in NLLB_LANG_MAP:
        info(
            f"'{language}' has no FLORES-200 mapping for NLLB-200 in this project; "
            f"keeping original text for every segment (no fallback translator configured)."
        )
        return segments

    info(f"Using NLLB-200 (local) for '{language}'")
    texts = _translate_with_nllb_local(language, [s.text for s in segments])
    return [Segment(s.start, s.end, t) for s, t in zip(segments, texts)]


def _translate_with_nllb_local(language: str, texts: list[str]) -> list[str]:
    """
    Requires: pip install torch transformers sentencepiece
    Model card: https://huggingface.co/facebook/nllb-200-distilled-600M
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    src_lang = NLLB_LANG_MAP[language]
    tgt_lang = "eng_Latn"

    if "model" not in _nllb_cache:
        info("  Loading NLLB-200 (first run downloads ~2.5GB, then it's cached)")
        tokenizer = AutoTokenizer.from_pretrained(_NLLB_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _nllb_cache["tokenizer"] = tokenizer
        _nllb_cache["model"] = model
        _nllb_cache["device"] = device
        info(f"  NLLB-200 loaded on {device}")

    tokenizer = _nllb_cache["tokenizer"]
    model = _nllb_cache["model"]
    device = _nllb_cache["device"]

    out = []
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    if not non_empty_indices:
        return texts

    # Translate in small batches rather than one giant batch, so a single
    # unusually long segment doesn't blow up memory for the whole run.
    batch_size = 8
    results_by_index: dict[int, str] = {}
    for start in range(0, len(non_empty_indices), batch_size):
        idx_batch = non_empty_indices[start:start + batch_size]
        text_batch = [texts[i] for i in idx_batch]

        tokenizer.src_lang = src_lang
        inputs = tokenizer(text_batch, truncation=True, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=256,
                num_beams=5,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, text in zip(idx_batch, decoded):
            results_by_index[i] = text

    for i, original in enumerate(texts):
        out.append(results_by_index.get(i, original))
    return out
