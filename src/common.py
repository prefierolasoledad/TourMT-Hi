"""Small shared helpers: language codes, config loading, length sorting.

Nothing in this file needs a GPU or torch, so it can be tested on a laptop.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Short code used in configs/filenames -> FLORES-200 code (also used by NLLB and IndicTrans2).
LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mai": "mai_Deva",
}


def parse_direction(direction: str) -> tuple[str, str]:
    """'en-hi' -> ('en', 'hi'). Raises ValueError for unknown languages."""
    parts = direction.split("-")
    if len(parts) != 2:
        raise ValueError(f"direction must look like 'en-hi', got {direction!r}")
    src, tgt = parts
    for lang in (src, tgt):
        if lang not in LANG_CODES:
            raise ValueError(f"unknown language {lang!r}; known: {sorted(LANG_CODES)}")
    if src == tgt:
        raise ValueError(f"source and target are the same: {direction!r}")
    return src, tgt


def load_config(path: str | Path) -> dict:
    """Read a YAML experiment config and check the required keys."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["run_name", "model", "model_family", "test_set", "split", "direction"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config keys {missing}")
    if cfg["model_family"] not in ("nllb", "indictrans2"):
        raise ValueError(f"{path}: model_family must be 'nllb' or 'indictrans2'")
    parse_direction(cfg["direction"])  # validate early
    cfg.setdefault("decoding", {})
    cfg["decoding"].setdefault("num_beams", 4)
    cfg["decoding"].setdefault("max_length", 256)
    cfg["decoding"].setdefault("batch_size", 16)
    cfg.setdefault("fp16", True)
    cfg.setdefault("seed", 42)
    cfg.setdefault("limit", None)
    cfg.setdefault("output_root", "outputs")
    return cfg


def check_indicnlp_normalizer() -> None:
    """Stop if the installed `indicnlp` package is the broken fork.

    `indic-nlp-library-itt` (pulled in by IndicTransToolkit's dependencies) writes the nukta as the
    literal text "\\u093C" instead of the character, corrupting Hindi text. Only the official
    `indic-nlp-library` must be installed (install IndicTransToolkit with --no-deps).
    """
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    normalizer = IndicNormalizerFactory().get_normalizer("hi")
    out = normalizer.normalize("ड़")  # ड़ as one code point -> should become ड + nukta characters
    if "\\" in out or out != "ड़":
        raise RuntimeError(
            f"Broken indicnlp normaliser (got {out!r}). Uninstall 'indic-nlp-library-itt' and install "
            "'indic-nlp-library'; install IndicTransToolkit with --no-deps."
        )


def sort_by_length(sentences: list[str]) -> tuple[list[str], list[int]]:
    """Sort sentences longest-first so each batch has similar lengths (less padding, faster).

    Returns the sorted sentences and the original index of each one.
    """
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
    return [sentences[i] for i in order], order


def restore_order(sorted_outputs: list[str], order: list[int]) -> list[str]:
    """Undo sort_by_length so output line i matches input line i."""
    restored = [""] * len(order)
    for out, original_index in zip(sorted_outputs, order):
        restored[original_index] = out
    return restored
