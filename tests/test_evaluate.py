"""Tests for evaluate.py that need no GPU and no model downloads."""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import make_normalizer, read_lines, upsert_results  # noqa: E402


def test_read_lines_keeps_empty_translations(tmp_path):
    p = tmp_path / "hyp.txt"
    p.write_text("a\n\nc\n", encoding="utf-8")  # an empty translation in the middle must stay
    assert read_lines(p) == ["a", "", "c"]


def test_upsert_replaces_same_run(tmp_path):
    path = tmp_path / "results.csv"
    upsert_results(path, {"run_name": "b", "chrF++": 1})
    upsert_results(path, {"run_name": "a", "chrF++": 2})
    upsert_results(path, {"run_name": "b", "chrF++": 3})
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [(r["run_name"], r["chrF++"]) for r in rows] == [("a", "2"), ("b", "3")]


def test_english_target_is_not_normalised():
    norm, desc = make_normalizer("en", enabled=True)
    assert desc == "none" and norm("Hello") == "Hello"


def test_hindi_normalisation_composes_nukta():
    pytest.importorskip("indicnlp")
    norm, _ = make_normalizer("hi", enabled=True)
    decomposed = "ड़"  # ड + nukta (two code points)
    composed = "ड़"          # ड़ (one code point)
    assert norm(decomposed) == norm(composed)


def test_chrf_and_bleu_perfect_match():
    pytest.importorskip("sacrebleu")
    from evaluate import string_metrics

    sents = ["यह एक परीक्षण वाक्य है।", "मंदिर सुबह छह बजे खुलता है।"]
    s = string_metrics(sents, sents)
    assert s["chrF++"] == 100.0 and s["BLEU"] == 100.0
    assert "nrefs:1" in s["bleu_signature"]


def test_installed_indicnlp_is_not_the_broken_fork():
    pytest.importorskip("indicnlp")
    from common import check_indicnlp_normalizer

    check_indicnlp_normalizer()  # raises RuntimeError if indic-nlp-library-itt replaced indicnlp
