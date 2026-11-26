"""Tests for the language-ID helpers (no model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lid import label_counts, off_target  # noqa: E402

HI, MAI, ENG = "__label__hin_Deva", "__label__mai_Deva", "__label__eng_Latn"


def test_off_target_counts_hindi_separately():
    labels = [MAI, MAI, HI, ENG]
    r = off_target(labels, "mai")
    assert r["off_target_rate"] == 0.5  # HI + ENG are wrong
    assert r["hindi_rate"] == 0.25
    assert r["label_counts"] == {MAI: 2, HI: 1, ENG: 1}


def test_label_counts_sorted_by_frequency():
    assert list(label_counts([HI, MAI, MAI])) == [MAI, HI]
