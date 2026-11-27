"""Tests for build_finetune_data.py (no downloads)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_finetune_data import (  # noqa: E402
    KeywordMatcher, OverlapIndex, clean_reason, detect_columns, is_tourism, norm, script_share,
)

CFG = {"min_words": 3, "max_words": 60, "max_length_ratio": 2.5, "min_script_share": 0.6}


def test_norm_keeps_hindi_vowel_signs():
    assert norm("मंदिर सुबह खुलता है।") == "मंदिर सुबह खुलता है"
    assert norm("The Temple, opens!") == "the temple opens"


def test_script_share():
    assert script_share("The temple opens at six.", "latin") == 1.0
    assert script_share("मंदिर सुबह खुलता है।", "devanagari") == 1.0
    assert script_share("मंदिर temple", "devanagari") < 0.6


def test_keyword_matcher_whole_words_and_phrases():
    m = KeywordMatcher(["hotel", "hill station", "fort", "ticket"])
    assert m.hits("The hotel near the hill station sells tickets.") == {"hotel", "hill station"}
    assert m.hits("A comfortable effort.") == set()  # 'fort' inside 'effort' must not match


def test_clean_reason():
    assert clean_reason("The fort opens at nine.", "किला नौ बजे खुलता है।", CFG) is None
    assert clean_reason("Hi there.", "नमस्ते।", CFG) == "too_short"
    assert clean_reason("The fort opens at nine every single day of the week.", "किला।", CFG) == "too_short"
    assert clean_reason("The fort opens at nine.", "The fort opens at nine.", CFG) == "wrong_script"


def test_overlap_exact_and_near_duplicate():
    idx = OverlapIndex(n=4)
    idx.add("The ferry to the island leaves every hour from the main jetty.", "tourism")
    idx.add("मंदिर सुबह छह बजे खुलता है और रात को बंद होता है।", "flores")
    assert idx.check("The ferry to the island leaves every hour from the main jetty!") == ("exact", "tourism")
    assert idx.check("Buses and the ferry to the island leaves at noon.") == ("near_dup", "tourism")
    assert idx.check("Something else.", "मंदिर सुबह छह बजे खुलता है।") == ("near_dup", "flores")
    assert idx.check("A completely different sentence about trains.") is None


def test_detect_columns_finds_english_and_hindi():
    rows = [["1", "मंदिर खुला है।", "The temple is open."], ["2", "किला बंद है।", "The fort is closed."]]
    assert detect_columns(rows) == (2, 1)


def test_is_tourism_needs_core_and_no_news():
    assert is_tourism({"temple"}, {"river"}, set(), 2)
    assert not is_tourism(set(), {"railway", "trains"}, set(), 2)        # transport news, no core word
    assert not is_tourism({"hotel"}, set(), set(), 2)                    # only one keyword
    assert not is_tourism({"hotels"}, {"restaurants"}, {"petition"}, 2)  # news about hotels
