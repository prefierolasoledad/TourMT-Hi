"""Tests for collect_wikivoyage.py helpers (no downloads)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collect_wikivoyage import categorise, has_named_entity_guess, make_shortlist, reject_reason  # noqa: E402

MAPPING = {"Eat": "food", "Get in": "transport"}


@pytest.mark.parametrize("sentence,reason", [
    ("The ghats are busiest at sunrise every day.", None),
    ("Very nice.", "too_short"),
    ("the ghats are busiest at sunrise every day.", "bad_start"),
    ("The ghats are busiest at sunrise every day", "bad_end"),
    ("Book online at www.example.com before you go.", "url_or_email"),
    ("Call +91 9369305877 to book a table now.", "phone_number"),
    ("The temple (near the river is open all day.", "unbalanced_brackets"),
    ("The word for water is पानी in Hindi here.", "devanagari"),
])
def test_reject_reason(sentence, reason):
    assert reject_reason(sentence, 5, 40) == reason


def test_prices_and_times_override_section():
    assert categorise("Eat", "A thali costs ₹150 at lunch.", MAPPING) == "prices_times"
    assert categorise("Get in", "The museum opens at 10:00 daily.", MAPPING) == "prices_times"
    assert categorise("Eat", "Try the lassi near the fort.", MAPPING) == "food"
    assert categorise("Go next", "Visit the hills nearby.", MAPPING) == "other"


def test_named_entity_guess():
    assert has_named_entity_guess("Take a boat to Assi Ghat at dawn.")
    assert not has_named_entity_guess("Take a boat at dawn to see the river.")


def test_shortlist_is_balanced_and_skips_other():
    cands = [{"id": f"t{i:03d}", "destination": f"d{i % 3}", "category_guess": cat}
             for i, cat in enumerate(["food"] * 30 + ["transport"] * 30 + ["other"] * 30)]
    short = make_shortlist(cands, size=20, seed=1)
    cats = [c["category_guess"] for c in short]
    assert cats.count("food") == 10 and cats.count("transport") == 10 and "other" not in cats
    assert len({c["destination"] for c in short if c["category_guess"] == "food"}) == 3  # spread over destinations


def test_shortlist_with_manual_work_is_detected(tmp_path):
    from collect_wikivoyage import shortlist_has_manual_work

    p = tmp_path / "shortlist.tsv"
    p.write_text("id\tenglish\tkeep\thindi\nt1\tHello there.\t\t\n", encoding="utf-8")
    assert not shortlist_has_manual_work(p)
    p.write_text("id\tenglish\tkeep\thindi\nt1\tHello there.\ty\tनमस्ते।\n", encoding="utf-8")
    assert shortlist_has_manual_work(p)
    assert not shortlist_has_manual_work(tmp_path / "missing.tsv")


def test_shortlist_number_quota():
    # 10% of food sentences have a number; with number_share=0.5 half the picks must have one
    cands = [{"id": f"t{i:03d}", "destination": f"d{i % 4}", "category_guess": "food",
              "has_number": int(i % 10 == 0)} for i in range(200)]
    short = make_shortlist(cands, size=20, seed=3, number_share=0.5)
    assert len(short) == 20
    assert sum(c["has_number"] for c in short) >= 10  # a minimum: the random rest may add more
    # without the quota, about 10% would have a number
    plain = make_shortlist(cands, size=20, seed=3)
    assert sum(c["has_number"] for c in plain) < 10
