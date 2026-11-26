"""Laptop-friendly tests (no GPU, no downloads). Run with:  python -m pytest tests/"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import load_config, parse_direction, restore_order, sort_by_length  # noqa: E402


def test_parse_direction():
    assert parse_direction("en-hi") == ("en", "hi")
    assert parse_direction("en-mai") == ("en", "mai")
    for bad in ["en", "en-xx", "en-en", "en-hi-mai"]:
        with pytest.raises(ValueError):
            parse_direction(bad)


def test_sort_and_restore_roundtrip():
    sents = ["a", "ccc", "bb", "dddd", ""]
    sorted_sents, order = sort_by_length(sents)
    assert sorted_sents == ["dddd", "ccc", "bb", "a", ""]
    fake_outputs = [s.upper() for s in sorted_sents]  # pretend translation
    assert restore_order(fake_outputs, order) == [s.upper() for s in sents]


def test_all_configs_are_valid():
    # translation configs only (wikivoyage.yaml configures data collection, not a model run)
    configs = sorted(p for p in (ROOT / "configs").glob("*.yaml") if p.name != "wikivoyage.yaml")
    assert configs, "no configs found"
    names = set()
    for path in configs:
        cfg = load_config(path)
        assert cfg["run_name"] == path.stem, f"{path.name}: run_name should equal the file name"
        assert cfg["run_name"] not in names
        names.add(cfg["run_name"])
