"""Tests for significance.py (no GPU, no downloads; the sacrebleu part is skipped if it is not installed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from significance import bootstrap_indices, bootstrap_mean_diff, compare, p_value_two_sided  # noqa: E402


def test_p_value_two_sided():
    assert p_value_two_sided(np.array([1.0] * 100), 1.0) == 0.0             # always better: tiny p
    assert p_value_two_sided(np.array([1.0] * 90 + [-1.0] * 10), 1.0) == 0.2
    assert p_value_two_sided(np.array([-1.0] * 50 + [1.0] * 50), 1.0) == 1.0  # coin flip


def test_bootstrap_is_paired_and_reproducible():
    idx = bootstrap_indices(5, 200, seed=1)
    assert (idx == bootstrap_indices(5, 200, seed=1)).all()
    base = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # a system exactly 0.1 better on EVERY sentence: paired deltas are always 0.1, whatever the resample
    assert np.allclose(bootstrap_mean_diff(base, base + 0.1, idx), 0.1)


def _write_run(d: Path, hyps, refs, direction="hi-en"):
    d.mkdir()
    (d / "hyp.txt").write_text("\n".join(hyps) + "\n", encoding="utf-8")
    (d / "ref.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    (d / "src.txt").write_text("\n".join("x" for _ in refs) + "\n", encoding="utf-8")
    (d / "ids.txt").write_text("\n".join(str(i) for i in range(len(refs))) + "\n", encoding="utf-8")
    (d / "config.yaml").write_text(f"run_name: {d.name}\ndirection: {direction}\n", encoding="utf-8")


def test_compare_matches_sacrebleu_and_detects_a_clear_gap(tmp_path):
    pytest.importorskip("sacrebleu")
    from sacrebleu.metrics import CHRF

    refs = [f"the temple opens at {i} in the morning near the old fort" for i in range(60)]
    good = list(refs)
    bad = [f"a shrine is open from {i} early close to an ancient castle" for i in range(60)]
    _write_run(tmp_path / "base", bad, refs)
    _write_run(tmp_path / "sys", good, refs)
    rows = {r["metric"]: r for r in compare(tmp_path / "base", tmp_path / "sys", n_samples=200, seed=1)}
    assert rows["chrF++"]["baseline_score"] == round(CHRF(word_order=2).corpus_score(bad, [refs]).score, 2)
    assert rows["chrF++"]["delta"] > 0 and rows["chrF++"]["significant"]
    assert rows["chrF++"]["ci_low"] > 0


def test_compare_refuses_different_test_sentences(tmp_path):
    pytest.importorskip("sacrebleu")
    _write_run(tmp_path / "a", ["x y z"], ["a b c"])
    _write_run(tmp_path / "b", ["x y z"], ["d e f"])
    with pytest.raises(ValueError):
        compare(tmp_path / "a", tmp_path / "b", n_samples=10, seed=1)
