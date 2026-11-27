"""Tests for train_lora.py helpers (no GPU, no downloads)."""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import load_config  # noqa: E402
from train_lora import load_train_config, read_pairs  # noqa: E402


def test_lora_configs_load_with_defaults():
    configs = sorted((ROOT / "configs").glob("lora_*.yaml"))
    assert configs, "no LoRA training configs"
    for path in configs:
        cfg = load_train_config(path)
        assert cfg["run_name"] == path.stem
        assert cfg["training"]["batch_size"] > 0 and cfg["lora"]["r"] > 0
        assert "patience" in cfg["training"]  # defaults are merged into partial sections


def test_adapter_eval_configs_point_to_a_training_run():
    runs = {load_train_config(p)["run_name"] for p in (ROOT / "configs").glob("lora_*.yaml")}
    for path in (ROOT / "configs").glob("lora-*_*.yaml"):
        cfg = load_config(path)
        assert cfg["model_family"] == "indictrans2"
        run_dir = Path(cfg["adapter"]).parent.name  # checkpoints/<run_name>_s<seed>/adapter
        assert run_dir.rsplit("_s", 1)[0] in runs, f"{path.name}: adapter from an unknown training run"


def test_read_pairs_reads_build_output_format(tmp_path):
    rows = [{"source": "ilci", "en": 'The "pink city" has forts.', "hi": "गुलाबी शहर में\tकिले हैं।",
             "keywords": "fort", "labse": 0.9},
            {"source": "samanantar", "en": "Back\\slash test here.", "hi": "बैकस्लैश परीक्षण यहाँ।",
             "keywords": "", "labse": 0.85}]
    path = tmp_path / "train.tsv"
    with open(path, "w", encoding="utf-8", newline="") as f:  # exactly as build_finetune_data.py writes it
        w = csv.DictWriter(f, fieldnames=["source", "en", "hi", "keywords", "labse"], delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    en, hi = read_pairs(path, "en", "hi")
    assert en == [r["en"] for r in rows] and hi == [r["hi"] for r in rows]
    hi2, en2 = read_pairs(path, "hi", "en")  # the other direction
    assert hi2 == hi and en2 == en


def test_read_pairs_empty_file_fails(tmp_path):
    path = tmp_path / "dev.tsv"
    path.write_text("source\ten\thi\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_pairs(path, "en", "hi")
