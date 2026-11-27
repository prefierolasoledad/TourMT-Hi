"""Fine-tune IndicTrans2 with LoRA on the tourism training data (plan section 5.2).

Usage (on a GPU, in the Python 3.11 environment /content/venv-eval - IndicTrans2 needs transformers<5):
    python src/train_lora.py --config configs/lora_it2-200m_arm-a_en-hi.yaml               # seed from config
    python src/train_lora.py --config configs/lora_it2-200m_arm-a_en-hi.yaml --seed 43     # another seed
    python src/train_lora.py --config ... --max_steps 30 --limit 500                       # 2-minute smoke test

What LoRA does: the 200M original weights stay FROZEN. Small extra matrices ("adapters", a few million
numbers) are added next to the attention and feed-forward layers, and only those are trained. That fits
a free T4, trains fast, and the result is a small file (~25 MB) instead of a full model copy.

Training stops early when the loss on the dev set (data/finetune/<arm>/dev.tsv - never a test set) stops
improving; the adapter from the best dev step is kept.

Writes:
    checkpoints/<run>/adapter/          the LoRA adapter (load it with translate.py: config key `adapter`)
    checkpoints/<run>/trainer/          Trainer checkpoints (only the best + last are kept)
    results/train_<run>.json            settings, data sizes, losses per evaluation, runtime (commit this)
where <run> = <run_name>_s<seed>.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANG_CODES, parse_direction  # noqa: E402

DEFAULTS = {
    "revision": None,
    "max_length": 256,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]},
    "training": {
        "learning_rate": 3e-4, "lr_scheduler": "inverse_sqrt", "warmup_steps": 200, "weight_decay": 0.01,
        "label_smoothing": 0.1, "batch_size": 32, "grad_accum_steps": 1, "max_epochs": 10,
        "eval_steps": 200, "patience": 5, "fp16": True,
    },
    "seed": 42,
    "output_root": "checkpoints",
}


def load_train_config(path: str | Path) -> dict:
    """Read a LoRA training config, fill defaults, check required keys."""
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    missing = [k for k in ("run_name", "base_model", "direction", "data_dir") if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config keys {missing}")
    parse_direction(cfg["direction"])
    for key, value in DEFAULTS.items():
        if isinstance(value, dict):
            cfg[key] = {**value, **(cfg.get(key) or {})}
        else:
            cfg.setdefault(key, value)
    return cfg


def read_pairs(path: str | Path, src: str, tgt: str) -> tuple[list[str], list[str]]:
    """Read train.tsv / dev.tsv written by build_finetune_data.py (columns source, en, hi, ...)."""
    csv.field_size_limit(10**7)
    sources, targets = [], []
    with open(path, encoding="utf-8", newline="") as f:
        # same format as the writer: tab-separated, no quoting, backslash escapes
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"):
            s, t = (row.get(src) or "").strip(), (row.get(tgt) or "").strip()
            if s and t:
                sources.append(s)
                targets.append(t)
    if not sources:
        raise ValueError(f"{path}: no '{src}'/'{tgt}' pairs found")
    return sources, targets


def check_data_is_clean(data_dir: Path) -> None:
    """Refuse to train on data built without the FLORES test-overlap check (plan 4.3 strict rule)."""
    name = data_dir.name
    report = ROOT / "results" / f"finetune_data_{name}.json"
    if not report.exists():
        raise FileNotFoundError(f"{report} not found: build the data with build_finetune_data.py first")
    rep = json.loads(report.read_text(encoding="utf-8"))
    if not rep.get("flores_overlap_checked", False):
        raise RuntimeError(f"{report}: data was built with --no-flores (no test-overlap check). Rebuild it.")


def versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("torch", "transformers", "peft", "accelerate", "indictranstoolkit"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="override the config seed (plan: 3 seeds)")
    parser.add_argument("--max_steps", type=int, help="stop after N steps (smoke test)")
    parser.add_argument("--limit", type=int, help="use only the first N training pairs (smoke test)")
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    smoke = bool(args.max_steps or args.limit)
    run = f"{cfg['run_name']}_s{cfg['seed']}" + ("_smoke" if smoke else "")
    out_dir = ROOT / cfg["output_root"] / run
    src, tgt = parse_direction(cfg["direction"])
    src_code, tgt_code = LANG_CODES[src], LANG_CODES[tgt]
    tr = cfg["training"]

    data_dir = ROOT / cfg["data_dir"]
    check_data_is_clean(data_dir)
    train_src, train_tgt = read_pairs(data_dir / "train.tsv", src, tgt)
    dev_src, dev_tgt = read_pairs(data_dir / "dev.tsv", src, tgt)
    if args.limit:
        train_src, train_tgt = train_src[: args.limit], train_tgt[: args.limit]
        dev_src, dev_tgt = dev_src[:100], dev_tgt[:100]
    print(f"{run}: {len(train_src)} train / {len(dev_src)} dev pairs, {cfg['direction']}, base {cfg['base_model']}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, EarlyStoppingCallback, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)
    from IndicTransToolkit import IndicDataCollator, IndicProcessor

    from common import check_indicnlp_normalizer
    check_indicnlp_normalizer()
    if not torch.cuda.is_available():
        print("WARNING: no GPU - training on CPU will take many hours. Use a T4 runtime.")
    set_seed(cfg["seed"])

    # --- 1. text -> model inputs. Same preprocessing as at translation time (IndicProcessor):
    # the source gets language tags ("eng_Latn hin_Deva ..."), the target is normalised without tags.
    ip = IndicProcessor(inference=False)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True, revision=cfg["revision"])

    def encode(srcs: list[str], tgts: list[str]) -> Dataset:
        s = ip.preprocess_batch(srcs, src_lang=src_code, tgt_lang=tgt_code, is_target=False)
        t = ip.preprocess_batch(tgts, src_lang=tgt_code, tgt_lang=src_code, is_target=True)
        enc = tok(s, truncation=True, max_length=cfg["max_length"])
        enc["labels"] = tok(text_target=t, truncation=True, max_length=cfg["max_length"])["input_ids"]
        return Dataset.from_dict(dict(enc))

    train_ds, dev_ds = encode(train_src, train_tgt), encode(dev_src, dev_tgt)

    # Safety check: targets must be encoded with the TARGET (Hindi) vocabulary. If they were encoded with
    # the source vocabulary, most Hindi pieces would become <unk> and training would be useless.
    unk = tok.unk_token_id
    label_tokens = [i for ex in train_ds.select(range(min(500, len(train_ds))))["labels"] for i in ex]
    unk_share = sum(i == unk for i in label_tokens) / max(len(label_tokens), 1)
    print(f"  unknown-token share in targets: {unk_share:.2%}")
    if unk_share > 0.05:
        raise RuntimeError("Too many <unk> tokens in the targets: the tokenizer's target mode is not working.")

    # --- 2. model + LoRA adapters (only the adapters are trained)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["base_model"], trust_remote_code=True,
                                                  revision=cfg["revision"], attn_implementation="eager")
    model.config.use_cache = False  # not needed for training; IndicTrans2's cache code breaks on new transformers
    model.set_label_smoothing(tr["label_smoothing"])  # IndicTrans2's own loss supports label smoothing
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="SEQ_2_SEQ_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})")

    # --- 3. training. Evaluate the dev loss every eval_steps; stop after `patience` evaluations
    # without improvement; keep the best checkpoint.
    steps_per_epoch = max(1, len(train_ds) // (tr["batch_size"] * tr["grad_accum_steps"]))
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=tr["batch_size"],
        per_device_eval_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["grad_accum_steps"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler"],
        warmup_steps=min(tr["warmup_steps"], (args.max_steps or 10**9) // 2),
        weight_decay=tr["weight_decay"],
        adam_beta2=0.98,
        max_grad_norm=1.0,
        num_train_epochs=tr["max_epochs"],
        max_steps=args.max_steps or -1,
        eval_strategy="steps",
        eval_steps=min(tr["eval_steps"], args.max_steps or 10**9),
        save_strategy="steps",
        save_steps=min(tr["eval_steps"], args.max_steps or 10**9),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=tr["fp16"] and torch.cuda.is_available(),
        group_by_length=True,       # batches of similar length: less padding, faster
        predict_with_generate=False,  # dev LOSS is enough for early stopping, and much faster
        seed=cfg["seed"],
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        # pads sources on the LEFT, as IndicTrans2 expects; label padding is ignored in the loss
        data_collator=IndicDataCollator(tokenizer=tok, model=model, padding="longest", pad_to_multiple_of=8,
                                        label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tr["patience"])],
    )
    print(f"  ~{steps_per_epoch} steps per epoch, at most {tr['max_epochs']} epochs; "
          f"dev loss every {targs.eval_steps} steps, patience {tr['patience']}")
    dev_loss_before = trainer.evaluate()["eval_loss"]  # the untouched base model, for comparison
    print(f"  dev loss before training: {dev_loss_before:.4f}")
    start = time.time()
    trainer.train()
    seconds = time.time() - start

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))  # best checkpoint (load_best_model_at_end); adapter weights only
    (adapter_dir / "train_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                                     encoding="utf-8")

    evals = [{"step": 0, "epoch": 0.0, "eval_loss": round(dev_loss_before, 4)}] + [{"step": h["step"], "epoch": round(h["epoch"], 2), "eval_loss": round(h["eval_loss"], 4)}
             for h in trainer.state.log_history if "eval_loss" in h and h["step"] > 0]
    train_losses = [{"step": h["step"], "loss": round(h["loss"], 4)} for h in trainer.state.log_history if "loss" in h]
    summary = {
        "run": run, "config": cfg, "smoke_test": smoke,
        "data": {"train": len(train_ds), "dev": len(dev_ds), "dir": cfg["data_dir"]},
        "trainable_parameters": trainable, "total_parameters": total,
        "best_step": trainer.state.best_global_step if hasattr(trainer.state, "best_global_step") else None,
        "best_dev_loss": trainer.state.best_metric, "steps_done": trainer.state.global_step,
        "dev_losses": evals, "train_losses": train_losses,
        "seconds": round(seconds, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": versions(),
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    res = ROOT / "results" / f"train_{run}.json"
    res.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {seconds / 60:.1f} min. Best dev loss {trainer.state.best_metric:.4f} "
          f"(before training: {dev_loss_before:.4f}). Adapter: {adapter_dir}\nSummary: {res}")


if __name__ == "__main__":
    main()
