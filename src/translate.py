"""Translate a test set with one model, using settings from a YAML config.

Usage (on the Colab GPU):
    python src/translate.py --config configs/nllb600m_flores-devtest_en-hi.yaml
    python src/translate.py --config configs/... --limit 20      # quick smoke test

Writes to <output_root>/<run_name>/:
    hyp.txt     model translations, one per line
    src.txt     source sentences
    ref.txt     reference translations
    ids.txt     sentence ids (to match lines back to the test set)
    config.yaml the exact config used (after command-line overrides)
    meta.json   runtime, GPU, package versions
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANG_CODES, check_indicnlp_normalizer, load_config, parse_direction, restore_order, sort_by_length  # noqa: E402
from data import load_pair  # noqa: E402


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NLLBTranslator:
    """NLLB-200: the target language is chosen by forcing its language tag as the first output token."""

    def __init__(self, model_name: str, src: str, tgt: str, fp16: bool, device: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, src_lang=LANG_CODES[src])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if fp16 and device == "cuda":
            model = model.half()  # half precision: half the memory, faster on a T4
        self.model = model.to(device).eval()
        self.device = device
        self.tgt_token_id = self.tok.convert_tokens_to_ids(LANG_CODES[tgt])

    def translate_batch(self, sentences: list[str], num_beams: int, max_length: int) -> list[str]:
        import torch

        inputs = self.tok(sentences, return_tensors="pt", padding=True, truncation=True,
                          max_length=max_length).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, forced_bos_token_id=self.tgt_token_id,
                                      num_beams=num_beams, max_length=max_length)
        return self.tok.batch_decode(out, skip_special_tokens=True)


class IndicTrans2Translator:
    """IndicTrans2: needs IndicTransToolkit for pre/post-processing (adds language tags, script handling)."""

    def __init__(self, model_name: str, src: str, tgt: str, fp16: bool, device: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor

        check_indicnlp_normalizer()  # IndicProcessor normalises Indic input with indicnlp

        self.src_code, self.tgt_code = LANG_CODES[src], LANG_CODES[tgt]
        self.ip = IndicProcessor(inference=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        if fp16 and device == "cuda":
            model = model.half()
        self.model = model.to(device).eval()
        self.device = device

    def translate_batch(self, sentences: list[str], num_beams: int, max_length: int) -> list[str]:
        import torch

        batch = self.ip.preprocess_batch(sentences, src_lang=self.src_code, tgt_lang=self.tgt_code)
        inputs = self.tok(batch, return_tensors="pt", padding="longest", truncation=True,
                          max_length=max_length, return_attention_mask=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, num_beams=num_beams, max_length=max_length,
                                      min_length=0, use_cache=True)
        decoded = self.tok.batch_decode(out, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(decoded, lang=self.tgt_code)


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for pkg in ["torch", "transformers", "datasets", "sentencepiece", "IndicTransToolkit"]:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = "not installed"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML config in configs/")
    parser.add_argument("--limit", type=int, help="only translate the first N sentences (smoke test)")
    parser.add_argument("--output_root", help="override where outputs are written")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.output_root:
        cfg["output_root"] = args.output_root

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found, running on CPU (slow). In Colab: Runtime > Change runtime type > T4 GPU.")
    set_seed(cfg["seed"])

    src, tgt = parse_direction(cfg["direction"])
    ids, sources, refs = load_pair(cfg["test_set"], cfg["split"], src, tgt)
    if cfg["limit"]:
        ids, sources, refs = ids[: cfg["limit"]], sources[: cfg["limit"]], refs[: cfg["limit"]]
    print(f"{cfg['run_name']}: {len(sources)} sentences, {cfg['direction']}, model {cfg['model']} on {device}")

    translator_cls = {"nllb": NLLBTranslator, "indictrans2": IndicTrans2Translator}[cfg["model_family"]]
    translator = translator_cls(cfg["model"], src, tgt, cfg["fp16"], device)

    dec = cfg["decoding"]
    sorted_src, order = sort_by_length(sources)
    sorted_hyps: list[str] = []
    start = time.time()
    for i in range(0, len(sorted_src), dec["batch_size"]):
        batch = sorted_src[i : i + dec["batch_size"]]
        sorted_hyps.extend(translator.translate_batch(batch, dec["num_beams"], dec["max_length"]))
        print(f"  {min(i + dec['batch_size'], len(sorted_src))}/{len(sorted_src)}", end="\r")
    seconds = time.time() - start
    hyps = restore_order(sorted_hyps, order)
    print(f"\nDone in {seconds:.1f}s")

    out_dir = Path(cfg["output_root"]) / cfg["run_name"]
    if cfg["limit"]:
        out_dir = out_dir.with_name(out_dir.name + f"_limit{cfg['limit']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in [("hyp.txt", hyps), ("src.txt", sources), ("ref.txt", refs), ("ids.txt", ids)]:
        # Replace newlines inside a sentence so line i always means sentence i.
        (out_dir / name).write_text("\n".join(s.replace("\n", " ") for s in lines) + "\n", encoding="utf-8")
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    meta = {
        "n_sentences": len(sources),
        "seconds": round(seconds, 1),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "versions": package_versions(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved to {out_dir}")
    for s, h, r in list(zip(sources, hyps, refs))[:3]:
        print(f"\nSRC: {s}\nHYP: {h}\nREF: {r}")


if __name__ == "__main__":
    main()
