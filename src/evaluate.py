"""Score translation runs with chrF++, BLEU, COMET and CometKiwi.

Usage (COMET needs the Colab GPU; chrF++/BLEU also run on a laptop):
    python src/evaluate.py --runs outputs/nllb600m_flores-devtest_en-hi outputs/nllb600m_flores-devtest_hi-en
    python src/evaluate.py --runs outputs/* --no-comet --no-kiwi     # quick, CPU only

Each run folder must contain hyp.txt, ref.txt, src.txt and config.yaml (written by translate.py).

Writes:
    <run>/scores.json            all corpus-level scores + metric signatures + normalisation setting
    <run>/segment_scores.tsv     per-sentence COMET / CometKiwi (for per-category analysis later)
    results/results.csv          one row per run (re-running a run replaces its row)

Metrics (plan section 6.1):
    chrF++    primary; sacrebleu CHRF with word n-grams (word_order=2)
    BLEU      sacrebleu default (13a tokenizer), for comparison with older papers
    COMET     Unbabel/wmt22-comet-da (uses source + reference)
    CometKiwi Unbabel/wmt22-cometkiwi-da (source only, no reference; gated on Hugging Face)
    off_target_rate  for Maithili targets only: share of outputs GlotLID does not label as Maithili

Normalisation: for Devanagari targets (hi, mai), references and outputs are normalised with
indic-nlp-library before every metric; the setting is saved with the scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_direction  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVANAGARI_TARGETS = {"hi", "mai"}
COMET_MODEL = "Unbabel/wmt22-comet-da"
KIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"
GLOTLID_REPO = "cis-lmu/glotlid"

RESULT_COLUMNS = [
    "run_name", "model", "test_set", "split", "direction", "n_sentences",
    "chrF++", "BLEU", "COMET", "CometKiwi", "off_target_rate",
    "normalization", "chrf_signature", "bleu_signature", "evaluated_at",
]


# ---------- reading a run ----------

def read_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # translate.py ends files with one newline
    return lines


def load_run(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    run = {"dir": run_dir, "cfg": cfg}
    for name in ["src", "hyp", "ref"]:
        run[name] = read_lines(run_dir / f"{name}.txt")
    n = {len(run["src"]), len(run["hyp"]), len(run["ref"])}
    if len(n) != 1:
        raise ValueError(f"{run_dir}: src/hyp/ref have different line counts")
    return run


# ---------- normalisation ----------

def make_normalizer(tgt: str, enabled: bool):
    """Return (function, description). Only Devanagari targets are normalised."""
    if not enabled or tgt not in DEVANAGARI_TARGETS:
        return (lambda s: s), "none"
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    # Hindi rules are used for Maithili too: same Devanagari script and the same character variants.
    normalizer = IndicNormalizerFactory().get_normalizer("hi")
    return normalizer.normalize, "indicnlp-hi (nukta + alternative forms)"


# ---------- metrics ----------

def string_metrics(hyps: list[str], refs: list[str]) -> dict:
    """chrF++ and BLEU from sacrebleu (never a custom implementation)."""
    from sacrebleu.metrics import BLEU, CHRF

    chrf = CHRF(word_order=2)  # word_order=2 makes chrF into chrF++
    bleu = BLEU()
    chrf_score = chrf.corpus_score(hyps, [refs])
    bleu_score = bleu.corpus_score(hyps, [refs])
    return {
        "chrF++": round(chrf_score.score, 2),
        "BLEU": round(bleu_score.score, 2),
        "chrf_signature": str(chrf.get_signature()),
        "bleu_signature": str(bleu.get_signature()),
    }


_COMET_CACHE: dict = {}


def comet_scores(model_name: str, srcs: list[str], hyps: list[str], refs: list[str] | None,
                 batch_size: int) -> tuple[float, list[float]]:
    """Run a COMET model. refs=None for reference-free models (CometKiwi)."""
    import torch
    from comet import download_model, load_from_checkpoint

    if model_name not in _COMET_CACHE:
        _COMET_CACHE[model_name] = load_from_checkpoint(download_model(model_name))
    model = _COMET_CACHE[model_name]
    if refs is None:
        data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
    else:
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0,
                        progress_bar=False)
    return round(float(out.system_score), 4), [round(float(x), 4) for x in out.scores]


def off_target_rate(hyps: list[str], expected_label: str = "__label__mai_Deva") -> tuple[float, list[str]]:
    """Share of outputs that GlotLID does not identify as the expected language."""
    import fasttext
    from huggingface_hub import hf_hub_download

    model = fasttext.load_model(hf_hub_download(GLOTLID_REPO, "model.bin"))
    labels = [model.predict(h.replace("\n", " "))[0][0] for h in hyps]
    wrong = sum(1 for lab in labels if lab != expected_label)
    return round(wrong / len(hyps), 4), labels


# ---------- results table ----------

def upsert_results(csv_path: Path, row: dict) -> None:
    """Add a row to results.csv, replacing any older row with the same run_name."""
    rows = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_name"] != row["run_name"]]
    rows.append({k: row.get(k, "") for k in RESULT_COLUMNS})
    rows.sort(key=lambda r: r["run_name"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------- main ----------

def evaluate_run(run_dir: Path, args) -> dict:
    run = load_run(run_dir)
    cfg = run["cfg"]
    src_lang, tgt_lang = parse_direction(cfg["direction"])
    norm, norm_desc = make_normalizer(tgt_lang, not args.no_normalize)
    hyps = [norm(h) for h in run["hyp"]]
    refs = [norm(r) for r in run["ref"]]
    srcs = run["src"]  # sources are not normalised (they are the model input, not compared text)

    print(f"\n== {cfg['run_name']} ({len(hyps)} sentences, normalisation: {norm_desc})")
    scores = {
        "run_name": cfg["run_name"], "model": cfg["model"], "test_set": cfg["test_set"],
        "split": cfg["split"], "direction": cfg["direction"], "n_sentences": len(hyps),
        "normalization": norm_desc,
    }
    scores.update(string_metrics(hyps, refs))
    print(f"  chrF++ {scores['chrF++']}   BLEU {scores['BLEU']}")

    segments: dict[str, list] = {}
    if not args.no_comet:
        scores["COMET"], segments["COMET"] = comet_scores(COMET_MODEL, srcs, hyps, refs, args.batch_size)
        print(f"  COMET {scores['COMET']}")
    if not args.no_kiwi:
        scores["CometKiwi"], segments["CometKiwi"] = comet_scores(KIWI_MODEL, srcs, hyps, None, args.batch_size)
        print(f"  CometKiwi {scores['CometKiwi']}")
    if tgt_lang == "mai" and not args.no_lid:
        scores["off_target_rate"], segments["lid_label"] = off_target_rate(run["hyp"])
        print(f"  off-target rate {scores['off_target_rate']}")
    if not args.no_comet:
        scores["comet_model"] = COMET_MODEL
    if not args.no_kiwi:
        scores["kiwi_model"] = KIWI_MODEL
    scores["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    if segments:
        with open(run_dir / "segment_scores.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["line", *segments])
            for i in range(len(hyps)):
                writer.writerow([i, *(segments[k][i] for k in segments)])
    upsert_results(Path(args.results_csv), scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run folders in outputs/")
    parser.add_argument("--results_csv", default=str(ROOT / "results" / "results.csv"))
    parser.add_argument("--batch_size", type=int, default=32, help="COMET batch size")
    parser.add_argument("--no-comet", dest="no_comet", action="store_true", help="skip COMET")
    parser.add_argument("--no-kiwi", dest="no_kiwi", action="store_true", help="skip CometKiwi")
    parser.add_argument("--no-lid", dest="no_lid", action="store_true", help="skip Maithili off-target check")
    parser.add_argument("--no-normalize", dest="no_normalize", action="store_true",
                        help="skip Indic normalisation (only for checking its effect)")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs if (Path(r) / "hyp.txt").exists()]
    skipped = [r for r in args.runs if not (Path(r) / "hyp.txt").exists()]
    if skipped:
        print(f"Skipping folders without hyp.txt: {skipped}")
    if not run_dirs:
        sys.exit("No run folders to evaluate.")
    for run_dir in run_dirs:
        evaluate_run(run_dir, args)
    print(f"\nResults table: {args.results_csv}")


if __name__ == "__main__":
    main()
