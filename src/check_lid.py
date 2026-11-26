"""How well does GlotLID tell Hindi from Maithili? (plan section 5.3: check this BEFORE trusting off-target rates)

Runs GlotLID on the professional FLORES references (Hindi and Maithili, same 1012 sentences) and reports
how often each is labelled correctly and how often they are confused with each other.

Usage (needs fasttext; runs fine on CPU):
    python src/check_lid.py --split devtest

Writes results/lid_check.json and prints a small table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_flores  # noqa: E402
from lid import GLOTLID_FILE, GLOTLID_REVISION, LID_LABELS, label_counts, predict_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--model_path", help="local GlotLID .bin (default: download the pinned model)")
    parser.add_argument("--out", default=str(ROOT / "results" / "lid_check.json"))
    args = parser.parse_args()

    report = {"split": args.split, "glotlid_file": GLOTLID_FILE, "glotlid_revision": GLOTLID_REVISION, "languages": {}}
    print(f"GlotLID on FLORES {args.split} references")
    print(f"{'true lang':<10}{'n':>6}{'correct':>10}{'as Hindi':>10}{'as Maithili':>13}  other labels")
    for lang in ["hi", "mai"]:
        texts = list(load_flores(lang, args.split).values())
        labels = predict_labels(texts, args.model_path)
        n = len(labels)
        correct = sum(lab == LID_LABELS[lang] for lab in labels) / n
        as_hi = sum(lab == LID_LABELS["hi"] for lab in labels) / n
        as_mai = sum(lab == LID_LABELS["mai"] for lab in labels) / n
        counts = label_counts(labels)
        others = {k: v for k, v in counts.items() if k not in (LID_LABELS["hi"], LID_LABELS["mai"])}
        report["languages"][lang] = {
            "n": n, "correct_rate": round(correct, 4), "as_hindi": round(as_hi, 4),
            "as_maithili": round(as_mai, 4), "label_counts": counts,
        }
        other_txt = ", ".join(f"{k.replace('__label__', '')}:{v}" for k, v in list(others.items())[:4])
        print(f"{lang:<10}{n:>6}{correct:>10.1%}{as_hi:>10.1%}{as_mai:>13.1%}  {other_txt}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
