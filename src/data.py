"""Load test sets as aligned (source, reference) sentence lists.

Supported test sets:
  - "flores":  FLORES-200 via Hugging Face (openlanguagedata/flores_plus), splits "dev" / "devtest"
  - "tourism": our own TourMT-Hi test set, read from data/tourism_test/tourism_test.tsv (built later)
"""
from __future__ import annotations

import csv
from pathlib import Path

from common import LANG_CODES

FLORES_REPO = "openlanguagedata/flores_plus"
TOURISM_TSV = Path(__file__).resolve().parent.parent / "data" / "tourism_test" / "tourism_test.tsv"


def load_flores(lang: str, split: str) -> dict[str, str]:
    """Return {sentence_id: text} for one language. Needs Hugging Face login (gated dataset)."""
    from datasets import load_dataset  # imported here so the rest of the file works without it

    ds = load_dataset(FLORES_REPO, LANG_CODES[lang], split=split)
    return {str(row["id"]): row["text"] for row in ds}


def load_tourism(lang: str, split: str = "test") -> dict[str, str]:
    """Return {sentence_id: text} from the tourism TSV (columns: id, en, hi, mai, ...)."""
    if not TOURISM_TSV.exists():
        raise FileNotFoundError(
            f"{TOURISM_TSV} does not exist yet. The tourism test set is built in a later step."
        )
    with open(TOURISM_TSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if rows and lang not in rows[0]:
        raise ValueError(f"tourism TSV has no column {lang!r}")
    # Skip rows where this language is empty (e.g. Maithili exists only for 100 sentences).
    return {r["id"]: r[lang] for r in rows if r.get(lang, "").strip()}


def load_pair(test_set: str, split: str, src: str, tgt: str) -> tuple[list[str], list[str], list[str]]:
    """Return (ids, source sentences, reference sentences), aligned by sentence id."""
    loader = {"flores": load_flores, "tourism": load_tourism}.get(test_set)
    if loader is None:
        raise ValueError(f"unknown test_set {test_set!r}; use 'flores' or 'tourism'")
    src_map, tgt_map = loader(src, split), loader(tgt, split)
    ids = [i for i in src_map if i in tgt_map]  # keep source order, only ids present in both
    if not ids:
        raise ValueError(f"no aligned sentences for {test_set}/{split} {src}-{tgt}")
    return ids, [src_map[i] for i in ids], [tgt_map[i] for i in ids]
