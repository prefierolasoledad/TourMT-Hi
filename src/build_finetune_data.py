"""Build tourism-like English-Hindi fine-tuning data (plan section 4.3), with a strict test-overlap check.

Steps:
  1. read pairs from the sources in the config (ILCI from BPCC, Samanantar), pinned to exact revisions
  2. clean: length, length ratio, script check (English side Latin, Hindi side Devanagari), duplicates
  3. keep "tourism-like" pairs: English side has a core tourism keyword, >= min_keyword_hits keywords
     in total, and no news words (Samanantar is mostly news: "minister said", "police", ...)
  4. REMOVE every pair that overlaps any test data (exact match or a shared word n-gram), on either side:
     FLORES-200 dev + devtest (en, hi, mai), all tourism candidates (data/tourism_test/candidates.tsv),
     and any Hindi/Maithili already written for the tourism set (shortlist.tsv, tourism_test.tsv)
  5. alignment check: drop pairs whose two sides do not mean the same thing (LaBSE similarity; Samanantar is
     mined from the web and contains mis-aligned pairs)
  6. sample up to max_pairs (ILCI first, it is human-translated), split off a dev set, write files + report

Usage (Colab or laptop; CPU only; Hugging Face login needed for the gated BPCC and FLORES+):
    python src/build_finetune_data.py --config configs/finetune_arm_a.yaml
    python src/build_finetune_data.py --config configs/finetune_arm_a.yaml --max_rows 50000   # quick try
Samanantar is read from its pinned parquet files (1.9 GB for Hindi, downloaded shard by shard).

Outputs:
    data/finetune/<name>/train.tsv, dev.tsv      (not committed: size + licences; re-create with this script)
    results/finetune_data_<name>.json            counts for every step, incl. pairs removed for overlap
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
TOURISM_DIR = ROOT / "data" / "tourism_test"


# ---------- text helpers ----------

def norm(text: str) -> str:
    """Lowercase, remove punctuation/symbols, collapse spaces. Keeps Devanagari vowel signs intact
    (unlike regex \\W, which would treat them as non-word characters and split Hindi words)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in "PSZ" else ch for ch in text)
    return " ".join(text.split())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def script_share(text: str, script: str) -> float:
    """Share of letters that belong to the given script ('latin' or 'devanagari')."""
    letters = [ch for ch in text if unicodedata.category(ch)[0] in "LM"]
    if not letters:
        return 0.0
    if script == "devanagari":
        ok = sum("ऀ" <= ch <= "ॿ" or "꣠" <= ch <= "ꣿ" for ch in letters)
    else:
        ok = sum(ch.isascii() for ch in letters)
    return ok / len(letters)


def is_tourism(core: set[str], support: set[str], negative: set[str], min_hits: int) -> bool:
    """Tourism-like = at least one CORE keyword, at least min_hits keywords in total, and no NEWS word."""
    return bool(core) and len(core | support) >= min_hits and not negative


class KeywordMatcher:
    def __init__(self, keywords: list[str]):
        words = sorted({k.strip().lower() for k in keywords if k.strip()}, key=len, reverse=True)
        self.re = re.compile(r"(?<![\w-])(" + "|".join(re.escape(w) for w in words) + r")(?![\w-])", re.I)

    def hits(self, text: str) -> set[str]:
        return {m.lower() for m in self.re.findall(text)}


def clean_reason(en: str, hi: str, cfg: dict) -> str | None:
    """None if the pair passes the cleaning filters, otherwise the reason."""
    n_en, n_hi = len(en.split()), len(hi.split())
    if min(n_en, n_hi) < cfg["min_words"]:
        return "too_short"
    if max(n_en, n_hi) > cfg["max_words"]:
        return "too_long"
    if max(len(en), len(hi)) / max(min(len(en), len(hi)), 1) > cfg["max_length_ratio"]:
        return "length_ratio"
    if script_share(en, "latin") < cfg["min_script_share"] or script_share(hi, "devanagari") < cfg["min_script_share"]:
        return "wrong_script"
    if norm(en) == norm(hi):
        return "untranslated"
    return None


# ---------- test-set overlap index ----------

class OverlapIndex:
    """Exact-match keys and word n-grams of every test sentence, remembering which test set they came from."""

    def __init__(self, n: int):
        self.n = n
        self.exact: dict[str, str] = {}
        self.grams: dict[tuple[str, ...], str] = {}
        self.sizes: Counter = Counter()

    def add(self, text: str, test_set: str) -> None:
        key = norm(text)
        if not key:
            return
        self.exact.setdefault(key, test_set)
        for g in ngrams(key.split(), self.n):
            self.grams.setdefault(g, test_set)
        self.sizes[test_set] += 1

    def check(self, *texts: str) -> tuple[str, str] | None:
        """(kind, test_set) if any text overlaps, else None. kind is 'exact' or 'near_dup'."""
        for t in texts:
            key = norm(t)
            if key in self.exact:
                return "exact", self.exact[key]
        for t in texts:
            for g in ngrams(norm(t).split(), self.n):
                if g in self.grams:
                    return "near_dup", self.grams[g]
        return None


def build_overlap_index(n: int, include_flores: bool = True) -> OverlapIndex:
    idx = OverlapIndex(n)
    if include_flores:
        from data import load_flores

        for split in ["dev", "devtest"]:
            for lang in ["en", "hi", "mai"]:
                for text in load_flores(lang, split).values():
                    idx.add(text, f"flores_{split}_{lang}")
    for name, cols in [("candidates.tsv", ["english"]), ("shortlist.tsv", ["english", "hindi"]),
                       ("tourism_test.tsv", ["en", "hi", "mai"])]:
        path = TOURISM_DIR / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                for col in cols:
                    if (row.get(col) or "").strip():
                        idx.add(row[col], f"tourism_{name.split('.')[0]}_{col}")
    return idx


# ---------- sources ----------

def detect_columns(rows: list[list[str]]) -> tuple[int, int]:
    """Find which TSV columns are English (Latin) and Hindi (Devanagari) from sample rows."""
    n_cols = max(len(r) for r in rows)

    def is_text(c: int) -> bool:  # real sentences have several words; language codes like "eng_Latn" do not
        vals = [r[c] for r in rows if len(r) > c]
        return bool(vals) and sum(len(v.split()) for v in vals) / len(vals) >= 2

    latin = [sum(script_share(r[c], "latin") for r in rows if len(r) > c) if is_text(c) else -1 for c in range(n_cols)]
    deva = [sum(script_share(r[c], "devanagari") for r in rows if len(r) > c) if is_text(c) else -1 for c in range(n_cols)]
    en_col, hi_col = max(range(n_cols), key=lambda c: latin[c]), max(range(n_cols), key=lambda c: deva[c])
    if en_col == hi_col:
        raise ValueError("could not tell the English and Hindi columns apart")
    return en_col, hi_col


def tsv_pairs(rows: Iterator[list[str]]) -> Iterator[tuple[str, str]]:
    """(English, Hindi) pairs from TSV rows (first row = header if it names the columns).

    BPCC files have the header  src_lang  tgt_lang  src  tgt  ; the language columns say which side is
    English (rows in other language pairs are skipped). Without such a header, the English and Hindi
    columns are detected from the script of the first 200 rows.
    """
    first = next(rows)
    names = [h.strip().lower() for h in first]
    if "src" in names and "tgt" in names:
        i_src, i_tgt = names.index("src"), names.index("tgt")
        i_sl = names.index("src_lang") if "src_lang" in names else None
        i_tl = names.index("tgt_lang") if "tgt_lang" in names else None
        print(f"  header: {first}")
        for row in rows:
            if len(row) <= max(i_src, i_tgt):
                continue
            sl = row[i_sl].strip() if i_sl is not None else "eng_Latn"
            tl = row[i_tl].strip() if i_tl is not None else "hin_Deva"
            if sl.startswith("eng") and tl.startswith("hin"):
                yield row[i_src].strip(), row[i_tgt].strip()
            elif sl.startswith("hin") and tl.startswith("eng"):
                yield row[i_tgt].strip(), row[i_src].strip()
        return
    head = [first] + [r for _, r in zip(range(199), rows)]
    en_col, hi_col = detect_columns(head)
    print(f"  no header; English = column {en_col}, Hindi = column {hi_col}")
    for row in _chain(head, rows):
        if len(row) > max(en_col, hi_col):
            yield row[en_col].strip(), row[hi_col].strip()


def read_ilci(src: dict, max_rows: int | None) -> Iterator[tuple[str, str]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(src["repo"], src["file"], repo_type="dataset", revision=src["revision"])
    csv.field_size_limit(10**7)
    with open(path, encoding="utf-8", newline="") as f:
        for i, pair in enumerate(tsv_pairs(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))):
            if max_rows and i >= max_rows:
                break
            yield pair


def _chain(first: list, rest) -> Iterator:
    yield from first
    yield from rest


def read_samanantar(src: dict, max_rows: int | None) -> Iterator[tuple[str, str]]:
    """Download the pinned parquet shards one by one (about 240 MB each, 8 for Hindi) and read them."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(src["repo"], repo_type="dataset", revision=src["revision"])
                   if f.startswith(f"{src['config']}/") and f.endswith(".parquet"))
    n = 0
    for file in files:
        path = hf_hub_download(src["repo"], file, repo_type="dataset", revision=src["revision"])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10000, columns=["src", "tgt"]):
            for en, hi in zip(batch.column("src").to_pylist(), batch.column("tgt").to_pylist()):
                if max_rows and n >= max_rows:
                    return
                n += 1
                yield (en or "").strip(), (hi or "").strip()  # src = English, tgt = Hindi


READERS = {"ilci": read_ilci, "samanantar": read_samanantar}


# ---------- alignment check (LaBSE) ----------

def labse_scores(en: list[str], hi: list[str], model_id: str, revision: str, batch_size: int) -> list[float]:
    """Cosine similarity of LaBSE sentence embeddings: high = the two sentences mean the same thing.
    Catches mis-aligned pairs from web mining (English sentence paired with an unrelated Hindi one)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  LaBSE on {device} for {len(en)} pairs" + ("" if device == "cuda" else " (CPU: a few minutes)"))
    model = SentenceTransformer(model_id, revision=revision, device=device)
    kw = {"batch_size": batch_size, "normalize_embeddings": True, "convert_to_numpy": True}
    e, h = model.encode(en, **kw), model.encode(hi, **kw)
    return [round(float(x), 4) for x in (e * h).sum(axis=1)]


def quantiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {f"p{q}": xs[min(len(xs) - 1, int(q / 100 * len(xs)))] for q in (5, 10, 25, 50, 75, 90)} if xs else {}


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_rows", type=int, help="read at most N rows per source (quick test)")
    parser.add_argument("--sources", nargs="+", help="only these sources (default: all in config)")
    parser.add_argument("--no-labse", dest="no_labse", action="store_true",
                        help="skip the LaBSE alignment check (ONLY for offline tests)")
    parser.add_argument("--no-flores", dest="no_flores", action="store_true",
                        help="skip the FLORES overlap check (ONLY for offline tests; never for real data)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    print("Building test-overlap index...")
    index = build_overlap_index(cfg["near_dup_ngram"], include_flores=not args.no_flores)
    print(f"  {sum(index.sizes.values())} test sentences: {dict(index.sizes)}")

    core_m, support_m = KeywordMatcher(cfg["core_keywords"]), KeywordMatcher(cfg["support_keywords"])
    negative_m = KeywordMatcher(cfg["negative_keywords"])
    report: dict = {"name": cfg["name"], "config": cfg, "test_sentences_indexed": dict(index.sizes), "sources": {}}
    kept: list[dict] = []
    seen: set[str] = set()
    for name, src in cfg["sources"].items():
        if args.sources and name not in args.sources:
            continue
        max_rows = args.max_rows or src.get("max_rows")
        stats: Counter = Counter()
        overlap: Counter = Counter()
        kw_freq: Counter = Counter()
        print(f"Reading {name} (max_rows={max_rows})...")
        for en, hi in READERS[name](src, max_rows):
            stats["read"] += 1
            reason = clean_reason(en, hi, cfg)
            if reason:
                stats[f"removed_{reason}"] += 1
                continue
            core, support, negative = core_m.hits(en), support_m.hits(en), negative_m.hits(en)
            if not is_tourism(core, support, negative, cfg["min_keyword_hits"]):
                stats["not_tourism_news" if (core and negative) else "not_tourism"] += 1
                continue
            hits = core | support
            key = norm(en)
            if key in seen:
                stats["removed_duplicate"] += 1
                continue
            seen.add(key)
            ov = index.check(en, hi)
            if ov:
                stats[f"removed_overlap_{ov[0]}"] += 1
                overlap[ov[1]] += 1
                continue
            stats["kept"] += 1
            kw_freq.update(hits)
            kept.append({"source": name, "en": en, "hi": hi, "keywords": ",".join(sorted(hits))})
            if stats["read"] % 500000 == 0:
                print(f"  {stats['read']} read, {stats['kept']} kept")
        report["sources"][name] = {"counts": dict(stats), "overlap_removed_by_test_set": dict(overlap),
                                   "top_keywords": dict(kw_freq.most_common(25))}
        print(f"  {name}: {dict(stats)}")
        print(f"  removed for test overlap: {sum(overlap.values())} {dict(overlap)}")

    # alignment check: drop pairs whose English and Hindi do not mean the same thing
    al = cfg["alignment"]
    if kept and not args.no_labse:
        print(f"Alignment check ({al['model']}, min score {al['min_score']})...")
        scores = labse_scores([p["en"] for p in kept], [p["hi"] for p in kept], al["model"], al["revision"],
                              al["batch_size"])
        for p, sc in zip(kept, scores):
            p["labse"] = sc
        for name in report["sources"]:
            src_scores = [p["labse"] for p in kept if p["source"] == name]
            low = sum(sc < al["min_score"] for sc in src_scores)
            report["sources"][name]["labse_quantiles"] = quantiles(src_scores)
            report["sources"][name]["counts"]["removed_low_alignment"] = low
            report["sources"][name]["counts"]["kept"] = report["sources"][name]["counts"].get("kept", 0) - low
            print(f"  {name}: removed {low} of {len(src_scores)} (score quantiles {quantiles(src_scores)})")
        report["low_alignment_examples"] = [{k: p[k] for k in ("source", "labse", "en", "hi")}
                                            for p in kept if p["labse"] < al["min_score"]][:15]
        kept = [p for p in kept if p["labse"] >= al["min_score"]]
    report["alignment_checked"] = not args.no_labse

    # sample: ILCI first (human-translated), then the rest
    rng = random.Random(cfg["seed"])
    total = cfg["max_pairs"] + cfg["dev_size"]
    chosen: list[dict] = []
    for name in ["ilci"] + [n for n in cfg["sources"] if n != "ilci"]:
        pool = [p for p in kept if p["source"] == name]
        rng.shuffle(pool)
        chosen += pool[: max(total - len(chosen), 0)]
    rng.shuffle(chosen)
    n_dev = min(cfg["dev_size"], len(chosen) // 10)  # never more than 10% of the data
    dev, train = chosen[:n_dev], chosen[n_dev:]

    out_dir = ROOT / "data" / "finetune" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in [("train", train), ("dev", dev)]:
        with open(out_dir / f"{split}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "en", "hi", "keywords", "labse"], delimiter="\t",
                               quoting=csv.QUOTE_NONE, escapechar="\\", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    report["final"] = {"train": len(train), "dev": len(dev),
                       "train_by_source": dict(Counter(p["source"] for p in train)),
                       "dev_by_source": dict(Counter(p["source"] for p in dev))}
    report["examples"] = [{k: p[k] for k in ("source", "en", "hi")} for p in train[:10]]
    report["flores_overlap_checked"] = not args.no_flores
    rep_path = ROOT / "results" / f"finetune_data_{cfg['name']}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    removed = sum(v for s in report["sources"].values() for k, v in s["counts"].items() if k.startswith("removed_overlap"))
    print(f"\nTotal pairs removed for test overlap: {removed}")
    print(f"Final: train {len(train)}, dev {len(dev)} {report['final']['train_by_source']}")
    print(f"Written: {out_dir}/train.tsv, dev.tsv and {rep_path}")


if __name__ == "__main__":
    main()
