"""Collect candidate tourism sentences from English Wikivoyage (plan section 4.2). No GPU needed.

Steps:
  1. download each destination page (rendered HTML) through the Wikivoyage API; pages are cached in
     data/tourism_test/raw/ and their revision ids are pinned in data/tourism_test/wikivoyage_revisions.tsv,
     so re-running gives exactly the same text (use --refresh to fetch the newest versions instead)
  2. take prose paragraphs (and, optionally, listing descriptions), skip lists of links, tables, maps
  3. split into sentences (pysbd), filter (5-40 words, clean start/end, no URLs/phone numbers/Devanagari),
     remove duplicates
  4. guess a category from the section (Eat -> food, ...) and flag numbers / likely named entities
  5. write all candidates, and a balanced shortlist (by category and destination) for manual selection

Usage:
    python src/collect_wikivoyage.py --config configs/wikivoyage.yaml

Outputs (data/tourism_test/):
    candidates.tsv   every sentence that passed the filters
    shortlist.tsv    balanced random sample + empty columns for your selection and Hindi translation
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "tourism_test"
RAW_DIR = OUT_DIR / "raw"
REVISIONS_TSV = OUT_DIR / "wikivoyage_revisions.tsv"
API = "https://en.wikivoyage.org/w/api.php"
USER_AGENT = "TourMT-Hi-research-script/0.1 (academic research; https://github.com)"

CANDIDATE_COLUMNS = [
    "id", "destination", "region", "section", "subsection", "source_type", "category_guess",
    "english", "n_words", "has_number", "has_named_entity_guess", "source_url", "revision_id",
]
SHORTLIST_EXTRA = ["keep", "category", "hindi", "minutes", "notes"]  # filled in by hand

PRICE_TIME_RE = re.compile(
    r"(₹|\bRs\.?\s?\d|\brupees?\b|\bINR\b|\bUS\$|\$\d|\b\d{1,2}(:\d{2})?\s?(am|pm|AM|PM)\b|\b\d{1,2}:\d{2}\b"
    r"|\bopen(s|ing)?\b|\bclosed\b|\bentry fee\b|\btickets?\b|\bcosts?\b|\bcharges?\b)",
)
URL_RE = re.compile(r"https?://|www\.|\.com\b|@")
PHONE_RE = re.compile(r"\+\d{2}|\d[\d\s-]{8,}\d")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


# ---------- 1. download ----------

def load_pinned_revisions() -> dict[str, int]:
    if not REVISIONS_TSV.exists():
        return {}
    with open(REVISIONS_TSV, encoding="utf-8", newline="") as f:
        return {r["destination"]: int(r["revision_id"]) for r in csv.DictReader(f, delimiter="\t")}


def fetch_page(title: str, revision_id: int | None) -> dict:
    """Rendered page from the API: by pinned revision id if given, else the current version."""
    params = {"action": "parse", "prop": "text|revid", "format": "json", "formatversion": "2", "redirects": "1"}
    if revision_id:
        params["oldid"] = str(revision_id)
    else:
        params["page"] = title
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    if "error" in data:
        raise RuntimeError(f"{title}: {data['error'].get('info')}")
    return data["parse"]


def get_pages(titles: list[str], refresh: bool) -> dict[str, dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pinned = {} if refresh else load_pinned_revisions()
    pages = {}
    for title in titles:
        rev = pinned.get(title)
        cache = RAW_DIR / f"{title.replace(' ', '_')}_{rev}.json" if rev else None
        if cache and cache.exists():
            page = json.loads(cache.read_text(encoding="utf-8"))
            how = "cache"
        else:
            page = fetch_page(title, rev)
            (RAW_DIR / f"{title.replace(' ', '_')}_{page['revid']}.json").write_text(
                json.dumps(page, ensure_ascii=False), encoding="utf-8")
            how = "downloaded"
            time.sleep(1)  # be polite to the Wikimedia servers
        print(f"  {title:<14} revision {page['revid']} ({how})")
        pages[title] = page
    with open(REVISIONS_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["destination", "revision_id", "url"])
        for title, page in pages.items():
            w.writerow([title, page["revid"], revision_url(page["revid"])])
    return pages


def revision_url(revision_id: int) -> str:
    return f"https://en.wikivoyage.org/w/index.php?oldid={revision_id}"


# ---------- 2. extract text blocks ----------

def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)  # footnote markers
    return re.sub(r"\s+", " ", text).strip()


def extract_blocks(html: str, include_listings: bool) -> list[dict]:
    """Return text blocks with their section: prose paragraphs and (optionally) listing descriptions."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select("style, script, figure, table, .mw-editsection, .listing-metadata, "
                            ".listing-sister-wikipedia, .listing-sister-wikidata, .listing-sister-osm, sup"):
        junk.decompose()
    blocks, section, subsection = [], "Intro", ""
    for el in soup.find_all(["h2", "h3", "p", "li"]):
        if el.name == "h2":
            section, subsection = el.get_text(" ", strip=True), ""
        elif el.name == "h3":
            subsection = el.get_text(" ", strip=True)
        elif el.name == "p":
            blocks.append({"section": section, "subsection": subsection, "source_type": "paragraph",
                           "text": clean_text(el.get_text(" "))})
        elif el.name == "li" and include_listings:
            content = el.select_one(".listing-content")
            if content is not None:  # structured listing: only its description text
                text = content.get_text(" ")
            else:  # plain bullet: drop the leading bold name ("<b>Name</b>: ...")
                first_bold = el.find("b")
                if first_bold is not None and el.get_text().strip().startswith(first_bold.get_text().strip()):
                    first_bold.decompose()
                text = el.get_text(" ")
            blocks.append({"section": section, "subsection": subsection, "source_type": "listing",
                           "text": clean_text(text)})
    # get_text(" ") puts spaces before punctuation around inline tags; undo that
    for b in blocks:
        b["text"] = re.sub(r"\s+([,.;:!?)])", r"\1", b["text"]).replace("( ", "(")
    return [b for b in blocks if b["text"]]


# ---------- 3. sentences + filters ----------

_SEGMENTER = None


def split_sentences(text: str) -> list[str]:
    global _SEGMENTER
    if _SEGMENTER is None:
        import pysbd

        _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in _SEGMENTER.segment(text) if s.strip()]


def balanced(s: str, pairs=("()", "[]", '""')) -> bool:
    for open_, close in pairs:
        if open_ == close:
            if s.count(open_) % 2:
                return False
        elif s.count(open_) != s.count(close):
            return False
    return True


def reject_reason(s: str, min_words: int, max_words: int) -> str | None:
    """None if the sentence is a good candidate, otherwise a short reason."""
    n = len(s.split())
    if n < min_words:
        return "too_short"
    if n > max_words:
        return "too_long"
    if not re.match(r"[A-Z0-9\"'‘“]", s):
        return "bad_start"
    if not re.search(r"[.!?][\"'’”)]?$", s):
        return "bad_end"
    if URL_RE.search(s):
        return "url_or_email"
    if PHONE_RE.search(s):
        return "phone_number"
    if DEVANAGARI_RE.search(s):
        return "devanagari"
    if not balanced(s):
        return "unbalanced_brackets"
    return None


def has_named_entity_guess(s: str) -> bool:
    """Rough guess: a capitalised word that is not the first word (you confirm by hand)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", s)
    return any(w[0].isupper() and w not in ("I",) for w in words[1:])


def categorise(section: str, sentence: str, mapping: dict[str, str]) -> str:
    if PRICE_TIME_RE.search(sentence):
        return "prices_times"
    return mapping.get(section, "other")


# ---------- 5. shortlist ----------

def _round_robin(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Pick up to k items at random, taking one destination at a time so destinations stay balanced."""
    by_dest: dict[str, list[dict]] = {}
    for c in items:
        by_dest.setdefault(c["destination"], []).append(c)
    for lst in by_dest.values():
        rng.shuffle(lst)
    dests = sorted(by_dest)
    rng.shuffle(dests)
    picked: list[dict] = []
    while len(picked) < k and any(by_dest.values()):
        for d in dests:
            if by_dest[d] and len(picked) < k:
                picked.append(by_dest[d].pop())
    return picked


def make_shortlist(cands: list[dict], size: int, seed: int, number_share: float = 0.0) -> list[dict]:
    """Equal share per category; inside each category, round-robin over destinations.

    number_share: minimum share of sentences containing a number in each category (if enough exist).
    Numbers (prices, times, distances) are kept on purpose because they are a known weak spot of MT.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for c in cands:
        if c["category_guess"] != "other":
            by_cat.setdefault(c["category_guess"], []).append(c)
    per_cat = size // max(len(by_cat), 1)
    chosen = []
    for cat in sorted(by_cat):
        with_num = [c for c in by_cat[cat] if str(c.get("has_number")) == "1"]
        picked = _round_robin(with_num, round(per_cat * number_share), rng) if number_share else []
        ids = {c["id"] for c in picked}
        rest = [c for c in by_cat[cat] if c["id"] not in ids]
        picked += _round_robin(rest, per_cat - len(picked), rng)
        chosen.extend(picked)
    return sorted(chosen, key=lambda c: c["id"])


# ---------- main ----------

def shortlist_has_manual_work(path: Path) -> bool:
    """True if any hand-filled column (keep, category, hindi, minutes, notes) is non-empty."""
    if not path.exists():
        return False
    with open(path, encoding="utf-8", newline="") as f:
        return any((row.get(col) or "").strip() for row in csv.DictReader(f, delimiter="\t") for col in SHORTLIST_EXTRA)


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "wikivoyage.yaml"))
    parser.add_argument("--refresh", action="store_true", help="fetch the newest page versions (re-pins revisions)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    region_of = {t: region for region, titles in cfg["destinations"].items() for t in titles}
    print(f"Pages ({len(region_of)} destinations):")
    pages = get_pages(list(region_of), args.refresh)

    seen, cands, rejected = set(), [], {}
    for title, page in pages.items():
        for block in extract_blocks(page["text"], cfg.get("include_listings", True)):
            for sent in split_sentences(block["text"]):
                reason = reject_reason(sent, cfg["min_words"], cfg["max_words"])
                key = re.sub(r"\W+", " ", sent.lower()).strip()
                if reason is None and key in seen:
                    reason = "duplicate"
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                seen.add(key)
                cands.append({
                    "destination": title, "region": region_of[title],
                    "section": block["section"], "subsection": block["subsection"],
                    "source_type": block["source_type"],
                    "category_guess": categorise(block["section"], sent, cfg["section_categories"]),
                    "english": sent, "n_words": len(sent.split()),
                    "has_number": int(bool(re.search(r"\d", sent))),
                    "has_named_entity_guess": int(has_named_entity_guess(sent)),
                    "source_url": revision_url(page["revid"]), "revision_id": page["revid"],
                })
    for i, c in enumerate(cands, 1):
        c["id"] = f"tour{i:05d}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_tsv(OUT_DIR / "candidates.tsv", cands, CANDIDATE_COLUMNS)
    shortlist = make_shortlist(cands, cfg["shortlist_size"], cfg["seed"], cfg.get("number_share", 0.0))
    shortlist_path = OUT_DIR / "shortlist.tsv"
    if shortlist_has_manual_work(shortlist_path):
        shortlist_path = OUT_DIR / "shortlist_new.tsv"
        print(f"\nNOTE: shortlist.tsv already contains your manual work, so it was NOT overwritten.\n"
              f"      The newly generated shortlist is in {shortlist_path.name}.")
    write_tsv(shortlist_path, shortlist, CANDIDATE_COLUMNS + SHORTLIST_EXTRA)

    print(f"\nCandidates: {len(cands)}   rejected: {sum(rejected.values())} {dict(sorted(rejected.items()))}")
    cats: dict[str, int] = {}
    for c in cands:
        cats[c["category_guess"]] = cats.get(c["category_guess"], 0) + 1
    print("By category guess:", dict(sorted(cats.items(), key=lambda kv: -kv[1])))
    sl_cats: dict[str, int] = {}
    for c in shortlist:
        sl_cats[c["category_guess"]] = sl_cats.get(c["category_guess"], 0) + 1
    print(f"Shortlist: {len(shortlist)} sentences {dict(sorted(sl_cats.items()))}")
    print(f"Written: {OUT_DIR / 'candidates.tsv'}, {OUT_DIR / 'shortlist.tsv'}")


if __name__ == "__main__":
    sys.exit(main())
