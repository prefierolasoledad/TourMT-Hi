"""Is a difference between two systems real, or could it be luck? Paired bootstrap resampling (plan 6.1).

Usage (CPU is enough: it reuses the hyp/ref files and the COMET segment scores already computed):
    python src/significance.py --baseline outputs/it2-200m_flores-devtest_en-hi \\
        --systems outputs/lora-a-s42_flores-devtest_en-hi outputs/lora-a-s43_flores-devtest_en-hi

How it works (Koehn 2004): the test set is one sample of all possible sentences. We draw 1,000 new
"test sets" of the same size from it, with replacement, and compute system - baseline on each, using the
SAME sentences for both systems (paired). If the difference keeps its sign in nearly all of them, it is
unlikely to be luck.
    delta    system - baseline on the real test set
    95% CI   range containing the middle 95% of the resampled deltas (if it includes 0: not significant)
    p        two-sided: 2 x the share of resampled deltas on the other side of 0 (capped at 1); with 1,000
             resamples the smallest p that can be shown is < 0.002 (stored as 0 in the csv)

Metrics: chrF++ and BLEU are computed by sacrebleu from its own per-sentence statistics (same numbers as
evaluate.py - no custom metric code); COMET, CometKiwi and the Maithili "labelled Hindi" rate are averages
of per-sentence values from segment_scores.tsv. Same normalisation as evaluate.py.

Writes/updates results/significance.csv (one row per baseline, system, metric).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_direction  # noqa: E402
from evaluate import load_run, make_normalizer, read_lines  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["baseline", "system", "metric", "n_sentences", "baseline_score", "system_score", "delta",
           "ci_low", "ci_high", "p_value", "significant", "n_samples", "seed", "computed_at"]


def p_value_two_sided(deltas: np.ndarray, observed: float) -> float:
    """Share of resampled deltas on the other side of zero than the observed delta, times two."""
    other_side = np.mean(deltas <= 0) if observed > 0 else np.mean(deltas >= 0)
    return float(min(1.0, 2 * other_side))


def bootstrap_indices(n: int, n_samples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_samples, n))


def bootstrap_mean_diff(base: np.ndarray, system: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """For metrics that are averages of sentence scores (COMET, rates)."""
    return system[idx].mean(axis=1) - base[idx].mean(axis=1)


def bootstrap_corpus_metric(metric, base_hyps, sys_hyps, refs, idx: np.ndarray) -> tuple[float, float, np.ndarray]:
    """For corpus metrics (chrF++, BLEU): sum sacrebleu's per-sentence statistics over each resample and let
    sacrebleu compute the score from the sums (this is how sacrebleu's own significance test does it)."""
    stats_b = np.array(metric._extract_corpus_statistics(base_hyps, [refs]))
    stats_s = np.array(metric._extract_corpus_statistics(sys_hyps, [refs]))
    score = lambda stats: metric._compute_score_from_stats(list(stats)).score  # noqa: E731
    observed_b, observed_s = score(stats_b.sum(0)), score(stats_s.sum(0))
    deltas = np.array([score(stats_s[i].sum(0)) - score(stats_b[i].sum(0)) for i in idx])
    return observed_b, observed_s, deltas


def read_segments(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "segment_scores.tsv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for col in ("COMET", "CometKiwi"):
        if rows and col in rows[0]:
            out[col] = np.array([float(r[col]) for r in rows])
    if rows and "lid_label" in rows[0]:
        out["hindi_rate"] = np.array([float("hin_" in r["lid_label"]) for r in rows])
    return out


def compare(base_dir: Path, sys_dir: Path, n_samples: int, seed: int) -> list[dict]:
    base, system = load_run(base_dir), load_run(sys_dir)
    for d in (base_dir, sys_dir):
        if not (d / "ids.txt").exists():
            raise FileNotFoundError(f"{d}/ids.txt missing")
    if read_lines(base_dir / "ids.txt") != read_lines(sys_dir / "ids.txt") or base["ref"] != system["ref"]:
        raise ValueError(f"{base_dir.name} and {sys_dir.name} are not on the same test sentences")
    if base["cfg"]["direction"] != system["cfg"]["direction"]:
        raise ValueError("different translation directions")
    _, tgt = parse_direction(base["cfg"]["direction"])
    norm, _ = make_normalizer(tgt, True)
    refs = [norm(r) for r in base["ref"]]
    hb, hs = [norm(h) for h in base["hyp"]], [norm(h) for h in system["hyp"]]
    n = len(refs)
    idx = bootstrap_indices(n, n_samples, seed)

    from sacrebleu.metrics import BLEU, CHRF

    results = []
    for name, metric in [("chrF++", CHRF(word_order=2)), ("BLEU", BLEU())]:
        sb, ss, deltas = bootstrap_corpus_metric(metric, hb, hs, refs, idx)
        results.append((name, sb, ss, deltas, 2))
    seg_b, seg_s = read_segments(base_dir), read_segments(sys_dir)
    for name in [k for k in ("COMET", "CometKiwi", "hindi_rate") if k in seg_b and k in seg_s]:
        b, s = seg_b[name], seg_s[name]
        results.append((name, b.mean(), s.mean(), bootstrap_mean_diff(b, s, idx), 4))

    rows = []
    for name, sb, ss, deltas, digits in results:
        observed = ss - sb
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = p_value_two_sided(deltas, observed) if observed != 0 else 1.0
        rows.append({
            "baseline": base_dir.name, "system": sys_dir.name, "metric": name, "n_sentences": n,
            "baseline_score": round(sb, digits), "system_score": round(ss, digits),
            "delta": round(observed, digits), "ci_low": round(lo, digits), "ci_high": round(hi, digits),
            "p_value": round(p, 4), "significant": p < 0.05, "n_samples": n_samples, "seed": seed,
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return rows


def upsert(csv_path: Path, new_rows: list[dict]) -> None:
    keys = {(r["baseline"], r["system"], r["metric"]) for r in new_rows}
    rows = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if (r["baseline"], r["system"], r["metric"]) not in keys]
    rows += new_rows
    rows.sort(key=lambda r: (r["baseline"], r["system"], r["metric"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="run folder of the baseline")
    parser.add_argument("--systems", nargs="+", required=True, help="run folders to compare with it")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345, help="seed of the resampling (fixed: reproducible)")
    parser.add_argument("--results_csv", default=str(ROOT / "results" / "significance.csv"))
    args = parser.parse_args()

    all_rows = []
    for sys_dir in args.systems:
        rows = compare(Path(args.baseline), Path(sys_dir), args.n_samples, args.seed)
        all_rows += rows
        print(f"\n{Path(sys_dir).name}  vs  {Path(args.baseline).name}  ({rows[0]['n_sentences']} sentences, "
              f"{args.n_samples} resamples)")
        print(f"  {'metric':<11}{'baseline':>10}{'system':>10}{'delta':>9}   {'95% CI':<20}{'p':>7}")
        for r in rows:
            ci = f"[{r['ci_low']}, {r['ci_high']}]"
            p_text = f"<{2 / args.n_samples:g}" if r["p_value"] == 0 else f"{r['p_value']:g}"
            star = "  *" if r["significant"] else ""
            print(f"  {r['metric']:<11}{r['baseline_score']:>10}{r['system_score']:>10}{r['delta']:>9}   "
                  f"{ci:<20}{p_text:>7}{star}")
    if len(args.systems) > 1:  # several seeds of one system: mean and spread of the delta
        print("\nAcross systems (e.g. seeds): delta mean +- std")
        for metric in dict.fromkeys(r["metric"] for r in all_rows):
            d = np.array([r["delta"] for r in all_rows if r["metric"] == metric])
            sig = sum(r["significant"] for r in all_rows if r["metric"] == metric)
            print(f"  {metric:<11}{d.mean():+.4f} +- {d.std(ddof=1):.4f}   significant in {sig}/{len(d)}")
    upsert(Path(args.results_csv), all_rows)
    print(f"\n* = p < 0.05. Saved to {args.results_csv}")


if __name__ == "__main__":
    main()
