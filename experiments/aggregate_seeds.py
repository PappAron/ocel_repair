"""Mean, standard deviation and 95% confidence intervals over seeds (B5).

  python aggregate_seeds.py results/seeds_p2p [--json summary.json]

Per model: mean ± SD and a t-based 95% CI over seeds for Hits@1/3/5/10 and MRR.
Also the paired difference GNN minus co-occurrence (same split and hidden links within a
seed), which is the comparison the paper's claim rests on.
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import math
import statistics
from pathlib import Path

# two-sided 95% Student t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       19: 2.093, 24: 2.064, 29: 2.045}


def t95(df):
    if df in T95:
        return T95[df]
    smaller = [d for d in T95 if d < df]
    return T95[max(smaller)] if df < 30 else 1.96


def summary(values):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else float("nan")
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--json", default=None)
    parser.add_argument("--baseline", default="co-occurrence")
    parser.add_argument("--model", default="gnn")
    args = parser.parse_args()

    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.folder).glob("seed*.json"))]
    if not runs:
        raise SystemExit("no seed*.json files found")
    metrics = ["hits@1", "hits@3", "hits@5", "hits@10", "mrr"]


    def value(run, model, metric):
        r = run["results"][model]
        return r["mrr"] if metric == "mrr" else r["hits_at_k"][metric.split("@")[1]]


    out = {"seeds": [r["seed"] for r in runs], "queries": {}, "models": {}, "paired_difference": {}}
    models = list(runs[0]["results"])
    for model in models:
        out["queries"][model] = [r["results"][model]["evaluated_links"] for r in runs]
        out["models"][model] = {m: summary([value(r, model, m) for r in runs]) for m in metrics}
    if args.model in models and args.baseline in models:
        out["paired_difference"] = {
            m: summary([value(r, args.model, m) - value(r, args.baseline, m) for r in runs]) for m in metrics
        }

    print(f"seeds: {out['seeds']}   test queries per seed: {out['queries'][models[0]]}")
    print(f"{'model':<16}" + "".join(f"{m:>22}" for m in metrics))
    rows = list(out["models"].items())
    if out["paired_difference"]:
        rows.append((f"{args.model}-{args.baseline[:6]}", out["paired_difference"]))
    for name, stats in rows:
        cells = [f"{s['mean']:.3f} [{s['ci_low']:.3f},{s['ci_high']:.3f}]" for s in (stats[m] for m in metrics)]
        print(f"{name:<16}" + "".join(f"{c:>22}" for c in cells))
    print("cells: mean over seeds [95% t-interval]")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
