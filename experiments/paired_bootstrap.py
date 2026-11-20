"""Paired bootstrap on the same gaps (reviewer point 2.6).

Compares two methods on identical hidden links, pooled over seeds; resamples gaps within
each seed (stratified), 10,000 replicates; reports the mean difference and its 95% interval
for MRR and Hits@1/3/10, and the share of replicates with difference <= 0 (one-sided p).

Method A and B are given as  FOLDER:MODEL  (a run_seeds.py folder, MODEL = a key of
results, e.g. gnn, co-occurrence, gnn-two-stage-1hop) or  HEUR_JSON:METHOD:VIEW  (output of
graph_heuristics.py; VIEW = full / 1hop / 2hop). Gaps are matched by (seed, event, object).

  python paired_bootstrap.py results/seeds_p2p_pftd:gnn results/seeds_p2p_pftd:co-occurrence
  python paired_bootstrap.py results/seeds_p2p_pftd:gnn results/p2p_heur.json:o2o_rule:full
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, json, os, random, sys
from collections import defaultdict


def load(spec):
    parts = spec.split(":")
    rows = {}
    if parts[0].endswith(".json"):
        path, method, view = parts
        for run in json.load(open(path))["runs"]:
            for r in run["methods"][method][f"{view}_per_query"]:
                r = r if len(r) == 7 else (*r[:3], 0, 0, 0, 0)
                rows[(run["seed"], r[0], r[1])] = (r[2], r[3], r[4], r[6])  # rr, h1, h3, h10
    else:
        folder, model = parts
        for f in sorted(glob.glob(os.path.join(folder, "seed*.json"))):
            d = json.load(open(f))
            for r in d["results"][model]["per_query"]:
                rows[(d["seed"], r[0], r[1])] = (r[2], r[3], r[4], r[5])
    return rows


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) & set(b), key=str)
    if not keys:
        sys.exit("no common gaps")
    by_seed = defaultdict(list)
    for k in keys:
        by_seed[k[0]].append(tuple(x - y for x, y in zip(a[k], b[k])))
    names = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    n = len(keys)
    mean = [sum(d[i] for s in by_seed.values() for d in s) / n for i in range(4)]
    rng = random.Random(0)
    reps = []
    for _ in range(10000):
        tot, cnt = [0.0] * 4, 0
        for diffs in by_seed.values():
            for _ in range(len(diffs)):
                d = diffs[rng.randrange(len(diffs))]
                for i in range(4):
                    tot[i] += d[i]
                cnt += 1
        reps.append([t / cnt for t in tot])
    print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}\n{n} common gaps over seeds {sorted(by_seed, key=str)}")
    for i, name in enumerate(names):
        v = sorted(r[i] for r in reps)
        lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
        p = sum(x <= 0 for x in v) / len(v)
        print(f"  {name:8s} A-B = {mean[i]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(diff<=0) = {p:.4f}")


if __name__ == "__main__":
    main()
