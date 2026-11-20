"""Non-learned reference methods for the revision (reviewer points 2.2 / 2.3), all on the same
gaps as the GNN runs (SampledGapSimulator, same seeds), with full ranking over all objects and
restricted to the same k-hop candidates as the two-stage GNN runs (khop_candidates logic of
the repository root/adapters/gnn_scalable.py, re-implemented here without torch).

Methods
  cooc        co-occurrence baseline of the paper: sum_c log(P(o|c) + 1e-9), counts from training events
  cooc_sum    smoothed co-occurrence: sum_c P(o|c)
  o2o_rule    number of observed objects O2O-linked to o, ties broken by cooc
  adamic_adar Adamic-Adar on the object graph (two objects adjacent if they share an event of the
              observed graph or are O2O-linked): sum_c sum_{z in N(c) & N(o)} 1/log|N(z)|
  ppr         personalized PageRank from the event node on the observed event-object graph
              (E2O + O2O edges, undirected), restart 0.15, 30 iterations
  random      (two-stage only) every candidate scores the same
Metrics: filtered, expected rank under random ties (as core/metrics.py); per-query rows saved for
paired statistics.

Usage (from inside ocel_repair or the repository root):
  python graph_heuristics.py SQLITE OUT_JSON --seeds 0 1 2 3 4 [--o2o-drop 1.0] [--default-split]
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, math, os, sys, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.getcwd())
from adapters.loaders import SqliteLoader  # noqa: E402
SqliteLoader._extension_attributes = staticmethod(lambda connection, table, identifier: {})
from core.gap_simulator import SampledGapSimulator  # noqa: E402
from core.metrics import expected_rank_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sqlite"); ap.add_argument("out")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--default-split", action="store_true", help="no --seed: the paper's default split")
ap.add_argument("--o2o-drop", type=float, default=0.0)
ap.add_argument("--hops", type=int, nargs="*", default=[1, 2])
ap.add_argument("--methods", nargs="*", default=["cooc", "cooc_sum", "o2o_rule", "adamic_adar", "ppr"])
args = ap.parse_args()

log = SqliteLoader().load(args.sqlite)
EPS = 1e-9


def khop(context_log, hops):
    eo = defaultdict(set)
    for l in context_log.event_object_links:
        eo[l.event_id].add(l.object_id)
    nb = defaultdict(set)
    for objs in eo.values():
        for o in objs:
            nb[o] |= objs
    for l in context_log.object_object_links:
        nb[l.source_id].add(l.target_id); nb[l.target_id].add(l.source_id)
    cache = {}
    def reach(o):
        if o not in cache:
            seen, frontier = {o}, {o}
            for _ in range(hops):
                frontier = {n for f in frontier for n in nb[f]} - seen
                seen |= frontier
            cache[o] = seen
        return cache[o]
    return lambda e: set().union(*(reach(o) for o in eo[e])) - eo[e] if eo[e] else set()


def evaluate(scores, default, truth, observed, universe, cand=None):
    """scores: {event: {obj: score}}; objects without explicit score get `default`."""
    rows = []
    for e in sorted(truth):
        targets = truth[e]
        s = scores.get(e, {})
        pool = (cand(e) if cand else universe) - observed[e]
        for t in sorted(targets):
            if t not in pool:
                rows.append((e, t, 0.0, 0.0, 0.0, 0.0)); continue
            p = pool - (targets - {t})
            ts = s.get(t, default)
            explicit = [v for o, v in s.items() if o in p]
            n_implicit = len(p) - len(explicit)
            vals = np.asarray(explicit)
            better = int((vals > ts).sum()) + (n_implicit if default > ts else 0)
            ties = int((vals == ts).sum()) + (n_implicit if default == ts else 0)
            rr, h = expected_rank_metrics(better, max(ties, 1), (1, 3, 5, 10))
            rows.append((e, t, rr, h[1], h[3], h[5], h[10]))
    return rows


def summary(rows):
    n = len(rows)
    rows = [r if len(r) == 7 else (*r[:3], 0.0, 0.0, 0.0, 0.0) for r in rows]
    return {"n": n, "hits@1": sum(r[3] for r in rows) / n, "hits@3": sum(r[4] for r in rows) / n,
            "hits@5": sum(r[5] for r in rows) / n, "hits@10": sum(r[6] for r in rows) / n,
            "mrr": sum(r[2] for r in rows) / n}


out = {"sqlite": args.sqlite, "o2o_drop": args.o2o_drop, "runs": []}
seeds = [None] if args.default_split else args.seeds
for seed in seeds:
    t0 = time.time()
    kw = {} if seed is None else dict(split_seed=seed, sample_seed=seed, target_seed=seed)
    c = SampledGapSimulator(sample_size=200, o2o_drop=args.o2o_drop, **kw).introduce_gaps(log)
    ctx = c.corrupted
    truth = defaultdict(set)
    for l in c.removed_event_object_links:
        truth[l.event_id].add(l.object_id)
    observed = defaultdict(set)
    for l in ctx.event_object_links:
        observed[l.event_id].add(l.object_id)
    universe = set(c.candidate_object_ids)
    # co-occurrence statistics from training events
    tr = defaultdict(set)
    for l in c.training.event_object_links:
        tr[l.event_id].add(l.object_id)
    occ = defaultdict(int); co = defaultdict(lambda: defaultdict(int))
    for objs in tr.values():
        for a in objs:
            occ[a] += 1
            for b in objs:
                if b != a: co[a][b] += 1
    o2o = defaultdict(set)
    for l in ctx.object_object_links:
        o2o[l.source_id].add(l.target_id); o2o[l.target_id].add(l.source_id)
    # object adjacency for Adamic-Adar (observed graph)
    eo_all = defaultdict(set)
    for l in ctx.event_object_links:
        eo_all[l.event_id].add(l.object_id)
    nb = defaultdict(set)
    for objs in eo_all.values():
        for o in objs: nb[o] |= objs - {o}
    for a, bs in o2o.items():
        nb[a] |= bs
    scores, defaults = {}, {}
    events = sorted(truth)
    if "cooc" in args.methods or "o2o_rule" in args.methods:
        S = {}
        for e in events:
            C = observed[e]; m = len(C); acc = defaultdict(float)
            for cobj in C:
                for o, k in co[cobj].items():
                    acc[o] += math.log(k / occ[cobj] + EPS) - math.log(EPS)
            S[e] = {o: v + m * math.log(EPS) for o, v in acc.items()}
        scores["cooc"] = S; defaults["cooc"] = None  # per-event default set below
    if "cooc_sum" in args.methods:
        scores["cooc_sum"] = {e: _ for e, _ in ((e, {}) for e in events)}
        for e in events:
            acc = defaultdict(float)
            for cobj in observed[e]:
                for o, k in co[cobj].items(): acc[o] += k / occ[cobj]
            scores["cooc_sum"][e] = dict(acc)
        defaults["cooc_sum"] = 0.0
    if "o2o_rule" in args.methods:
        R = {}
        for e in events:
            C = observed[e]; m = len(C); base = scores["cooc"][e]
            cnt = defaultdict(int)
            for cobj in C:
                for o in o2o[cobj]: cnt[o] += 1
            keys = set(base) | set(cnt)
            R[e] = {o: 1000.0 * cnt.get(o, 0) + base.get(o, m * math.log(EPS)) for o in keys}
        scores["o2o_rule"] = R
    if "adamic_adar" in args.methods:
        A = {}
        for e in events:
            acc = defaultdict(float)
            for cobj in observed[e]:
                for z in nb[cobj]:
                    w = 1.0 / math.log(max(len(nb[z]), 2))
                    for o in nb[z]: acc[o] += w
            A[e] = dict(acc)
        scores["adamic_adar"] = A; defaults["adamic_adar"] = 0.0
    if "ppr" in args.methods:
        nodes = {}
        for e in eo_all: nodes.setdefault(("e", e), len(nodes))
        for o in universe: nodes.setdefault(("o", o), len(nodes))
        rows_, cols_ = [], []
        for e, objs in eo_all.items():
            for o in objs:
                if ("o", o) in nodes:
                    i, j = nodes[("e", e)], nodes[("o", o)]; rows_ += [i, j]; cols_ += [j, i]
        for a, bs in o2o.items():
            for b in bs:
                if ("o", a) in nodes and ("o", b) in nodes:
                    rows_.append(nodes[("o", a)]); cols_.append(nodes[("o", b)])
        N = len(nodes)
        Adj = sp.csr_matrix((np.ones(len(rows_), dtype=np.float32), (rows_, cols_)), shape=(N, N))
        deg = np.asarray(Adj.sum(1)).ravel(); deg[deg == 0] = 1
        P = sp.diags(1 / deg) @ Adj  # row-stochastic
        PT = P.T.tocsr()
        idx = [nodes[("e", e)] for e in events]
        Sm = np.zeros((N, len(events)), dtype=np.float32); Sm[idx, range(len(events))] = 1
        r = Sm.copy()
        for _ in range(30):
            r = 0.15 * Sm + 0.85 * (PT @ r)
        obj_items = [(o, nodes[("o", o)]) for o in universe]
        oi = np.array([i for _, i in obj_items]); on = [o for o, _ in obj_items]
        PR = {}
        for k, e in enumerate(events):
            col = r[oi, k]; nz = np.nonzero(col > 0)[0]
            PR[e] = {on[i]: float(col[i]) for i in nz}
        scores["ppr"] = PR; defaults["ppr"] = 0.0
    run = {"seed": seed, "queries": sum(len(t) for t in truth.values()), "methods": {}}
    cands = {h: khop(ctx, h) for h in args.hops}
    for name in [m for m in args.methods] + ["random"]:
        if name == "random":
            S, dflt = {e: {} for e in events}, 0.0
        else:
            S = scores[name]; dflt = defaults.get(name)
        def ev(cand):
            if dflt is None:  # cooc / o2o_rule: default = m * log(eps) per event
                rows = []
                for e in events:
                    m = len(observed[e])
                    rows += evaluate({e: S[e]}, m * math.log(EPS), {e: truth[e]}, observed, universe, cand)
                return rows
            return evaluate(S, dflt, truth, observed, universe, cand)
        res = {}
        if name != "random":
            rows = ev(None); res["full"] = summary(rows); res["full_per_query"] = rows
        for h in args.hops:
            rows = ev(cands[h]); res[f"{h}hop"] = summary(rows); res[f"{h}hop_per_query"] = rows
        run["methods"][name] = res
    rec = {h: sum(len(truth[e] & cands[h](e)) for e in events) / run["queries"] for h in args.hops}
    size = {h: float(np.mean([len(cands[h](e)) for e in events])) for h in args.hops}
    run["candidates"] = {"recall": rec, "mean_size": size}
    run["seconds"] = time.time() - t0
    out["runs"].append(run)
    print(f"seed {seed}: recall {rec} size {size} ({run['seconds']:.0f}s)", flush=True)
    for name, res in run["methods"].items():
        print("  %-12s " % name + "  ".join(f"{k}: H1 {v['hits@1']:.3f} H10 {v['hits@10']:.3f} MRR {v['mrr']:.3f}"
                                           for k, v in res.items() if not k.endswith("per_query")), flush=True)
json.dump(out, open(args.out, "w"))
