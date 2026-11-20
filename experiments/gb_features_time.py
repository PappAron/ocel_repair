"""Feature-only ranker with gradient boosting (reviewer question after Galanti & de Leoni 2023):
does a strong learner on the same pairwise features as our model close the gap to the GNN?

LightGBM LambdaRank on the five pairwise features of core/adapters (gnn.py _pair_vector):
training queries = (training event, one of its objects) with leave-one-out counts, ranked against
all candidates with non-zero features (capped) plus random zero-feature objects; rounds chosen by
validation MRR (early stopping); test = full ranking over all objects (filtered, expected rank).

  python gb_features.py SQLITE OUT_JSON --seeds 0 1 2 3 4 [--o2o-drop 1.0] [--default-split]
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, math, os, random, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.getcwd())
from adapters.loaders import SqliteLoader  # noqa
SqliteLoader._extension_attributes = staticmethod(lambda connection, table, identifier: {})
from adapters.gnn import GnnConfig, GnnModelAdapter, _pair_vector, PAIR_FEATURES  # noqa
from core.gap_simulator import SampledGapSimulator  # noqa
from core.metrics import expected_rank_metrics  # noqa
import lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument("sqlite"); ap.add_argument("out")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--default-split", action="store_true")
ap.add_argument("--o2o-drop", type=float, default=0.0)
ap.add_argument("--train-queries", type=int, default=20000)
ap.add_argument("--cap", type=int, default=300)
ap.add_argument("--time", action="store_true", help="add the temporal feature phi_6 (dense)")
ap.add_argument("--random-negatives", type=int, default=300, help="(--time) random objects per training/validation query")
args = ap.parse_args()
log = SqliteLoader().load(args.sqlite)
D = PAIR_FEATURES + (1 if args.time else 0)
EVENT_TIMES = {}
if args.time:
    import sqlite3
    _c = sqlite3.connect(args.sqlite)
    for (tbl,) in _c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'event\\_%' ESCAPE '\\'"):
        cols = {r[1] for r in _c.execute(f'PRAGMA table_info("{tbl}")')}
        if {"ocel_id", "ocel_time"} <= cols:
            EVENT_TIMES.update(_c.execute(f'SELECT ocel_id, ocel_time FROM "{tbl}"'))
ADAPTER = None


def sparse_features(tables, context, leave_out=None):
    """{candidate: feature vector} for all candidates with a non-zero feature."""
    co, occ, nb = tables
    m = len(context)
    acc = {}
    for c in context:
        n = occ[c] - (1 if leave_out is not None else 0)
        for cand, k in co[c].items():
            if leave_out is not None and cand == leave_out:
                k -= 1
            if k <= 0:
                continue
            a = acc.setdefault(cand, [0, 0, 0.0, 0]); a[0] += k; a[1] += 1; a[2] += k / n if n > 0 else 0.0
        for cand in nb[c]:
            acc.setdefault(cand, [0, 0, 0.0, 0])[3] += 1
    return {cand: _pair_vector(a[0], a[1], a[2], a[3], m) for cand, a in acc.items()}


def build_queries(events_targets, observed, tables, n_objects, rng, cap, loo):
    X, y, groups = [], [], []
    for e, t in events_targets:
        ctx = [o for o in observed[e] if o != t]
        if not ctx:
            continue
        feats = sparse_features(tables, ctx, leave_out=t if loo else None)
        feats.pop(None, None)
        for o in ctx:
            feats.pop(o, None)
        pos = feats.pop(t, [0.0] * PAIR_FEATURES)
        cands = list(feats.items())
        if len(cands) > cap:
            cands = rng.sample(cands, cap)
        if not args.time:
            zeros = 10
            rows = [pos] + [v for _, v in cands] + [[0.0] * PAIR_FEATURES] * zeros
        else:
            excl = {t, *ctx, *(k for k, _ in cands)}
            rand = [o for o in rng.sample(range(n_objects), args.random_negatives + 10) if o not in excl][:args.random_negatives]
            ids = [t] + [k for k, _ in cands] + rand
            base = [pos] + [v for _, v in cands] + [[0.0] * PAIR_FEATURES] * len(rand)
            tf = ADAPTER._time_features_one(e, ids)
            rows = [b + list(x) for b, x in zip(base, tf)]
        X += rows; y += [1] + [0] * (len(rows) - 1); groups.append(len(rows))
    return np.asarray(X, dtype=np.float32), np.asarray(y), groups


def evaluate_dense(model, truth, observed, tables, universe_size, idx):
    rows = []
    for e in sorted(truth):
        targets = truth[e]
        ctx = [idx[o] for o in observed[e] if o in idx]
        X = np.zeros((universe_size, D), dtype=np.float32)
        X[:, PAIR_FEATURES:] = ADAPTER._time_features_all(e)
        feats = sparse_features(tables, ctx) if ctx else {}
        if feats:
            k = np.fromiter(feats, dtype=np.int64)
            X[k, :PAIR_FEATURES] = np.asarray([feats[i] for i in feats], dtype=np.float32)
        s = model.predict(X)
        s[list({idx[o] for o in observed[e] if o in idx})] = -np.inf
        for t in sorted(targets):
            ti = idx[t]
            f = s.copy()
            for o in targets - {t}:
                f[idx[o]] = -np.inf
            better = int((f > f[ti]).sum()); ties = int((f == f[ti]).sum())
            rr, h = expected_rank_metrics(better, ties, (1, 3, 5, 10))
            rows.append((e, t, rr, h[1], h[3], h[5], h[10]))
    n = len(rows)
    return rows, {"n": n, "mrr": sum(r[2] for r in rows) / n, "hits@1": sum(r[3] for r in rows) / n,
                  "hits@3": sum(r[4] for r in rows) / n, "hits@10": sum(r[6] for r in rows) / n}


def evaluate(model, truth, observed, tables, universe_size, idx):
    if args.time:
        return evaluate_dense(model, truth, observed, tables, universe_size, idx)
    zero = float(model.predict(np.zeros((1, PAIR_FEATURES), dtype=np.float32))[0])
    rows = []
    for e in sorted(truth):
        targets = truth[e]
        ctx = [idx[o] for o in observed[e] if o in idx]
        feats = sparse_features(tables, ctx) if ctx else {}
        obs = {idx[o] for o in observed[e] if o in idx}
        for o in obs:
            feats.pop(o, None)
        keys = list(feats)
        sc = dict(zip(keys, model.predict(np.asarray([feats[k] for k in keys], dtype=np.float32)))) if keys else {}
        pool_size = universe_size - len(obs)
        for t in sorted(targets):
            ti = idx.get(t)
            others = {idx[o] for o in targets - {t} if o in idx}
            ts = sc.get(ti, zero)
            vals = np.asarray([v for k, v in sc.items() if k not in others])
            n_imp = (pool_size - len(others)) - len(vals)
            better = int((vals > ts).sum()) + (n_imp if zero > ts else 0)
            ties = int((vals == ts).sum()) + (n_imp if zero == ts else 0)
            rr, h = expected_rank_metrics(better, max(ties, 1), (1, 3, 5, 10))
            rows.append((e, t, rr, h[1], h[3], h[5], h[10]))
    n = len(rows)
    return rows, {"n": n, "mrr": sum(r[2] for r in rows) / n, "hits@1": sum(r[3] for r in rows) / n,
                  "hits@3": sum(r[4] for r in rows) / n, "hits@10": sum(r[6] for r in rows) / n}


out = {"sqlite": args.sqlite, "o2o_drop": args.o2o_drop, "runs": []}
for seed in ([None] if args.default_split else args.seeds):
    t0 = time.time()
    kw = {} if seed is None else dict(split_seed=seed, sample_seed=seed, target_seed=seed)
    c = SampledGapSimulator(sample_size=200, o2o_drop=args.o2o_drop, **kw).introduce_gaps(log)
    objs = [o for o in c.training.objects if o.id in c.candidate_object_ids]
    idx = {o.id: i for i, o in enumerate(objs)}
    adapter = GnnModelAdapter(GnnConfig(pair_features=True, time_feature=args.time))
    adapter._build_pair_tables(c.training, idx)
    if args.time:
        adapter.event_times = EVENT_TIMES
        adapter._build_time_index(c, idx)
        ADAPTER = adapter
    tables = adapter._pair_tables
    rng = random.Random(0 if seed is None else seed)
    tr_obs = defaultdict(set)
    for l in c.training.event_object_links:
        if l.object_id in idx: tr_obs[l.event_id].add(idx[l.object_id])
    pairs = [(e, o) for e, s in tr_obs.items() if len(s) >= 2 for o in s]
    pairs = rng.sample(pairs, min(args.train_queries, len(pairs)))
    Xtr, ytr, gtr = build_queries(pairs, tr_obs, tables, len(idx), rng, args.cap, loo=True)
    # validation: hidden validation links, context = validation log (no leave-one-out)
    val_obs = defaultdict(set)
    for l in c.validation.event_object_links:
        if l.object_id in idx: val_obs[l.event_id].add(idx[l.object_id])
    vpairs = [(l.event_id, idx[l.object_id]) for l in c.validation_removed_event_object_links if l.object_id in idx]
    for e, t in vpairs: val_obs[e].add(t)  # context excludes t inside build_queries
    Xva, yva, gva = build_queries(vpairs, val_obs, tables, len(idx), rng, 10**9, loo=False)
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, verbose=-1, random_state=0)
    # with the dense temporal feature, ndcg@1 on 200 validation queries plateaus after one tree
    # (a one-tree model gives massive ties in the full ranking); stop on ndcg@10 with more patience
    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
              eval_at=[10] if args.time else [1, 10],
              callbacks=[lgb.early_stopping(100 if args.time else 50, verbose=False)])
    truth = defaultdict(set)
    for l in c.removed_event_object_links: truth[l.event_id].add(l.object_id)
    te_obs = defaultdict(set)
    for l in c.corrupted.event_object_links: te_obs[l.event_id].add(l.object_id)
    rows, summ = evaluate(model, truth, te_obs, tables, len(idx), idx)
    summ.update(seed=seed, best_iteration=int(model.best_iteration_ or 0), train_queries=len(gtr), seconds=round(time.time() - t0))
    out["runs"].append({"seed": seed, "summary": summ, "per_query": rows})
    print(json.dumps(summ), flush=True)
json.dump(out, open(args.out, "w"))
