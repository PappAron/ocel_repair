"""Time-proximity rule on the partial-gap protocol (same splits as the main runs).
Score of candidate o for damaged event e = -min |t(e) - t(e')| over the events e' of o in the
observed graph (training events + test events without hidden links). Variants:
  time      : all objects are candidates (as for every other method)
  time+type : only objects of the hidden object's type (type oracle; upper reference)
Usage (from the repository root): python experiments/time_rule.py SQLITE OUT_JSON --seeds 0 1 2 3 4 [--default-split] [--o2o-drop 1.0]
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, sys, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.getcwd())
from adapters.loaders import SqliteLoader
SqliteLoader._extension_attributes = staticmethod(lambda c, t, i: {})
from core.gap_simulator import SampledGapSimulator
from core.metrics import expected_rank_metrics
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("sqlite"); ap.add_argument("out")
ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--default-split", action="store_true")
ap.add_argument("--o2o-drop", type=float, default=0.0)
a = ap.parse_args()
log = SqliteLoader().load(a.sqlite)
import sqlite3
_c = sqlite3.connect(a.sqlite)
etime = {}
for (tbl,) in _c.execute("select 'event_'||ocel_type_map from event_map_type"):
    for i, ts in _c.execute(f'select ocel_id, ocel_time from "{tbl}"'):
        etime[i] = pd.Timestamp(ts).value / 3.6e12  # hours
assert all(e.id in etime for e in log.events), "missing timestamps"
objs = [o.id for o in log.objects]; oidx = {o: i for i, o in enumerate(objs)}
otype = np.array([o.type for o in log.objects])
runs = []
for seed in ([None] if a.default_split else a.seeds):
    kw = {} if seed is None else dict(split_seed=seed, sample_seed=seed, target_seed=seed)
    corr = SampledGapSimulator(sample_size=200, validation=True, o2o_drop=a.o2o_drop, **kw).introduce_gaps(log)
    pairs = sorted((oidx[l.object_id], etime[l.event_id]) for l in corr.corrupted.event_object_links)
    po = np.array([p[0] for p in pairs]); pt = np.array([p[1] for p in pairs])
    has = np.zeros(len(objs), bool); has[po] = True
    starts = np.searchsorted(po, np.arange(len(objs)))
    observed = defaultdict(set)
    for l in corr.corrupted.event_object_links: observed[l.event_id].add(oidx[l.object_id])
    truth = defaultdict(set)
    for l in corr.removed_event_object_links: truth[l.event_id].add(oidx[l.object_id])
    res = {"time": [], "time+type": []}
    for e, targets in truth.items():
        d = np.abs(pt - etime[e])
        dist = np.full(len(objs), np.inf)
        idx = np.where(has)[0]
        dist[idx] = np.minimum.reduceat(d, starts[idx])
        score = -dist
        for t in targets:
            for name in res:
                mask = np.ones(len(objs), bool)
                mask[list(observed[e])] = False
                mask[[x for x in targets if x != t]] = False
                if name == "time+type": mask &= otype == otype[t]
                s = score[mask]; st = score[t]
                better = int(np.sum(s > st)); ties = int(np.sum(s == st)) 
                rr, h = expected_rank_metrics(better, ties, (1, 3, 10))
                res[name].append((rr, h[1], h[3], h[10]))
    summ = {k: {"n": len(v), "mrr": float(np.mean([r[0] for r in v])), "hits@1": float(np.mean([r[1] for r in v])),
                "hits@10": float(np.mean([r[3] for r in v]))} for k, v in res.items()}
    print(json.dumps({"seed": seed, **summ})); runs.append({"seed": seed, "summary": summ})
json.dump({"sqlite": a.sqlite, "o2o_drop": a.o2o_drop, "runs": runs}, open(a.out, "w"), indent=1)
