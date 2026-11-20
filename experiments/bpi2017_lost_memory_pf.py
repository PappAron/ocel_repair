"""[pf/type+degree variant, 9 Nov 2026] The frozen main model (pairwise features, no object IDs)
on the Lost Memory benchmark. Run from anywhere; OCEL_REPAIR_DIR = the repository root.

Original docstring: The paper's GNN on the Lost Memory benchmark of Sadeghianasl et al. (2026), BPI 2017.

Same damaged logs and the same scoring as bpi2017_lost_memory_reconstruction.py
(rules and co-occurrence), so all numbers go in one table.

- Log: the authors' filtered BPI 2017 log, damaged with their Lost Memory injection
  (reimplemented exactly in bpi2017_lost_memory_reconstruction.inject).
- Graph: events + applications + offers, E2O links of the DAMAGED log only (no O2O,
  no attributes, no timestamps): the "structure only" setting.
- Training: the paper's protocol, unchanged (ocel_repair GnnModelAdapter): sampled-softmax
  link prediction on observed links; checkpoint selected on 200 validation links hidden
  at random from the damaged log (one link per event, events with >= 2 objects).
  The real erased links are never used for training or selection.
- Test: for every damaged event, rank all 42,995 offers minus those still on the event;
  expected rank under random ties; Hits@1/3/10, MRR; single- vs multi-offer applications.

Usage: python bpi2017_lost_memory_gnn.py CSV OCEL_REPAIR_DIR OUT_JSON [SEED] [EPOCHS]
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

csv_path, repo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
sys.path.insert(0, os.path.abspath(repo))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import bpi2017_lost_memory_reconstruction as R  # noqa: E402
from core.domain import CorruptionResult, Event, EventObjectLink, Object, OcelLog  # noqa: E402
from adapters.gnn_scalable import ScalableGnnAdapter, ScalableGnnConfig  # noqa: E402
from core.logging_config import configure_logging  # noqa: E402

configure_logging()
t0 = time.time()
f, keep = R.load(csv_path)
removed, sampled = R.inject(f, seed)
print(f"seed {seed}: rows {len(f)}, affected events {len(removed)}, erased offers {len(sampled)}", flush=True)

full = pd.read_csv(csv_path, usecols=["offer"])["offer"]
lst = lambda x: R.ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
all_offers = sorted({o for l in full.map(lst) for o in l})
apps = sorted({a for l in f["app"] for a in l})

clean = [set(a) | set(o) for a, o in zip(f["app"], f["off"])]
damaged = [set(s) for s in clean]
for r, o in removed.items():
    damaged[r].discard(o)

otype = lambda x: "application" if x.startswith("Application_") else "offer"
events = tuple(Event(f"e{r}", act) for r, act in enumerate(f["event_activity"]))
objects = tuple(Object(x, "application") for x in apps) + tuple(Object(x, "offer") for x in all_offers)


def links(sets):
    return tuple(EventObjectLink(f"e{r}", x, otype(x)) for r, s in enumerate(sets) for x in sorted(s))


original = OcelLog(events, objects, links(clean), ())
damaged_log = OcelLog(events, objects, links(damaged), ())

# validation: 200 random links hidden from the damaged log (paper protocol: events with >= 2 objects)
rng = random.Random(1000 + seed)
eligible = [r for r, s in enumerate(damaged) if len(s) >= 2 and r not in removed]
val_rows = rng.sample(eligible, 200)
val_links = tuple(EventObjectLink(f"e{r}", x, otype(x)) for r in val_rows for x in [rng.choice(sorted(damaged[r]))])
val_keys = {(l.event_id, l.object_id, l.qualifier) for l in val_links}
train_log = damaged_log.without_event_object_links(val_keys)
truth = tuple(EventObjectLink(f"e{r}", o, "offer") for r, o in removed.items())
corruption = CorruptionResult(
    original=original, corrupted=damaged_log, removed_event_object_links=truth,
    training=train_log, validation=train_log, validation_removed_event_object_links=val_links,
)

adapter = ScalableGnnAdapter(ScalableGnnConfig(
    epochs=epochs, negatives_per_positive=20, seed=seed, sage_aggregation="sum",
    sampled_softmax_temperature=0.1, same_type_negative_fraction=0.5,
    pair_features=True, object_inputs="type+degree", model_dir="models_bpi_lost_memory_pf"))
fit_start = time.time()
model, event_map, object_map, _ = adapter.fit(corruption)
fit_seconds = time.time() - fit_start

torch, F = adapter.torch, adapter.F
graph, _ = adapter._build_graph(damaged_log, event_map, object_map)
model.eval()
offer_idx = torch.tensor([object_map[o] for o in all_offers], device=adapter.device)
pos = {o: i for i, o in enumerate(all_offers)}
observed = {f"e{r}": set(damaged[r]) for r in removed}
rows = sorted(removed)
ranks = []
with torch.no_grad():
    emb = model(graph)
    ev = F.normalize(emb["event"], dim=-1)
    ob_all = F.normalize(emb["object"], dim=-1)
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        ids = [f"e{r}" for r in chunk]
        sc = model.score_all(ev[[event_map[i] for i in ids]], ob_all)
        adapter._add_pair_scores(model, sc, ids, observed, object_map)
        sc = sc[:, offer_idx].cpu().numpy()
        for i, r in enumerate(chunk):
            s = sc[i].astype(np.float64)
            for x in damaged[r]:
                if x in pos:
                    s[pos[x]] = -np.inf
            ranks.append(R.expected_rank(s, s[pos[removed[r]]]))

offers_per_app = Counter()
seen = set()
for a, o in zip(f["app"], f["off"]):
    for x in a:
        for y in o:
            if (x, y) not in seen:
                seen.add((x, y))
                offers_per_app[x] += 1
single = [i for i, r in enumerate(rows) if offers_per_app[f["app"].iloc[r][0]] == 1]
multi = [i for i, r in enumerate(rows) if offers_per_app[f["app"].iloc[r][0]] > 1]
res = {"model": "pf type+degree", "seed": seed, "epochs": epochs, "affected_events": len(rows), "fit_seconds": round(fit_seconds, 1),
       "gnn": {"all": R.evaluate(ranks),
               "single_offer_app": R.evaluate([ranks[i] for i in single]),
               "multi_offer_app": R.evaluate([ranks[i] for i in multi])},
       "median_rank": float(np.median(ranks)), "seconds": round(time.time() - t0, 1)}
print(json.dumps(res), flush=True)
json.dump(res, open(out_path, "w"), indent=2)
