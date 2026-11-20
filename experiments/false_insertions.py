"""False insertions of the decision rule on COMPLETE events (reviewer 3: "what may go wrong").

For a finished run (seed JSON + its log, which names the saved checkpoint), rebuild the same split,
draw 200 held-out events that are neither test nor validation events and hide NOTHING in them,
add them with all their links to the scored graph, and score them with the saved model (the ID-free
model scores unseen events without retraining). Every inserted link on such an event is wrong, so
  false-insertion rate = share of complete events whose score margin (top-1 minus top-2 among
  objects not linked to the event) reaches the run's threshold for a target precision.
Sanity check: the test events are re-scored and their Hits@1 compared with the run's JSON.

Run from the repository root:
  python experiments/false_insertions.py data/ocel2-p2p.sqlite RUN_JSON [RUN_JSON ...] --out OUT.json [--slim]
RUN_JSON is e.g. experiments/results/seeds_p2p_pft/seed0.json (its .log must sit next to it).
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, json, os, random, re, sqlite3, sys
from collections import defaultdict
sys.path.insert(0, os.getcwd())
ap = argparse.ArgumentParser()
ap.add_argument("sqlite"); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--slim", action="store_true"); ap.add_argument("--clean", type=int, default=200)
args = ap.parse_args()
from adapters.loaders import SqliteLoader  # noqa: E402
if args.slim:
    SqliteLoader._extension_attributes = staticmethod(lambda c, t, i: {})
from adapters.gnn_scalable import ScalableGnnAdapter, ScalableGnnConfig  # noqa: E402
from core.domain import OcelLog  # noqa: E402
from core.gap_simulator import SampledGapSimulator  # noqa: E402

connection = sqlite3.connect(args.sqlite)
event_times = {}
for (table,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'event\\_%' ESCAPE '\\'"):
    columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    if {"ocel_id", "ocel_time"} <= columns:
        event_times.update(connection.execute(f'SELECT ocel_id, ocel_time FROM "{table}"'))
connection.close()
log = SqliteLoader().load(args.sqlite)
by_event = defaultdict(list)
for l in log.event_object_links:
    by_event[l.event_id].append(l)

out = {"sqlite": args.sqlite, "runs": []}
for run_path in args.runs:
    run = json.load(open(run_path))
    text = open(os.path.splitext(run_path)[0] + ".log", "rb").read()
    if text[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = text.decode("utf-16")
    else:
        text = text.decode("utf-8", "replace")
    checkpoint = re.findall(r"Saved best GNN checkpoint: path=(\S+)", text)[-1]
    seed = run.get("seed")
    kw = {} if seed is None else dict(split_seed=seed, sample_seed=seed, target_seed=seed)
    corr = SampledGapSimulator(sample_size=200, o2o_drop=run.get("o2o_drop", 0.0) or 0.0,
                               context_drop=run.get("context_drop", 0.0) or 0.0, **kw).introduce_gaps(log)
    train_ids = {l.event_id for l in corr.training.event_object_links}
    test_ids = {l.event_id for l in corr.removed_event_object_links}
    val_ids = {l.event_id for l in corr.validation_removed_event_object_links}
    pool = sorted(e for e, ls in by_event.items() if len(ls) >= 2 and e not in train_ids | test_ids | val_ids)
    clean = random.Random(1000 + (seed or 0)).sample(pool, min(args.clean, len(pool)))
    clean_links = tuple(l for e in clean for l in by_event[e])
    scored_log = OcelLog(corr.corrupted.events, corr.corrupted.objects,
                         corr.corrupted.event_object_links + clean_links, corr.corrupted.object_object_links)
    config = ScalableGnnConfig(epochs=run.get("epochs", 2000), top_k=10000, sage_aggregation="sum",
                               negatives_per_positive=20, sampled_softmax_temperature=0.1,
                               same_type_negative_fraction=0.5, pair_features=bool(run.get("pair_features")),
                               time_feature=bool(run.get("time_feature")), object_inputs=run.get("object_inputs", "id"),
                               checkpoint_path=checkpoint, seed=42 if seed is None else seed)
    adapter = ScalableGnnAdapter(config)
    adapter.event_times = event_times
    model, event_map, object_map, _ = adapter.load(checkpoint)
    adapter._fitted = (model, event_map, object_map, None)
    torch, F = adapter.torch, adapter.F
    if config.pair_features:
        adapter._build_pair_tables(corr.training, object_map)
    if config.time_feature:
        adapter._build_time_index(corr, object_map)

    def margins(events, graph_log):
        observed = defaultdict(set)
        for l in graph_log.event_object_links:
            observed[l.event_id].add(l.object_id)
        graph, _ = adapter._build_graph(graph_log, event_map, object_map)
        model.eval()
        with torch.no_grad():
            emb = model(graph)
            ev = F.normalize(emb["event"], dim=-1); ob = F.normalize(emb["object"], dim=-1)
            scores = model.score_all(ev[[event_map[e] for e in events]], ob)
            if config.pair_features:
                adapter._add_pair_scores(model, scores, list(events), observed, object_map)
        res = {}
        for row, e in enumerate(events):
            s = scores[row].clone()
            for o in observed[e]:
                s[object_map[o]] = -torch.inf
            top = torch.topk(s, 2)
            res[e] = (float(top.values[0] - top.values[1]), int(top.indices[0]))
        return res

    clean_m = margins(clean, scored_log)
    inv = {i: o for o, i in object_map.items()}
    truth = defaultdict(set)
    for l in corr.removed_event_object_links:
        truth[l.event_id].add(l.object_id)
    test_m = margins(sorted(truth), corr.corrupted)
    hits1 = sum(inv[clean_idx] in truth[e] for e, (_, clean_idx) in test_m.items()) / len(test_m)
    decision = run["results"]["gnn"]["decision"]["targets"]
    entry = {"run": run_path, "checkpoint": checkpoint, "clean_events": len(clean),
             "test_hits1_recomputed": hits1, "test_hits1_json": run["results"]["gnn"]["hits_at_k"]["1"],
             "targets": {}}
    for target, d in decision.items():
        thr = d["threshold"]
        if thr is None:
            continue
        fired = sum(m >= thr for m, _ in clean_m.values())
        entry["targets"][target] = {"threshold": thr, "false_insertion_rate": fired / len(clean_m),
                                    "test_coverage": d["test"]["coverage"], "test_precision": d["test"]["precision"]}
    out["runs"].append(entry)
    t9 = entry["targets"].get("0.9", {})
    print(json.dumps({"run": run_path, "hits1": [round(hits1, 3), round(entry["test_hits1_json"], 3)],
                      "false@0.9": t9.get("false_insertion_rate"), "coverage@0.9": t9.get("test_coverage")}), flush=True)
json.dump(out, open(args.out, "w"), indent=1)
