"""Inductive transfer test (8 Nov 2026).

The log is cut into two halves A and B along its connected components (links E2O + O2O),
so the two halves share no event and no object. A model is trained on A and then used
on B without any retraining: every object of B is new to it. A model with per-object
ID vectors cannot do this (B's objects have no trained vector); the ID-free models can.

Reported on B's test queries:
  transfer   model trained on A, applied to B (no training on B)
  retrain    the same model type trained on B itself (in-domain reference)
  co-occurrence and random-among-2-hop-candidates references

  python inductive_transfer.py data/ocel2-p2p.sqlite --seed 0 --epochs 600 --pair-features \
      --object-inputs type+degree --out results/scaling_experiments/transfer_pf_typedeg_seed0.json
"""
from __future__ import annotations
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import random
import time
from pathlib import Path

from adapters.gnn_scalable import (
    CandidateFilteredPredictor, ScalableGnnAdapter, ScalableGnnConfig,
)
from adapters.loaders import SqliteLoader
from adapters.predictors import CoOccurrenceBaseline
from core.domain import OcelLog
from core.gap_simulator import SampledGapSimulator
from core.metrics import evaluate_rankings


def split_by_components(log: OcelLog, seed: int):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in log.events:
        find("e:" + e.id)
    for o in log.objects:
        find("o:" + o.id)
    for l in log.event_object_links:
        union("e:" + l.event_id, "o:" + l.object_id)
    for l in log.object_object_links:
        union("o:" + l.source_id, "o:" + l.target_id)
    roots = sorted({find(x) for x in parent})
    random.Random(seed).shuffle(roots)
    # assign components to A until it holds half of the events
    size = {}
    for e in log.events:
        size[find("e:" + e.id)] = size.get(find("e:" + e.id), 0) + 1
    half, in_a, count = len(log.events) / 2, set(), 0
    for r in roots:
        if count < half:
            in_a.add(r)
            count += size.get(r, 0)

    def part(keep):
        ev = {e.id for e in log.events if (find("e:" + e.id) in in_a) == keep}
        ob = {o.id for o in log.objects if (find("o:" + o.id) in in_a) == keep}
        return OcelLog(
            tuple(e for e in log.events if e.id in ev),
            tuple(o for o in log.objects if o.id in ob),
            tuple(l for l in log.event_object_links if l.event_id in ev),
            tuple(l for l in log.object_object_links if l.source_id in ob),
        )

    return part(True), part(False), len(roots)


def metrics(result):
    return {"hits_at_k": {str(k): v for k, v in result.hits_at_k.items()}, "mrr": result.mrr,
            "evaluated_links": result.evaluated_links}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--object-inputs", choices=["type", "type+degree"], default="type+degree")
    parser.add_argument("--pair-features", action="store_true")
    parser.add_argument("--time-feature", action="store_true")
    parser.add_argument("--no-retrain", action="store_true", help="skip the in-domain reference on B")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log = SqliteLoader().load(args.source_path)
    log_a, log_b, components = split_by_components(log, args.seed)
    print(f"components={components} A: events={len(log_a.events)} objects={len(log_a.objects)} | "
          f"B: events={len(log_b.events)} objects={len(log_b.objects)}", flush=True)
    sim = lambda: SampledGapSimulator(sample_size=200, split_seed=args.seed,
                                      sample_seed=args.seed, target_seed=args.seed)
    corr_a, corr_b = sim().introduce_gaps(log_a), sim().introduce_gaps(log_b)
    config = ScalableGnnConfig(
        epochs=args.epochs, top_k=10000, sage_aggregation="sum", negatives_per_positive=20,
        sampled_softmax_temperature=0.1, same_type_negative_fraction=0.5,
        validation_interval=50, seed=args.seed, pair_features=args.pair_features,
        object_inputs=args.object_inputs, model_dir="models/transfer",
        time_feature=args.time_feature,
    )
    out = {"source": args.source_path, "seed": args.seed, "epochs": args.epochs,
           "object_inputs": args.object_inputs, "pair_features": args.pair_features,
           "time_feature": args.time_feature,
           "components": components,
           "A": {"events": len(log_a.events), "objects": len(log_a.objects)},
           "B": {"events": len(log_b.events), "objects": len(log_b.objects)}}

    # 1. train on A (validation on A), score A's own test queries
    adapter = ScalableGnnAdapter(config)
    if args.time_feature:
        import sqlite3
        connection = sqlite3.connect(args.source_path)
        event_times = {}
        for (table,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'event\\_%' ESCAPE '\\'"):
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            if {"ocel_id", "ocel_time"} <= columns:
                event_times.update(connection.execute(f'SELECT ocel_id, ocel_time FROM "{table}"'))
        connection.close()
        adapter.event_times = event_times
    started = time.perf_counter()
    out["A_in_domain"] = metrics(adapter.evaluate(corr_a, adapter.predict(corr_a)))
    out["train_seconds_A"] = time.perf_counter() - started
    print("A in-domain", out["A_in_domain"], flush=True)

    # 2. apply the A model to B: new maps, B's own type indices and degree features,
    #    pairwise features counted on B's observed links. No parameter is updated.
    model = adapter._fitted[0]
    event_types = {t: i for i, t in enumerate(dict.fromkeys(e.type for e in corr_a.training.events))}
    object_types = {t: i for i, t in enumerate(dict.fromkeys(
        o.type for o in corr_a.training.objects if o.id in corr_a.candidate_object_ids))}
    torch = adapter.torch
    b_objects = [o for o in corr_b.training.objects if o.id in corr_b.candidate_object_ids]
    event_map = {e.id: i for i, e in enumerate(corr_b.training.events)}
    object_map = {o.id: i for i, o in enumerate(b_objects)}
    b_object_types = torch.tensor([object_types[o.type] for o in b_objects], device=adapter.device)
    adapter.node_type_override = (
        torch.tensor([event_types[e.type] for e in corr_b.training.events], device=adapter.device),
        b_object_types)
    adapter._fitted = (model, event_map, object_map, b_object_types)
    adapter._pair_tables = None
    adapter._time_index = None  # temporal tables are rebuilt on B's observed log
    transfer_predictions = adapter.predict(corr_b)
    out["B_transfer"] = metrics(adapter.evaluate(corr_b, transfer_predictions))
    print("B transfer (trained on A only)", out["B_transfer"], flush=True)

    # two-stage version of the transfer model, and references on B
    class Fixed:  # serves the predictions already made
        def predict(self, corruption): return transfer_predictions
        def timing(self): return 0.0, 0.0
    two = CandidateFilteredPredictor(Fixed(), 2, name="transfer-two-stage-2hop")
    out["B_transfer_two_stage_2hop"] = metrics(two.evaluate(corr_b, two.predict(corr_b)))
    cooc = CoOccurrenceBaseline(top_k=10000)
    out["B_cooccurrence"] = metrics(evaluate_rankings("co-occurrence", corr_b, cooc.predict(corr_b)))
    rnd = CandidateFilteredPredictor(None, 2, name="random-two-stage-2hop")
    out["B_random_two_stage_2hop"] = metrics(rnd.evaluate(corr_b, rnd.predict(corr_b)))
    print("B co-occurrence", out["B_cooccurrence"], flush=True)

    # 3. in-domain reference: the same model type trained on B
    if not args.no_retrain:
        retrain = ScalableGnnAdapter(config)
        retrain.event_times = getattr(adapter, "event_times", None)
        out["B_retrain"] = metrics(retrain.evaluate(corr_b, retrain.predict(corr_b)))
        print("B retrain (trained on B)", out["B_retrain"], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[*] Written {args.out}")


if __name__ == "__main__":
    main()
