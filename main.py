import argparse
import json
import sys
from pathlib import Path

from adapters.loaders import JsonLoader, SqliteLoader
from adapters.predictors import CoOccurrenceBaseline, GnnPredictor
from adapters.gnn import GnnConfig
from adapters.gnn_scalable import (
    CandidateFilteredPredictor, ScalableGnnConfig, ScalableGnnPredictor,
)
from adapters.displays import ConsoleDisplay
from core.gap_simulator import (
    RandomEventObjectGapSimulator,
    SampledGapSimulator,
    LogWideGapSimulator,
    SingleTargetGapSimulator,
)
from core.pipeline import OcelPipeline
from core.analyzer import OcelAnalyzer
from core.logging_config import configure_logging

def get_loader(loader_type: str):
    loaders = {"json": JsonLoader, "sqlite": SqliteLoader}
    if loader_type not in loaders:
        sys.exit(f"Error: Unsupported loader '{loader_type}'")
    return loaders[loader_type]()

def get_predictor(predictor_type: str, epochs: int = 400):
    predictors = {"baseline": CoOccurrenceBaseline, "gnn": GnnPredictor}
    if predictor_type not in predictors:
        sys.exit(f"Error: Unsupported predictor '{predictor_type}'")
    if predictor_type == "gnn":
        return GnnPredictor(
            GnnConfig(
                epochs=epochs,
                negatives_per_positive=10,
                sampled_softmax_temperature=0.1,
                cooccurrence_candidates=False,
            )
        )
    return predictors[predictor_type]()

def get_simulator(protocol: str):
    if protocol == "single-target":
        return SingleTargetGapSimulator()
    return RandomEventObjectGapSimulator()

def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="OCEL Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command 1: Pipeline
    run_parser = subparsers.add_parser("run", help="Run the gap simulator pipeline")
    run_parser.add_argument("source_path", type=str)
    run_parser.add_argument("--loader", type=str, choices=["json", "sqlite"], default="sqlite")
    run_parser.add_argument("--predictor", type=str, choices=["baseline", "gnn"], default="baseline")
    run_parser.add_argument("--epochs", type=int, default=400)
    run_parser.add_argument(
        "--protocol", choices=["single-target", "multi-link"],
        default="single-target",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run baseline and GNN comparison with a validation-based split and model selection",
    )
    compare_parser.add_argument("source_path", type=str)
    compare_parser.add_argument("--loader", type=str, choices=["json", "sqlite"], default="sqlite")
    compare_parser.add_argument("--epochs", type=int, default=400)
    compare_parser.add_argument("--samples", type=int, default=200)
    compare_parser.add_argument("--validation-interval", type=int, default=50)
    compare_parser.add_argument(
        "--seed", type=int, default=None,
        help="One seed for the train/test split, the test/validation samples, the "
             "hidden targets and GNN training. Omit to reproduce the submitted "
             "paper's split (split 42, sample 0, targets 42, GNN 42).",
    )
    compare_parser.add_argument(
        "--drop-fraction", type=float, default=0.3,
        help="Share of an evaluated event's objects that are hidden "
             "(at least one per event). Default 0.3 as in the paper.",
    )
    compare_parser.add_argument(
        "--results-json", type=str, default=None,
        help="Write the comparison metrics and run settings to this JSON file.",
    )
    compare_parser.add_argument(
        "--one-object-context", action="store_true",
        help="Diagnostic: add the links of one-object events to all graphs as context "
             "(never damaged, queried or counted in co-occurrence statistics).",
    )
    compare_parser.add_argument(
        "--context-drop", type=float, default=0.0,
        help="Sensitivity (C3): remove each link of every training event independently "
             "with this probability, keeping at least one object per event. Default 0.",
    )
    compare_parser.add_argument(
        "--pair-features", action="store_true",
        help="GNN: add pairwise structural features (co-occurrence, O2O) to the score.",
    )
    compare_parser.add_argument(
        "--missing-rate", type=float, default=None,
        help="Log-wide protocol (Section 6.4): remove every E2O link of the log with this probability.",
    )
    compare_parser.add_argument(
        "--features-only", action="store_true",
        help="Ablation: score candidates by the pairwise-feature MLP only (implies --pair-features).",
    )
    compare_parser.add_argument(
        "--time-negatives", type=float, default=0.0,
        help="Share of same-type negatives drawn among objects close in time (requires --time-feature).",
    )
    compare_parser.add_argument(
        "--time-rank", action="store_true",
        help="Add the relative time-rank feature (requires --time-feature).",
    )
    compare_parser.add_argument(
        "--time-feature", action="store_true",
        help="Add the temporal-proximity pairwise feature (requires --pair-features).",
    )
    compare_parser.add_argument(
        "--edge-dropout", type=float, default=0.0,
        help="GNN: hide this share of E2O edges in every training epoch. Default 0.",
    )
    compare_parser.add_argument(
        "--o2o-drop", type=float, default=0.0,
        help="Sensitivity (E11): remove this share of all object-object links. Default 0.",
    )
    # Scaling experiments (8 Nov 2026, adapters/gnn_scalable.py). Defaults = submitted model.
    compare_parser.add_argument(
        "--object-inputs", choices=["id", "type", "type+degree"], default="id",
        help="GNN object inputs: id = learned vector per object (submitted, transductive); "
             "type / type+degree = no per-object vectors (fixed size, inductive).",
    )
    compare_parser.add_argument(
        "--sampled", action="store_true",
        help="GNN: train on sampled neighbourhoods (fan-out per layer) instead of the full graph.",
    )
    compare_parser.add_argument(
        "--fanouts", type=int, nargs=3, default=[15, 10, 5],
        help="Neighbours sampled per node and relation for each of the 3 layers (with --sampled).",
    )
    compare_parser.add_argument(
        "--two-stage-hops", type=int, nargs="*", default=[],
        help="Also report two-stage repair: rerank only objects within h steps "
             "(shared event or O2O link) of the event's observed objects, for each h given.",
    )
    compare_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path: required to load a model in --mode infer; "
             "must NOT be set for --mode train (which always creates a new one)",
    )
    compare_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Use the original 70/30 test protocol without validation checkpoint selection",
    )
    compare_parser.add_argument(
        "--mode", choices=["train", "infer", "both"], default="both",
        help="train: fit the GNN and save a checkpoint, then stop (no evaluation). "
             "infer: load a saved checkpoint (requires --checkpoint) and evaluate "
             "against the baseline, skipping training. "
             "both (default): train from scratch and evaluate in one run.",
    )

    # Command 2: Overview
    overview_parser = subparsers.add_parser("overview", help="Generate a database overview")
    overview_parser.add_argument("source_path", type=str)
    overview_parser.add_argument("--loader", type=str, choices=["json", "sqlite"], default="sqlite")

    args = parser.parse_args()
    display = ConsoleDisplay()

    if args.command == "run":
        loader = get_loader(args.loader)
        predictor = get_predictor(args.predictor, args.epochs)
        pipeline = OcelPipeline(
            loader,
            predictor,
            display,
            get_simulator(args.protocol),
        )
        
        print(f"[*] Running Pipeline on {args.source_path}...\n")
        pipeline.run(args.source_path)

    elif args.command == "overview":
        loader = get_loader(args.loader)
        analyzer = OcelAnalyzer(loader, display)
        
        print(f"[*] Generating overview for {args.source_path}...\n")
        analyzer.generate_overview(args.source_path)

    elif args.command == "compare":
        if args.mode == "train" and args.checkpoint:
            sys.exit("Error: --checkpoint must not be set in --mode train (it always creates a new checkpoint).")
        if args.mode == "infer" and not args.checkpoint:
            sys.exit("Error: --mode infer requires --checkpoint <path>.")
        if args.mode == "both" and args.checkpoint:
            sys.exit("Error: --checkpoint must not be set in --mode both (it always trains from scratch); use --mode infer to load a checkpoint instead.")

        loader = get_loader(args.loader)
        seed_kwargs = {}
        if args.seed is not None:
            seed_kwargs = dict(
                split_seed=args.seed, sample_seed=args.seed, target_seed=args.seed,
            )
        if args.missing_rate is not None:
            simulator = LogWideGapSimulator(args.missing_rate, sample_size=args.samples,
                                            seed=0 if args.seed is None else args.seed,
                                            o2o_drop=args.o2o_drop)
        else:
          simulator = SampledGapSimulator(
            sample_size=args.samples,
            validation=not args.no_validation,
            drop_fraction=args.drop_fraction,
            context_drop=args.context_drop,
            o2o_drop=args.o2o_drop,
            one_object_context=args.one_object_context,
            **seed_kwargs,
        )
        scalable = (args.object_inputs != "id" or args.sampled)
        gnn_config = (ScalableGnnConfig if scalable else GnnConfig)(
            epochs=args.epochs,
            top_k=10000,
            sage_aggregation="sum",
            negatives_per_positive=20,
            sampled_softmax_temperature=0.1,
            same_type_negative_fraction=0.5,
            cooccurrence_candidates=False,
            checkpoint_path=args.checkpoint,
            validation_interval=args.validation_interval,
            seed=42 if args.seed is None else args.seed,
            pair_features=args.pair_features or args.features_only,
            features_only=args.features_only,
            time_feature=args.time_feature,
            time_rank=args.time_rank,
            time_negatives=args.time_negatives,
            edge_dropout=args.edge_dropout,
            **(dict(object_inputs=args.object_inputs, sampled_training=args.sampled,
                    fanouts=tuple(args.fanouts)) if scalable else {}),
        )
        gnn = ScalableGnnPredictor(gnn_config) if scalable else GnnPredictor(gnn_config)
        if args.time_feature:
            # timestamps of all events: OCEL 2.0 SQLite keeps them in the per-activity tables,
            # and some events (e.g. P2P "System Update") sit in another activity's table
            import sqlite3
            connection = sqlite3.connect(args.source_path)
            event_times = {}
            for (table,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'event\\_%' ESCAPE '\\'"):
                columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
                if {"ocel_id", "ocel_time"} <= columns:
                    event_times.update(connection.execute(f'SELECT ocel_id, ocel_time FROM "{table}"'))
            connection.close()
            gnn.model.event_times = event_times

        if args.mode == "train":
            raw_data = loader.load(args.source_path)
            corruption = simulator.introduce_gaps(raw_data)
            print(f"[*] Training GNN on {args.source_path}...\n")
            gnn.train(corruption)
            return

        pipeline = OcelPipeline(
            loader,
            CoOccurrenceBaseline(top_k=10000),
            display,
            simulator,
        )
        cooccurrence = CoOccurrenceBaseline(top_k=10000)
        models = {"co-occurrence": cooccurrence, "gnn": gnn}
        for hops in args.two_stage_hops:
            # same candidates for every ranker: random order, co-occurrence, GNN
            models[f"random-two-stage-{hops}hop"] = CandidateFilteredPredictor(
                None, hops, name=f"random-two-stage-{hops}hop")
            models[f"co-occurrence-two-stage-{hops}hop"] = CandidateFilteredPredictor(
                cooccurrence, hops, name=f"co-occurrence-two-stage-{hops}hop")
            models[f"gnn-two-stage-{hops}hop"] = CandidateFilteredPredictor(
                gnn, hops, name=f"gnn-two-stage-{hops}hop")
        results = pipeline.compare(args.source_path, models)
        if args.results_json:
            payload = {
                "source_path": args.source_path,
                "seed": args.seed,
                "split_seed": getattr(simulator, "split_seed", args.seed),
                "sample_seed": getattr(simulator, "sample_seed", args.seed),
                "target_seed": getattr(simulator, "target_seed", args.seed),
                "gnn_seed": gnn_config.seed,
                "drop_fraction": args.drop_fraction,
                "context_drop": args.context_drop,
                "o2o_drop": args.o2o_drop,
                "pair_features": args.pair_features or args.features_only,
                "features_only": args.features_only,
                "time_feature": args.time_feature,
                "missing_rate": args.missing_rate,
                "time_rank": args.time_rank,
                "time_negatives": args.time_negatives,
                "edge_dropout": args.edge_dropout,
                "samples": args.samples,
                "epochs": args.epochs,
                "validation": not args.no_validation,
                "object_inputs": args.object_inputs,
                "sampled": args.sampled,
                "fanouts": args.fanouts if args.sampled else None,
                "gnn_parameters": sum(
                    p.numel() for p in gnn.model._fitted[0].parameters()
                ) if gnn.model._fitted else None,
                "two_stage": {
                    name: m.stats for name, m in models.items()
                    if isinstance(m, CandidateFilteredPredictor)
                },
                "results": {
                    name: {
                        "evaluated_links": result.evaluated_links,
                        "hits_at_k": {str(k): float(v) for k, v in result.hits_at_k.items()},
                        "mrr": float(result.mrr),
                        "runtime_seconds": result.runtime_seconds,
                        "fit_seconds": result.fit_seconds,
                        "inference_seconds": result.inference_seconds,
                        "decision": result.decision,
                        "per_query": result.per_query,
                    }
                    for name, result in results.items()
                },
            }
            out = Path(args.results_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[*] Results written to {out}")

if __name__ == "__main__":
    main()