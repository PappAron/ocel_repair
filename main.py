import argparse
import sys

from adapters.loaders import JsonLoader, SqliteLoader
from adapters.predictors import CoOccurrenceBaseline, GnnPredictor
from adapters.gnn import GnnConfig
from adapters.displays import ConsoleDisplay
from core.gap_simulator import (
    RandomEventObjectGapSimulator,
    SampledGapSimulator,
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
        simulator = SampledGapSimulator(
            sample_size=args.samples,
            validation=not args.no_validation,
        )
        gnn_config = GnnConfig(
            epochs=args.epochs,
            top_k=10000,
            sage_aggregation="sum",
            negatives_per_positive=20,
            sampled_softmax_temperature=0.1,
            same_type_negative_fraction=0.5,
            cooccurrence_candidates=False,
            checkpoint_path=args.checkpoint,
            validation_interval=args.validation_interval,
        )

        if args.mode == "train":
            raw_data = loader.load(args.source_path)
            corruption = simulator.introduce_gaps(raw_data)
            print(f"[*] Training GNN on {args.source_path}...\n")
            GnnPredictor(gnn_config).train(corruption)
            return

        pipeline = OcelPipeline(
            loader,
            CoOccurrenceBaseline(top_k=10000),
            display,
            simulator,
        )
        pipeline.compare(
            args.source_path,
            {
                "co-occurrence": CoOccurrenceBaseline(top_k=10000),
                "gnn": GnnPredictor(gnn_config),
            },
        )

if __name__ == "__main__":
    main()