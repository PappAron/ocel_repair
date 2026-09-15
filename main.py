import argparse
import sys

from adapters.loaders import JsonLoader, SqliteLoader
from adapters.predictors import CoOccurrenceBaseline, GnnPredictor
from adapters.gnn import GnnConfig
from adapters.displays import ConsoleDisplay
from core.gap_simulator import (
    NotebookSampleGapSimulator,
    RandomEventObjectGapSimulator,
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
                masked_query_training=True,
                negatives_per_positive=10,
                sampled_softmax_temperature=0.1,
                cooccurrence_candidates=True,
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
        "compare", help="Run baseline and GNN on the same corrupted dataset"
    )
    compare_parser.add_argument("source_path", type=str)
    compare_parser.add_argument("--loader", type=str, choices=["json", "sqlite"], default="sqlite")
    compare_parser.add_argument("--epochs", type=int, default=400)
    compare_parser.add_argument(
        "--protocol", choices=["single-target", "multi-link"],
        default="single-target",
    )

    notebook_parser = subparsers.add_parser(
        "notebook-compare",
        help="Run leakage-free GNN and baseline on the notebook's 200-case sample",
    )
    notebook_parser.add_argument("source_path", type=str)
    notebook_parser.add_argument("--loader", type=str, choices=["json", "sqlite"], default="sqlite")
    notebook_parser.add_argument("--epochs", type=int, default=400)
    notebook_parser.add_argument("--samples", type=int, default=200)
    notebook_parser.add_argument(
        "--visualize",
        type=str,
        default="sample_reconstructions.png",
        help="Output PNG path for sample-level reconstruction visualization",
    )
    notebook_parser.add_argument("--visualization-samples", type=int, default=5)
    notebook_parser.add_argument("--visualization-top-n", type=int, default=5)

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

    elif args.command == "compare":
        loader = get_loader(args.loader)
        pipeline = OcelPipeline(
            loader,
            CoOccurrenceBaseline(),
            display,
            get_simulator(args.protocol),
        )
        pipeline.compare(
            args.source_path,
            {
                "co-occurrence": CoOccurrenceBaseline(),
                "gnn": GnnPredictor(
                    GnnConfig(
                        epochs=args.epochs,
                        masked_query_training=True,
                        negatives_per_positive=10,
                        sampled_softmax_temperature=0.1,
                        cooccurrence_candidates=True,
                    )
                ),
            },
        )

    elif args.command == "overview":
        loader = get_loader(args.loader)
        analyzer = OcelAnalyzer(loader, display)
        
        print(f"[*] Generating overview for {args.source_path}...\n")
        analyzer.generate_overview(args.source_path)

    elif args.command == "notebook-compare":
        loader = get_loader(args.loader)
        simulator = NotebookSampleGapSimulator(sample_size=args.samples)
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
                "gnn": GnnPredictor(
                    GnnConfig(
                        epochs=args.epochs,
                        top_k=10000,
                        notebook_multi_positive=True,
                        sage_aggregation="mean",
                        masked_query_training=True,
                        negatives_per_positive=20,
                        sampled_softmax_temperature=0.1,
                        cooccurrence_candidates=False,
                        same_type_candidates=False,
                        same_type_negative_fraction=0.5,
                        pair_scorer=False,
                        hard_negative_fraction=0.0,
                    )
                ),
            },
            visualization_path=args.visualize,
            visualization_samples=args.visualization_samples,
            visualization_top_n=args.visualization_top_n,
        )

if __name__ == "__main__":
    main()