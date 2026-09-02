"""
CLI Entry Point for Module 2: Ontology-Grounded Graph Pipeline.

Usage:
    python run_module2.py
    python run_module2.py --input data/processed --output data/graphs --window-minutes 120
"""

import argparse
import sys
from module2.pipeline.graph_pipeline import GraphPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Module 2: Ontology-Grounded Dynamic Knowledge Graph Construction"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0 (Module 2 Knowledge Graph Engine - powered by standard argparse)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed",
        help="Directory containing Module 1 canonical CSV splits",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/graphs",
        help="Directory to save PyTorch Geometric .pt graph files",
    )
    parser.add_argument(
        "--reports",
        type=str,
        default="data/reports",
        help="Directory to save graph generation summary reports",
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default="diabetes_insulin_ontology.owl",
        help="Path to the finalized OWL ontology file",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=120,
        help="Sliding window duration in minutes (default: 120)",
    )
    parser.add_argument(
        "--stride-minutes",
        type=int,
        default=5,
        help="Sliding window stride in minutes (default: 5)",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional limit on number of graphs to generate per split (useful for fast testing)",
    )
    parser.add_argument(
        "--include-all-bolus",
        action="store_true",
        help="Always include 100% of all valid active bolus windows (y > 0) regardless of stride",
    )

    args = parser.parse_args()

    pipeline = GraphPipeline(
        ontology_path=args.ontology,
        window_minutes=args.window_minutes,
        stride_minutes=args.stride_minutes,
        include_all_bolus=args.include_all_bolus,
    )

    metrics = pipeline.run(
        processed_dir=args.input,
        output_dir=args.output,
        reports_dir=args.reports,
        max_graphs_per_split=args.max_graphs,
    )

    if not metrics.get("splits"):
        print("[!] No graphs were generated. Check your input directory path.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
