#!/usr/bin/env python3
"""
Top-Level CLI Execution Script for Module 1 Data Ingestion & Preprocessing Pipeline.

Examples:
    # Run default OhioT1DM dataset preprocessing:
    python run_pipeline.py

    # Run OhioT1DM with custom paths:
    python run_pipeline.py --dataset ohiot1dm --input OhioT1DM --output data

    # Run HUPA-UCM dataset preprocessing:
    python run_pipeline.py --dataset hupa_ucm --input "HUPA-UCM Diabetes Dataset/Preprocessed" --output data_hupa

    # List all available registered adapters:
    python run_pipeline.py --list-adapters
"""

import argparse
import sys
import os

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from module1.adapters.registry import AdapterRegistry
from module1.pipeline.preprocessing_pipeline import preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Module 1: Dataset-Agnostic Data Ingestion and Preprocessing Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0 (Module 1 Data Preprocessing Engine - powered by standard argparse)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ohiot1dm",
        help="Dataset identifier (e.g. 'ohiot1dm', 'hupa_ucm').",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"OhioT1DM",
        help="Path to directory containing raw patient files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"data",
        help="Output base directory for canonical, processed, reports, and scalers.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to custom YAML configuration file.",
    )
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="List all registered dataset adapters and exit.",
    )

    args = parser.parse_args()

    if args.list_adapters:
        print("\nRegistered Dataset Adapters in Module 1:")
        for name in AdapterRegistry.list_adapters():
            print(f"  - {name}")
        print()
        return

    # Normalize relative paths
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)

    print(f"Starting Module 1 Pipeline for dataset: {args.dataset}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    result = preprocess(
        dataset=args.dataset,
        input_path=input_path,
        output_path=output_path,
        config_path=args.config,
    )

    print("Execution Summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
