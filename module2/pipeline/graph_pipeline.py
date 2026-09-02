"""
Module 2 Graph Pipeline Orchestrator.

Executes the end-to-end Module 2 pipeline:
Ontology Ingestion -> Canonical Data Slicing -> Graph Construction
-> Zero-Leakage Validation -> PyTorch Geometric Serialization -> Quality Reporting.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from module2.ontology.ontology_parser import OntologyParser
from module2.graph.window_slicer import WindowSlicer
from module2.graph.graph_builder import GraphBuilder
from module2.graph.graph_validator import GraphValidator
from module2.dataset.pyg_converter import PyGConverter

logger = logging.getLogger(__name__)


class GraphPipeline:
    """
    End-to-end pipeline generating PyTorch Geometric datasets from Module 1 data.
    """

    def __init__(
        self,
        ontology_path: Optional[str] = None,
        window_minutes: int = 120,
        step_minutes: int = 5,
        stride_minutes: int = 5,
        max_cgm_missing_steps: int = 6,
        include_all_bolus: bool = False,
    ):
        self.ontology_parser = OntologyParser(ontology_path=ontology_path)
        self.window_slicer = WindowSlicer(
            window_minutes=window_minutes,
            step_minutes=step_minutes,
            stride_minutes=stride_minutes,
            max_cgm_missing_steps=max_cgm_missing_steps,
            include_all_bolus=include_all_bolus,
        )
        self.graph_builder = GraphBuilder()
        self.graph_validator = GraphValidator()

    def run(
        self,
        processed_dir: str = "data/processed",
        output_dir: str = "data/graphs",
        reports_dir: str = "data/reports",
        splits: Optional[List[str]] = None,
        max_graphs_per_split: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute full pipeline across all splits.
        """
        start_time = time.time()
        splits = splits or ["train", "val", "test"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        print("\n==================================================")
        print("[START] INITIATING MODULE 2 GRAPH PIPELINE")
        print("==================================================")

        # 1. Parse Ontology
        print("[1/3] Parsing ontology schema...", end=" ", flush=True)
        schema = self.ontology_parser.parse()
        print(f"Done ({len(schema.classes)} classes, {len(schema.object_properties) + len(schema.datatype_properties)} properties)")

        summary_metrics: Dict[str, Any] = {
            "ontology_uri": schema.ontology_uri,
            "classes_count": len(schema.classes),
            "splits": {},
        }

        # 2. Process Each Split
        print("[2/3] Processing time windows & constructing graphs...")
        for split in splits:
            split_file = os.path.join(processed_dir, f"canonical_{split}.csv")
            if not os.path.exists(split_file):
                print(f"  [!] Warning: Split file not found at {split_file}. Skipping.")
                continue

            print(f"  --> Processing split: {split.upper()} ({split_file})...")
            df = pd.read_csv(split_file)

            data_list: List[Data] = []
            node_counts: List[int] = []
            edge_counts: List[int] = []
            target_values: List[float] = []
            rejected_count = 0

            for w_slice in self.window_slicer.slice_dataframe(df):
                g_dict = self.graph_builder.build_graph(w_slice)
                is_valid, issues = self.graph_validator.validate(g_dict)

                if is_valid:
                    pyg_data = PyGConverter.to_pyg_data(g_dict)
                    data_list.append(pyg_data)
                    node_counts.append(pyg_data.num_nodes)
                    edge_counts.append(pyg_data.edge_index.shape[1])
                    target_values.append(float(pyg_data.y.item()))
                else:
                    rejected_count += 1

                if max_graphs_per_split and len(data_list) >= max_graphs_per_split:
                    break

            # Save split to .pt
            out_file = os.path.join(output_dir, f"{split}.pt")
            torch.save(data_list, out_file)

            bolus_events = sum(1 for y in target_values if y > 0.0)
            split_summary = {
                "total_graphs": len(data_list),
                "rejected_windows": rejected_count,
                "mean_nodes_per_graph": float(np.mean(node_counts)) if node_counts else 0.0,
                "mean_edges_per_graph": float(np.mean(edge_counts)) if edge_counts else 0.0,
                "total_bolus_targets": bolus_events,
                "bolus_target_percentage": float(bolus_events / max(1, len(data_list)) * 100),
                "file_path": out_file,
            }
            summary_metrics["splits"][split] = split_summary
            print(
                f"      [OK] {split.upper()}: {len(data_list):,} graphs saved ({bolus_events:,} active bolus targets) -> {out_file}"
            )

        elapsed = time.time() - start_time
        summary_metrics["elapsed_seconds"] = elapsed
        print(f"[3/3] Generating summary report (completed in {elapsed:.2f}s)...")

        # Save JSON report
        json_report_path = os.path.join(reports_dir, "graph_generation_report.json")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        # Save Markdown report
        md_report_path = os.path.join(reports_dir, "graph_generation_report.md")
        self._write_markdown_report(summary_metrics, md_report_path)

        print("==================================================")
        print("[COMPLETE] MODULE 2 GRAPH PIPELINE SUCCEEDED")
        print(f"Report: {md_report_path}")
        print("==================================================\n")

        return summary_metrics

    @staticmethod
    def _write_markdown_report(metrics: Dict[str, Any], file_path: str) -> None:
        """Format and write human-readable Markdown summary report."""
        lines = [
            "# Module 2 Knowledge Graph Generation Report",
            "",
            "## 1. Executive Summary",
            f"- **Ontology:** `{metrics.get('ontology_uri')}`",
            f"- **Ontology Classes Ingested:** {metrics.get('classes_count')}",
            f"- **Execution Time:** {metrics.get('elapsed_seconds', 0.0):.2f} seconds",
            "",
            "## 2. Split Summary",
            "",
            "| Split | Total Graphs | Mean Nodes | Mean Edges | Active Bolus Targets | Target % | Output File |",
            "|---|---|---|---|---|---|---|",
        ]
        for split, d in metrics.get("splits", {}).items():
            lines.append(
                f"| **{split}** | {d['total_graphs']:,} | {d['mean_nodes_per_graph']:.1f} | {d['mean_edges_per_graph']:.1f} | {d['total_bolus_targets']:,} | {d['bolus_target_percentage']:.2f}% | `{os.path.basename(d['file_path'])}` |"
            )

        lines.extend([
            "",
            "## 3. Structural & Leakage Safeguards Verified",
            "- **Zero-Leakage Enforcement:** Target bolus at time $t$ is strictly mapped to `data.y` and isolated from input features $X$.",
            "- **Sparse Topology:** Consecutive temporal edges ($t_i \\to t_{i+1}$) ensure sparse graphs and prevent GCN over-smoothing.",
            "- **Dynamic Modalities:** Missing sensors are pruned without artificial zero-padding.",
        ])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def build_graph_dataset(
    processed_dir: str = "data/processed",
    output_dir: str = "data/graphs",
    reports_dir: str = "data/reports",
) -> Dict[str, Any]:
    """Convenience top-level entry point."""
    pipeline = GraphPipeline()
    return pipeline.run(
        processed_dir=processed_dir,
        output_dir=output_dir,
        reports_dir=reports_dir,
    )
