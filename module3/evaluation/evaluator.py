"""
Model Evaluator and Publication-Ready Report Generator.

Evaluates trained checkpoints on test split (data/graphs/test.pt),
saves predictions vs ground truth, and generates Markdown reports.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from module3.models.base import BaseModel
from module3.config import ExperimentConfig
from module3.training.trainer import Trainer

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates trained models and formats scientific reports.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.reports_dir, exist_ok=True)

    def evaluate_model(
        self,
        model: BaseModel,
        test_loader: DataLoader,
        model_name: str = "model",
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run test evaluation and serialize results.
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path, map_location=self.config.train.device)
            logger.info("Loaded checkpoint from: %s", checkpoint_path)

        trainer = Trainer(
            model=model,
            config=self.config.train,
            model_name=model_name,
            save_dir=self.config.models_dir,
        )

        test_loss, metrics, y_true, y_pred = trainer.evaluate(test_loader)

        # 1. Save Predictions vs Ground Truth CSV
        pred_df = pd.DataFrame({
            "sample_index": np.arange(len(y_true)),
            "actual_bolus_dose": y_true,
            "predicted_bolus_dose": y_pred,
            "absolute_error": np.abs(y_true - y_pred),
            "is_active_bolus": (y_true > 0.1).astype(int),
        })
        pred_csv_path = os.path.join(self.config.results_dir, f"{model_name}_test_predictions.csv")
        pred_df.to_csv(pred_csv_path, index=False)

        # 2. Save Metrics JSON
        results = {
            "model_name": model_name,
            "parameters": model.count_parameters(),
            "test_loss": float(test_loss),
            "metrics": metrics,
            "predictions_csv": pred_csv_path,
        }
        metrics_json_path = os.path.join(self.config.results_dir, f"{model_name}_test_metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n[Test Results: {model_name.upper()}]")
        print(f"  MAE:              {metrics['mae']:.3f} Units")
        print(f"  RMSE:             {metrics['rmse']:.3f} Units")
        print(f"  R² Score:         {metrics['r2']:.3f}")
        print(f"  Pearson r:        {metrics['pearson_r']:.3f}")
        print(f"  Active Bolus MAE: {metrics['active_bolus_mae']:.3f} Units (n={metrics['active_samples']})")
        print(f"  Bolus F1-Score:   {metrics['bolus_detection_f1']:.3f}")
        print(f"  Saved to:         {pred_csv_path}")

        return results

    def generate_comparison_report(
        self,
        results_list: list,
        report_path: Optional[str] = None,
    ) -> str:
        """
        Compile comparative experiment results into a clean Markdown table.
        """
        report_path = report_path or os.path.join(self.config.reports_dir, "module3_experiment_report.md")

        lines = [
            "# Module 3 Deep Learning Experiment & Comparison Report",
            "",
            "## 1. Executive Summary",
            "This report summarizes the empirical benchmark comparing baseline recurrent models (LSTM, GRU, BiLSTM) "
            "against the proposed Ontology-Grounded Graph Convolutional Network (GCN + LSTM) on the OhioT1DM dataset.",
            "",
            "**Prediction Target:** `target_current_bolus_dose` (Rapid-acting insulin bolus dosage in Units at prediction time $t$).",
            "**Zero-Leakage Guarantee:** Target dosage at time $t$ is strictly isolated to $y$ and never present in input node features $X$ or graph topology.",
            "",
            "## 2. Comparative Performance Benchmark",
            "",
            "| Model Architecture | Parameters | Test MAE (U) | Test RMSE (U) | R² Score | Pearson r | Active Bolus MAE (U) | Bolus F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for r in results_list:
            m = r["metrics"]
            lines.append(
                f"| **{r['model_name'].upper()}** | {r['parameters']:,} | "
                f"{m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} | {m['pearson_r']:.3f} | "
                f"{m['active_bolus_mae']:.3f} | {m['bolus_detection_f1']:.3f} |"
            )

        lines.extend([
            "",
            "## 3. Scientific Methodology & Discussion",
            "- **Fair Ablation Baseline:** All models were trained and evaluated on identical train/val/test splits with identical random seeds.",
            "- **Preservation of 24-Step Temporal Structure:** The 120-minute window ($24 \\times 5\\text{ min}$ steps) is preserved across all models.",
            "- **Proposed Model Mechanism:** GCN convolutions propagate relational medical knowledge across the patient subgraph, followed by step-wise pooling into recurrent layers.",
            "",
            "## 4. Checkpoints and Output Files",
        ])

        for r in results_list:
            lines.append(f"- **{r['model_name'].upper()}:** `{r['predictions_csv']}`")

        report_content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\nComparative Experiment Report written to: {report_path}")
        return report_path
