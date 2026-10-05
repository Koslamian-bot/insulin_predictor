"""
CLI Entry Point for Module 3: Deep-Learning Prediction Engine.

Usage:
    # Train GCN + LSTM model:
    python run_module3.py --mode train --model gcn_lstm --epochs 15

    # Run comparative benchmark across all 4 architectures (LSTM, GRU, BiLSTM, GCN+LSTM):
    python run_module3.py --mode compare --epochs 15

    # Evaluate an existing model checkpoint on test set:
    python run_module3.py --mode evaluate --model gcn_lstm

    # Show version:
    python run_module3.py --version
"""

import argparse
import sys
import os

from module3.config import ExperimentConfig, ModelConfig, TrainConfig
from module3.pipeline.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Module 3: Deep-Learning Prediction Engine for Bolus Insulin Dosage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0 (Module 3 Deep Learning Engine - powered by standard argparse)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "compare"],
        default="compare",
        help="Execution mode: 'train' single model, 'evaluate' checkpoint on test split, or 'compare' all 4 architectures",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn_lstm", "lstm", "gru", "bilstm", "gcn_gru", "gcn_bilstm"],
        default="gcn_lstm",
        help="Architecture to train/evaluate (used when --mode is 'train' or 'evaluate')",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to evaluate in 'compare' mode (e.g. 'gcn_lstm,gcn_gru,gcn_bilstm')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for recurrent layer",
    )
    parser.add_argument(
        "--gcn-dim",
        type=int,
        default=64,
        help="Feature dimension for GCN layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["weighted_smooth_l1", "smooth_l1", "mse", "mae"],
        default="weighted_smooth_l1",
        help="Loss function ('weighted_smooth_l1' compensates for bolus sparsity)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=8.0,
        help="Loss weight multiplier for active bolus targets (y > 0.1 U)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/graphs",
        help="Directory containing PyG graph datasets (train.pt, val.pt, test.pt)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory to save trained model checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory to save prediction CSVs and metrics JSON",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/reports",
        help="Directory to save Markdown benchmark reports",
    )

    args = parser.parse_args()

    # Build configurations
    model_cfg = ModelConfig(
        model_type=args.model,
        gcn_hidden_dim=args.gcn_dim,
        temporal_hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_fn=args.loss,
        pos_weight=args.pos_weight,
        seed=args.seed,
    )
    exp_cfg = ExperimentConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        model=model_cfg,
        train=train_cfg,
    )

    runner = ExperimentRunner(exp_cfg)

    if args.mode == "compare":
        models_to_run = (
            [m.strip().lower() for m in args.models.split(",") if m.strip()]
            if args.models
            else ["lstm", "gru", "bilstm", "gcn_lstm", "gcn_gru", "gcn_bilstm"]
        )
        runner.run_comparative_benchmark(
            model_types=models_to_run,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.mode == "train":
        # Train and evaluate single model
        runner.run_single_experiment(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.mode == "evaluate":
        # Evaluate checkpoint on test set
        from module3.models import build_model
        from module3.data.graph_dataset import load_pyg_split, create_dataloader

        test_list = load_pyg_split(args.data_dir, "test")
        test_loader = create_dataloader(test_list, batch_size=args.batch_size, shuffle=False)
        model = build_model(model_cfg)
        chk_path = os.path.join(args.models_dir, f"{args.model}_best.pt")
        runner.evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            model_name=args.model,
            checkpoint_path=chk_path,
        )


if __name__ == "__main__":
    main()
