"""
Experiment Runner & Comparative Benchmark Pipeline for Module 3.
"""

import os
import copy
import logging
from typing import Dict, List, Any, Optional

from module3.config import ExperimentConfig, ModelConfig, TrainConfig
from module3.data.graph_dataset import load_pyg_split, create_dataloader, create_weighted_dataloader
from module3.models import build_model
from module3.training.trainer import Trainer
from module3.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Executes training, evaluation, and comparative benchmarking across all models.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.evaluator = Evaluator(self.config)

    def run_single_experiment(
        self,
        model_type: str = "gcn_lstm",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Train and evaluate a single model architecture.
        """
        # Load datasets
        train_list = load_pyg_split(self.config.data_dir, "train")
        val_list = load_pyg_split(self.config.data_dir, "val")
        test_list = load_pyg_split(self.config.data_dir, "test")

        train_cfg = copy.deepcopy(self.config.train)
        if epochs is not None:
            train_cfg.epochs = epochs
        if batch_size is not None:
            train_cfg.batch_size = batch_size
        if learning_rate is not None:
            train_cfg.learning_rate = learning_rate

        train_loader = create_weighted_dataloader(
            train_list,
            batch_size=train_cfg.batch_size,
            pos_weight=getattr(train_cfg, "pos_weight", 8.0),
            shuffle=True,
        )
        val_loader = create_dataloader(val_list, batch_size=train_cfg.batch_size, shuffle=False)
        test_loader = create_dataloader(test_list, batch_size=train_cfg.batch_size, shuffle=False)

        model_cfg = copy.deepcopy(self.config.model)
        model_cfg.model_type = model_type

        model = build_model(model_cfg)
        trainer = Trainer(
            model=model,
            config=train_cfg,
            model_name=model_type,
            save_dir=self.config.models_dir,
        )

        train_results = trainer.fit(train_loader, val_loader)
        test_results = self.evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            model_name=model_type,
            checkpoint_path=train_results["checkpoint_path"],
        )

        return {
            "model_type": model_type,
            "train_results": train_results,
            "test_results": test_results,
        }

    def run_comparative_benchmark(
        self,
        model_types: Optional[List[str]] = None,
        epochs: int = 15,
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Execute fair empirical comparison across all baseline and proposed architectures:
        1. LSTM
        2. GRU
        3. BiLSTM
        4. GCN + LSTM
        """
        model_types = model_types or ["lstm", "gru", "bilstm", "gcn_lstm"]

        print("\n" + "=" * 80)
        print("STARTING MODULE 3 COMPARATIVE BENCHMARK EXPERIMENT")
        print(f"Models to evaluate: {', '.join(m.upper() for m in model_types)}")
        print(f"Epochs per model:   {epochs} | Batch size: {batch_size}")
        print("=" * 80)

        # Load datasets once
        train_list = load_pyg_split(self.config.data_dir, "train")
        val_list = load_pyg_split(self.config.data_dir, "val")
        test_list = load_pyg_split(self.config.data_dir, "test")

        train_loader = create_weighted_dataloader(
            train_list,
            batch_size=batch_size,
            pos_weight=getattr(self.config.train, "pos_weight", 8.0),
            shuffle=True,
        )
        val_loader = create_dataloader(val_list, batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(test_list, batch_size=batch_size, shuffle=False)

        all_results = []
        for mtype in model_types:
            model_cfg = copy.deepcopy(self.config.model)
            model_cfg.model_type = mtype

            train_cfg = copy.deepcopy(self.config.train)
            train_cfg.epochs = epochs
            train_cfg.batch_size = batch_size

            model = build_model(model_cfg)
            trainer = Trainer(
                model=model,
                config=train_cfg,
                model_name=mtype,
                save_dir=self.config.models_dir,
            )

            train_res = trainer.fit(train_loader, val_loader)
            test_res = self.evaluator.evaluate_model(
                model=model,
                test_loader=test_loader,
                model_name=mtype,
                checkpoint_path=train_res["checkpoint_path"],
            )
            all_results.append(test_res)

        # Generate combined report
        self.evaluator.generate_comparison_report(all_results)
        print("=" * 80)
        print("MODULE 3 COMPARATIVE BENCHMARK COMPLETE")
        print("=" * 80 + "\n")

        return all_results
