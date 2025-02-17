"""
Model Management Utilities.

This module provides utilities for managing model versions, checkpoints,
and evaluation results.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.paths import RL_MODELS, MODELS_CACHE
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class ModelManager:
    """Manages model versions, checkpoints, and evaluation results."""

    def __init__(self, model_type: str = "rl"):
        """Initialize model manager.

        Args:
            model_type: Type of model to manage (e.g., 'rl', 'lstm')
        """
        self.model_type = model_type
        self.model_dir = RL_MODELS if model_type == "rl" else None
        if self.model_dir is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create necessary directories
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.best_models_dir = self.model_dir / "best_models"
        self.eval_results_dir = self.model_dir / "eval_results"
        self.archive_dir = self.model_dir / "archive"
        self.evaluation_dir = self.model_dir / "evaluation"
        self.plots_dir = self.model_dir / "plots"
        self.metadata_file = self.model_dir / "model_metadata.json"

        for dir_path in [
            self.checkpoints_dir,
            self.best_models_dir,
            self.eval_results_dir,
            self.archive_dir,
            self.evaluation_dir,
            self.plots_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load or create metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        else:
            metadata = {
                "models": {},
                "best_model": None,
                "last_cleanup": None,
                "experiments": {},
            }
            self._save_metadata(metadata)
            return metadata

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    def save_checkpoint(
        self,
        model_path: Path,
        metrics: Dict[str, float],
        episode: int,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint with metadata.

        Args:
            model_path: Path to model file
            metrics: Dictionary of evaluation metrics
            episode: Current training episode
            is_best: Whether this is the best model so far
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{timestamp}"

        # Copy model file to checkpoints directory
        checkpoint_path = self.checkpoints_dir / f"{model_id}_ep{episode}.pth"
        shutil.copy2(model_path, checkpoint_path)

        # Update metadata
        self.metadata["models"][model_id] = {
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
            "episode": episode,
            "timestamp": timestamp,
        }

        if is_best:
            # Copy to best models directory
            best_model_path = self.best_models_dir / f"best_model_{timestamp}.pth"
            shutil.copy2(model_path, best_model_path)
            self.metadata["best_model"] = {
                "model_id": model_id,
                "path": str(best_model_path),
                "metrics": metrics,
            }

        self._save_metadata(self.metadata)
        logger.info(f"Saved {'best ' if is_best else ''}checkpoint: {checkpoint_path}")

    def load_best_model(self) -> Optional[Path]:
        """Load the best performing model.

        Returns:
            Path to best model file or None if no best model exists
        """
        if self.metadata["best_model"] is None:
            return None

        best_model_path = Path(self.metadata["best_model"]["path"])
        if not best_model_path.exists():
            logger.warning(f"Best model file not found: {best_model_path}")
            return None

        return best_model_path

    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Get evaluation metrics for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary of metrics or None if model not found
        """
        return self.metadata["models"].get(model_id, {}).get("metrics")

    def cleanup_old_checkpoints(
        self,
        max_checkpoints: int = 5,
        min_keep_days: int = 7,
    ) -> None:
        """Clean up old checkpoints while keeping important ones.

        Args:
            max_checkpoints: Maximum number of checkpoints to keep
            min_keep_days: Minimum days to keep checkpoints
        """
        if not self.metadata["models"]:
            return

        # Sort checkpoints by timestamp
        checkpoints = [
            (model_id, meta) for model_id, meta in self.metadata["models"].items()
        ]
        checkpoints.sort(
            key=lambda x: datetime.strptime(x[1]["timestamp"], "%Y%m%d_%H%M%S"),
            reverse=True,
        )

        # Keep recent checkpoints
        current_time = datetime.now()
        to_delete = []

        for model_id, meta in checkpoints[max_checkpoints:]:
            checkpoint_time = datetime.strptime(meta["timestamp"], "%Y%m%d_%H%M%S")
            days_old = (current_time - checkpoint_time).days

            # Skip if checkpoint is too recent or is best model
            if days_old < min_keep_days or (
                self.metadata["best_model"]
                and model_id == self.metadata["best_model"]["model_id"]
            ):
                continue

            # Delete checkpoint file
            checkpoint_path = Path(meta["checkpoint_path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            to_delete.append(model_id)

        # Update metadata
        for model_id in to_delete:
            del self.metadata["models"][model_id]

        self.metadata["last_cleanup"] = current_time.isoformat()
        self._save_metadata(self.metadata)

        logger.info(f"Cleaned up {len(to_delete)} old checkpoints")

    def save_evaluation_results(
        self,
        model_id: str,
        results: Dict[str, Any],
    ) -> None:
        """Save detailed evaluation results.

        Args:
            model_id: Model identifier
            results: Dictionary of evaluation results
        """
        # Create evaluation results directory for this model
        eval_dir = self.eval_results_dir / model_id
        eval_dir.mkdir(exist_ok=True)

        # Save results as CSV files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        if "metrics" in results:
            pd.DataFrame([results["metrics"]]).to_csv(
                eval_dir / f"metrics_{timestamp}.csv"
            )

        # Save portfolio performance
        if "portfolio_performance" in results:
            pd.DataFrame(results["portfolio_performance"]).to_csv(
                eval_dir / f"portfolio_performance_{timestamp}.csv"
            )

        # Save trade history
        if "trade_history" in results:
            pd.DataFrame(results["trade_history"]).to_csv(
                eval_dir / f"trade_history_{timestamp}.csv"
            )

        logger.info(f"Saved evaluation results for model {model_id}")

    def get_training_progress(self, model_id: str) -> Optional[pd.DataFrame]:
        """Get training progress data for visualization.

        Args:
            model_id: Model identifier

        Returns:
            DataFrame with training progress or None if not found
        """
        if model_id not in self.metadata["models"]:
            return None

        progress_file = self.checkpoints_dir / model_id / "training_progress.csv"
        if not progress_file.exists():
            return None

        return pd.read_csv(progress_file)

    def compare_models(
        self,
        model_ids: List[str],
        metric_name: str,
    ) -> pd.DataFrame:
        """Compare multiple models based on a specific metric.

        Args:
            model_ids: List of model identifiers to compare
            metric_name: Name of metric to compare

        Returns:
            DataFrame with model comparison
        """
        comparison_data = []

        for model_id in model_ids:
            metrics = self.get_model_metrics(model_id)
            if metrics and metric_name in metrics:
                comparison_data.append(
                    {
                        "model_id": model_id,
                        "timestamp": self.metadata["models"][model_id]["timestamp"],
                        metric_name: metrics[metric_name],
                    }
                )

        return pd.DataFrame(comparison_data)

    def save_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        episode: int,
        experiment_id: str,
    ) -> None:
        """Save training metrics and generate plots.

        Args:
            metrics: Dictionary of training metrics
            episode: Current episode number
            experiment_id: Unique identifier for the experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment directory
        exp_dir = self.evaluation_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save metrics as CSV
        metrics_df = pd.DataFrame(metrics)
        metrics_file = exp_dir / f"training_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file)

        # Generate and save plots
        self._save_training_plots(metrics, episode, experiment_id)

        # Update metadata
        if experiment_id not in self.metadata["experiments"]:
            self.metadata["experiments"][experiment_id] = {
                "start_time": timestamp,
                "metrics_files": [],
                "plot_files": [],
            }

        self.metadata["experiments"][experiment_id]["metrics_files"].append(
            str(metrics_file)
        )
        self._save_metadata(self.metadata)

        logger.info(f"Saved training metrics for experiment {experiment_id}")

    def _save_training_plots(
        self,
        metrics: Dict[str, List[float]],
        episode: int,
        experiment_id: str,
    ) -> None:
        """Generate and save training plots.

        Args:
            metrics: Dictionary of training metrics
            episode: Current episode number
            experiment_id: Unique identifier for the experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create plots directory for this experiment
        plots_dir = self.plots_dir / experiment_id
        plots_dir.mkdir(exist_ok=True)

        # Plot training curves
        plt.figure(figsize=(15, 10))

        # Returns plot
        plt.subplot(2, 2, 1)
        sns.lineplot(data=metrics["episode_returns"])
        plt.title("Episode Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")

        # Portfolio values plot
        plt.subplot(2, 2, 2)
        sns.lineplot(data=metrics["portfolio_values"])
        plt.title("Portfolio Value")
        plt.xlabel("Episode")
        plt.ylabel("Value ($)")

        # Losses plot
        plt.subplot(2, 2, 3)
        sns.lineplot(data=metrics["actor_losses"], label="Actor")
        sns.lineplot(data=metrics["critic_losses"], label="Critic")
        plt.title("Losses")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()

        # Entropy plot
        plt.subplot(2, 2, 4)
        sns.lineplot(data=metrics["entropy"])
        plt.title("Policy Entropy")
        plt.xlabel("Episode")
        plt.ylabel("Entropy")

        plt.tight_layout()

        # Save plot
        plot_file = plots_dir / f"training_curves_ep{episode}_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()

        # Update metadata
        self.metadata["experiments"][experiment_id]["plot_files"].append(str(plot_file))
        self._save_metadata(self.metadata)

    def archive_experiment(self, experiment_id: str) -> None:
        """Archive an experiment's data.

        Args:
            experiment_id: Unique identifier for the experiment
        """
        if experiment_id not in self.metadata["experiments"]:
            logger.warning(f"Experiment {experiment_id} not found")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"{experiment_id}_{timestamp}"
        archive_path.mkdir(exist_ok=True)

        # Move experiment data to archive
        exp_data = self.metadata["experiments"][experiment_id]

        # Archive metrics files
        for metrics_file in exp_data["metrics_files"]:
            if Path(metrics_file).exists():
                shutil.move(metrics_file, archive_path)

        # Archive plot files
        for plot_file in exp_data["plot_files"]:
            if Path(plot_file).exists():
                shutil.move(plot_file, archive_path)

        # Archive evaluation results
        eval_dir = self.eval_results_dir / experiment_id
        if eval_dir.exists():
            shutil.move(str(eval_dir), archive_path)

        # Update metadata
        exp_data["archived_at"] = timestamp
        exp_data["archive_path"] = str(archive_path)
        self._save_metadata(self.metadata)

        logger.info(f"Archived experiment {experiment_id} to {archive_path}")
