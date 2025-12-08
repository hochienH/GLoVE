from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, Callback

from darts import TimeSeries
from darts.models import TSMixerModel
from model_train import WeightedLoss


# Prefer Tensor Cores when available for better throughput
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


class PatchedPruningCallback(PyTorchLightningPruningCallback, Callback):
    """Compatibility shim for Lightning pruning callbacks."""
    pass


class TSMixerOptunaRunner:
    """
    Reusable Optuna runner for TSMixer.
    Handles search and trial logging; final training stays in caller.
    """

    def __init__(
        self,
        args,
        train_targets: List[Optional[TimeSeries]],
        val_targets: List[Optional[TimeSeries]],
        train_covs: List[Optional[TimeSeries]],
        val_covs: List[Optional[TimeSeries]],
        input_chunk_length: int,
        use_static: bool,
    ):
        self.args = args
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.train_covs = train_covs
        self.val_covs = val_covs
        self.input_chunk_length = input_chunk_length
        self.use_static = use_static
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.devices = "auto" if self.accelerator == "gpu" else 1
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    def _objective(self, trial: optuna.Trial) -> float:
        args = self.args
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        num_blocks = trial.suggest_categorical("num_blocks", args.search_num_blocks)
        dropout = trial.suggest_categorical("dropout", args.search_dropout)
        hidden_size = trial.suggest_categorical("hidden_size", args.search_hidden_size)
        ff_size = trial.suggest_categorical("ff_size", args.search_ff_size)

        pruning_cb = PatchedPruningCallback(trial, monitor="val_loss")
        early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")
        callbacks = [pruning_cb, early_stop]

        model = TSMixerModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=1,
            hidden_size=hidden_size,
            ff_size=ff_size,
            dropout=dropout,
            num_blocks=num_blocks,
            use_static_covariates=self.use_static,
            use_reversible_instance_norm=True,
            loss_fn=WeightedLoss(args.lambda_weight),
            random_state=args.seed,
            optimizer_kwargs={"lr": lr},
            save_checkpoints=False,
            force_reset=True,
            add_encoders=None,
            pl_trainer_kwargs={
                "accelerator": self.accelerator,
                "devices": self.devices,
                "default_root_dir": args.log_dir,
                "enable_progress_bar": False,
                "callbacks": callbacks,
                "precision": "bf16-mixed" if self.accelerator == "gpu" else 32,
            },
        )

        model.fit(
            series=self.train_targets,
            past_covariates=self.train_covs,
            val_series=self.val_targets,
            val_past_covariates=self.val_covs,
            epochs=args.epochs,
            dataloader_kwargs={"batch_size": args.batch_size},
            verbose=True,
        )

        best_val = early_stop.best_score
        return float(best_val.item() if isinstance(best_val, torch.Tensor) else best_val)

    def optimize(self) -> Tuple[optuna.Study, Dict]:
        args = self.args
        timeout_sec = args.optuna_timeout_sec if args.optuna_timeout_sec is not None else None
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=args.pruner_startup_trials,
                n_warmup_steps=args.pruner_warmup_steps,
            ),
        )
        study.optimize(self._objective, n_trials=args.trials, timeout=timeout_sec)
        return study, study.best_params

    def save_trials(self, study: optuna.Study) -> pd.DataFrame:
        args = self.args
        trial_csv_path = Path(args.trial_csv)
        trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_trials = study.trials_dataframe(attrs=("number", "value", "state", "duration"))
        df_params = pd.DataFrame([t.params for t in study.trials])
        df_out = pd.concat([df_trials, df_params], axis=1)
        df_out.to_csv(trial_csv_path, index=False)
        print(f"Saved trial history to {trial_csv_path}")
        return df_out
