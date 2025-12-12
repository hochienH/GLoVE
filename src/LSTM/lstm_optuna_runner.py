# lstm_optuna_runner.py
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import numpy as np
import optuna
import pandas as pd
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, Callback

from darts import TimeSeries
from darts.models import RNNModel
from model_train_lstm import WeightedLoss   # 直接沿用你原本的 loss


class PatchedPruningCallback(PyTorchLightningPruningCallback, Callback):
    pass


class LSTMOptunaRunner:
    """
    可重複使用的 LSTM Optuna Runner。
    和你的 TSMixerOptunaRunner 設計完全一致。
    """

    def __init__(
        self,
        args,
        train_targets: List[Optional[TimeSeries]],
        val_targets: List[Optional[TimeSeries]],
        train_covs: List[Optional[TimeSeries]],
        val_covs: List[Optional[TimeSeries]],
        input_chunk_length: int,
        covariate_mode: str,
        future_covariates_train,
        future_covariates_val,
    ):
        self.args = args
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.train_covs = train_covs
        self.val_covs = val_covs
        self.input_chunk_length = input_chunk_length
        self.covariate_mode = covariate_mode
        self.future_covariates_train = future_covariates_train
        self.future_covariates_val = future_covariates_val

        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.devices = "auto" if self.accelerator == "gpu" else 1

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # -------------------------
    # Optuna 的 objective
    # -------------------------
    def _objective(self, trial: optuna.Trial) -> float:
        args = self.args

        # 搜尋空間
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", args.search_hidden_dim)
        num_layers = trial.suggest_categorical("num_layers", args.search_num_layers)
        dropout = trial.suggest_categorical("dropout", args.search_dropout)

        pruning_cb = PatchedPruningCallback(trial, monitor="val_loss")
        early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")
        callbacks = [pruning_cb, early_stop]

        precision = "bf16-mixed" if self.accelerator == "gpu" else 32

        pl_trainer_kwargs = {
            "accelerator": self.accelerator,
            "devices": self.devices,
            "default_root_dir": args.log_dir,
            "enable_progress_bar": False,
            "callbacks": callbacks,
            "precision": precision,
            "gradient_clip_val": args.grad_clip,
        }

        # 建立 LSTM model
        model = RNNModel(
            model="LSTM",
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=1,
            hidden_dim=hidden_dim,
            n_rnn_layers=num_layers,
            dropout=dropout,
            training_length=self.input_chunk_length,
            loss_fn=WeightedLoss(args.lambda_weight),
            random_state=args.seed,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": lr},
            pl_trainer_kwargs=pl_trainer_kwargs,
        )

        # 決定是否用 covariates
        fit_kwargs = {
            "series": self.train_targets,
            "val_series": self.val_targets,
            "epochs": args.epochs,
            "dataloader_kwargs": {"batch_size": args.batch_size},
            "verbose": False,
        }

        if self.covariate_mode == "alpha":
            fit_kwargs["future_covariates"] = self.future_covariates_train
            fit_kwargs["val_future_covariates"] = self.future_covariates_val

        # 訓練
        model.fit(**fit_kwargs)

        best_val = early_stop.best_score
        return float(best_val.item() if isinstance(best_val, torch.Tensor) else best_val)

    # -------------------------
    # Optuna optimize
    # -------------------------
    def optimize(self) -> Tuple[optuna.Study, Dict]:
        args = self.args
        timeout_sec = args.optuna_timeout_sec if args.optuna_timeout_sec else None

        study = optuna.create_study(direction="minimize")
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