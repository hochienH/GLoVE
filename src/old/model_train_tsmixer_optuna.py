import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, Callback

from darts import TimeSeries
from darts.models import TSMixerModel
from model_train import WeightedLoss  # reuse the same custom loss

# Prefer Tensor Cores when available for better throughput
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

class PatchedPruningCallback(PyTorchLightningPruningCallback, Callback):
    """Compatibility shim for Lightning pruning callbacks."""
    pass


def _cast_series_list(series_list: List[Optional[TimeSeries]]) -> List[Optional[TimeSeries]]:
    casted: List[Optional[TimeSeries]] = []
    for ts in series_list:
        casted.append(ts.astype(np.float32) if ts is not None else None)
    return casted


def _mean_weighted_loss(
    preds: np.ndarray, true_vals: np.ndarray, garch_vals: np.ndarray, lambda_weight: float
) -> float:
    mse_true = np.mean((preds - true_vals) ** 2)
    mse_garch = np.mean((preds - garch_vals) ** 2)
    return lambda_weight * mse_true + (1.0 - lambda_weight) * mse_garch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuner for TSMixer learning rate.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per trial.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lambda_weight", type=float, default=0.01, help="Lambda for WeightedLoss.")
    parser.add_argument("--lr_min", type=float, default=1e-4, help="Lower bound for lr search (log scale).")
    parser.add_argument("--lr_max", type=float, default=1e-2, help="Upper bound for lr search (log scale).")
    parser.add_argument("--input_chunk_length", type=int, default=None, help="Optional override for input_chunk_length.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Fixed hidden size.")
    parser.add_argument("--ff_size", type=int, default=64, help="Fixed ff_size.")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of mixer blocks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--log_dir", default="logs", help="Lightning log dir.")
    parser.add_argument("--output_model", default="models/tsmixer_optuna.pth", help="Path to save best model.")
    parser.add_argument("--trial_csv", default="outputs/optuna_lr_trials.csv", help="Where to save lr trial history CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets = _cast_series_list(dataset["val"]["target"])
    train_covs = _cast_series_list(dataset["train"]["cov"])
    val_covs = _cast_series_list(dataset["val"]["cov"])

    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)
    use_static = dataset.get("static_mode", "none") != "none"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = "auto" if accelerator == "gpu" else 1

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        num_blocks= trial.suggest_categorical("num_blocks", [1, 2, 4])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        ff_size = trial.suggest_categorical("ff_size", [16, 32, 64])

        pruning_cb = PatchedPruningCallback(trial, monitor="val_loss")
        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        callbacks = [pruning_cb, early_stop]
        model_name = f"tsmixer_trial_{trial.number}"
        model = TSMixerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=1,
            hidden_size=hidden_size,
            ff_size=ff_size,
            dropout=dropout,
            num_blocks=num_blocks,
            use_static_covariates=use_static,
            use_reversible_instance_norm=True,
            loss_fn=WeightedLoss(args.lambda_weight),
            random_state=args.seed,
            optimizer_kwargs={"lr": lr},
            save_checkpoints=False, # 如果是用 Early Stopping 的 Val_loss 就可以關掉
            force_reset=True,
            model_name=model_name,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "devices": devices,
                "default_root_dir": args.log_dir,
                "enable_progress_bar": False,
                "callbacks": callbacks,
                "precision": "bf16-mixed" if accelerator == "gpu" else 32,
            },
        )

        model.fit(
            series=train_targets,
            past_covariates=train_covs,
            val_series=val_targets,
            val_past_covariates=val_covs,
            epochs=args.epochs,
            dataloader_kwargs={"batch_size": args.batch_size},
            verbose=True,
        )

        best_val = early_stop.best_score
        if isinstance(best_val, torch.Tensor):
            best_val = best_val.item()

        return best_val

        ## 以下是利用 Historical Forecast 預測出來的值, 在 Horizon != 1 時可能會有顯著差異，但 Horizon = 1 時應該是沒有差。
        # # reload best checkpoint (tracked on val_loss)
        # model = TSMixerModel.load_from_checkpoint(model_name, best=True)

        # losses = []
        # for idx, val_ts in enumerate(val_targets):
        #     if val_ts is None or len(val_ts) == 0:
        #         continue
        #     train_cov = train_covs[idx] if idx < len(train_covs) else None
        #     val_cov = val_covs[idx] if idx < len(val_covs) else None
        #     if train_cov is None or val_cov is None:
        #         # Need covariates extending through the validation horizon to avoid Darts length errors
        #         continue
        #     cov_full = train_cov.append(val_cov)
        #     target_full = train_targets[idx].append(val_ts)
        #     forecast = model.historical_forecasts(
        #         series=target_full,
        #         past_covariates=cov_full,
        #         start=train_targets[idx].end_time(),
        #         forecast_horizon=1,
        #         stride=1,
        #         last_points_only=True,
        #         verbose=False,
        #         retrain=False,
        #     )
            
        #     forecast_df = forecast.to_dataframe().reindex(val_ts.time_index)
        #     df_val = val_ts.to_dataframe()
        #     pred_vals = forecast_df.iloc[:, 0].to_numpy()
        #     true_vals = df_val.iloc[:, 0].to_numpy()
        #     garch_vals = df_val.iloc[:, 1].to_numpy()
        #     losses.append(_mean_weighted_loss(pred_vals, true_vals, garch_vals, args.lambda_weight))

        # return float(np.mean(losses)) if losses else float("inf")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,   # 前幾個 trial 不 prune（暖身）
            # n_warmup_steps=3,     # 前幾個 epoch 不 prune，TFT/TSMixer 都 OK
        ),
    )
    study.optimize(objective, n_trials=args.trials, timeout=4*60*60)

    best_params = study.best_params

    best_lr = best_params["lr"]
    best_dropout = best_params["dropout"]
    best_hidden_size = best_params["hidden_size"]
    best_ff_size = best_params["ff_size"]
    best_num_blocks = best_params["num_blocks"]
    print(
        f"Best trial: value={study.best_value:.6f}, "
        f"lr={best_lr}, dropout={best_dropout}, "
        f"hidden_size={best_hidden_size}, ff_size={best_ff_size}"
    )

    # Save lr trial history
    trial_csv_path = Path(args.trial_csv)
    trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 先取得基本欄位（不包含 params 本體）
    df_trials = study.trials_dataframe(attrs=("number", "value", "state", "duration"))
    # 取出所有 trial 的 params，展開成 DataFrame
    params_list = [t.params for t in study.trials]
    df_params = pd.DataFrame(params_list)
    # 合併基本資訊 + 超參
    df_out = pd.concat([df_trials, df_params], axis=1)
    df_out.to_csv(trial_csv_path, index=False)
    print(f"Saved trial history to {trial_csv_path}")

    # Train final model on train+val with best lr
    combined_targets = []
    combined_covs = []
    for idx, ts in enumerate(train_targets):
        full_ts = ts
        full_cov = train_covs[idx] if idx < len(train_covs) else None
        if idx < len(val_targets) and val_targets[idx] is not None:
            full_ts = full_ts.append(val_targets[idx])
            if full_cov is not None and idx < len(val_covs) and val_covs[idx] is not None:
                full_cov = full_cov.append(val_covs[idx])
            elif full_cov is None and idx < len(val_covs) and val_covs[idx] is not None:
                full_cov = val_covs[idx]
        combined_targets.append(full_ts)
        combined_covs.append(full_cov)

    final_model = TSMixerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        hidden_size=best_hidden_size,
        ff_size=best_ff_size,
        dropout=best_dropout,
        num_blocks=best_num_blocks,
        use_static_covariates=use_static,
        use_reversible_instance_norm=True,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        optimizer_kwargs={"lr": best_lr},
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": devices,
            "default_root_dir": args.log_dir,
            "enable_progress_bar": False,
            "precision": "bf16-mixed" if accelerator == "gpu" else 32,
        },
        model_name="TSMixer",
        force_reset=True,
        save_checkpoints=True,
    )

    final_model.fit(
        series=combined_targets,
        past_covariates=combined_covs,
        epochs=args.epochs,
        dataloader_kwargs={"batch_size": args.batch_size},
        verbose=False,
    )

    final_model = TSMixerModel.load_from_checkpoint(model_name="TSMixer", best=True)
    final_model.save(args.output_model)
    print(f"Saved best model to {args.output_model}")

if __name__ == "__main__":
    main()
