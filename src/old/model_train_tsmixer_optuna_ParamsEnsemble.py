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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna + params ensemble for TSMixer.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--trial_csv", default="outputs/optuna_lr_trials.csv", help="Optuna trial CSV with params.")
    parser.add_argument("--ensemble_runs", type=int, default=10, help="Number of top parameter sets to train.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lambda_weight", type=float, default=0.01, help="Lambda for WeightedLoss.")
    parser.add_argument("--lr_min", type=float, default=1e-4, help="Lower bound for lr search (log scale).")
    parser.add_argument("--lr_max", type=float, default=1e-2, help="Upper bound for lr search (log scale).")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of mixer blocks.")
    parser.add_argument("--log_dir", default="logs", help="Lightning log dir.")
    parser.add_argument("--output_model", default="models/tsmixer_optuna.pth", help="Base path to save ensemble models.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--input_chunk_length", type=int, default=None, help="Optional override for input_chunk_length.")
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
            save_checkpoints=False,
            force_reset=True,
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
            verbose=False,
        )

        best_val = early_stop.best_score
        if isinstance(best_val, torch.Tensor):
            best_val = best_val.item()
        return best_val

    # study = optuna.create_study(
    #     direction="minimize",
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=np.minimum(args.ensemble_runs, 10),
    #         n_warmup_steps=0),
    # )

    # study.optimize(objective, n_trials=args.trials, timeout=None)

    # Save trial history with params
    trial_csv_path = Path(args.trial_csv)
    # trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
    # df_trials = study.trials_dataframe(attrs=("number", "value", "state", "duration"))
    # df_params = pd.DataFrame([t.params for t in study.trials])
    # df_out = pd.concat([df_trials, df_params], axis=1)
    # df_out.to_csv(trial_csv_path, index=False)
    # print(f"Saved trial history to {trial_csv_path}")

    df_out = pd.read_csv(trial_csv_path)

    if "state" in df_out.columns:
        df_out = df_out[df_out["state"] == "COMPLETE"]
    if df_out.empty:
        raise ValueError("No COMPLETE trials found to build the ensemble.")

    top_trials = df_out.nsmallest(args.ensemble_runs, "value")

    # Prepare combined train+val
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

    base_path = Path(args.output_model)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(top_trials.iterrows(), start=1):
        seed_i = args.seed + i - 1
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)

        lr = float(row.get("params_lr", row.get("lr", 1e-3)))
        dropout = float(row.get("params_dropout", row.get("dropout", 0.1)))
        hidden_size = int(row.get("params_hidden_size", row.get("hidden_size", 32)))
        ff_size = int(row.get("params_ff_size", row.get("ff_size", 64)))
        num_blocks = int(row.get("params_num_blocks", row.get("num_blocks", 2)))

        model_path = base_path.with_name(f"{base_path.stem}_{i}{base_path.suffix}")
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
            random_state=seed_i,
            optimizer_kwargs={"lr": lr},
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

        model.fit(
            series=combined_targets,
            past_covariates=combined_covs,
            epochs=args.epochs,
            dataloader_kwargs={"batch_size": args.batch_size},
            verbose=False,
        )

        model = TSMixerModel.load_from_checkpoint(model_name="TSMixer", best=True)
        model.save(str(model_path))
        print(f"Saved params-ensemble model {i} to {model_path}")


if __name__ == "__main__":
    main()
