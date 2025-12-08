import argparse
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts import TimeSeries
from darts.models import TSMixerModel
from model_train import WeightedLoss  # reuse the same custom loss

# Prefer Tensor Cores when available for better throughput
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def _cast_series_list(series_list: List[Optional[TimeSeries]]) -> List[Optional[TimeSeries]]:
    casted: List[Optional[TimeSeries]] = []
    for ts in series_list:
        casted.append(ts.astype(np.float32) if ts is not None else None)
    return casted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Params ensemble for TSMixer using top trials from CSV.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Lambda for WeightedLoss.")
    parser.add_argument("--input_chunk_length", type=int, default=None, help="Optional override for input_chunk_length.")
    parser.add_argument("--log_dir", default="logs", help="Lightning log dir.")
    parser.add_argument("--output_model", default="models/ParamsEnsemble/tsmixer_optuna.pth", help="Base path to save ensemble models.")
    parser.add_argument("--trial_csv", default="outputs/optuna_lr_trials.csv", help="Optuna trial CSV to read params.")
    parser.add_argument("--ensemble_runs", type=int, default=10, help="Number of top parameter sets to train.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience.")
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

    # Load top hyperparameters from trial CSV
    trial_csv_path = Path(args.trial_csv)
    if not trial_csv_path.exists():
        raise FileNotFoundError(f"Trial CSV not found at {trial_csv_path}")
    df_out = pd.read_csv(trial_csv_path)
    # Try COMPLETE first; if不足，再補 PRUNED
    if "state" in df_out.columns:
        complete_df = df_out[df_out["state"] == "COMPLETE"]
        pruned_df = df_out[df_out["state"] == "PRUNED"]
    else:
        complete_df = df_out
        pruned_df = pd.DataFrame()

    top_trials = complete_df.nsmallest(args.ensemble_runs, "value")
    if len(top_trials) < args.ensemble_runs and not pruned_df.empty:
        need = args.ensemble_runs - len(top_trials)
        extra = pruned_df.nsmallest(need, "value")
        top_trials = pd.concat([top_trials, extra], axis=0)

    if top_trials.empty:
        raise ValueError("No COMPLETE or PRUNED trials found in CSV.")

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
        model_name = f"TSMixer_params_{i}"
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
            add_encoders=None,
            pl_trainer_kwargs={
                "accelerator": accelerator,
                "devices": devices,
                "default_root_dir": args.log_dir,
                "enable_progress_bar": False,
                "callbacks": [EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")],
                "precision": "bf16-mixed" if accelerator == "gpu" else 32,
            },
            model_name=model_name,
            force_reset=True,
            save_checkpoints=True,
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

        model = TSMixerModel.load_from_checkpoint(model_name=model_name, best=True)
        model.save(str(model_path))
        print(f"Saved params-ensemble model {i} to {model_path}")


if __name__ == "__main__":
    main()
