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
    parser = argparse.ArgumentParser(description="Deep ensemble for TSMixer using best params from trial CSV.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per ensemble member.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Lambda for WeightedLoss.")
    parser.add_argument("--input_chunk_length", type=int, default=None, help="Optional override for input_chunk_length.")
    parser.add_argument("--log_dir", default="logs", help="Lightning log dir.")
    parser.add_argument("--output_model", default="models/DeepEnsemble/tsmixer_optuna.pth", help="Base path to save ensemble models.")
    parser.add_argument("--trial_csv", default="outputs/optuna_lr_trials.csv", help="Optuna trial CSV to read best params.")
    parser.add_argument("--ensemble_runs", type=int, default=10, help="Number of seeds/models to train.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience.")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient clipping value (same as model_train.py).")
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

    # Load best hyperparameters from trial CSV
    trial_csv_path = Path(args.trial_csv)
    if not trial_csv_path.exists():
        raise FileNotFoundError(f"Trial CSV not found at {trial_csv_path}")
    df_out = pd.read_csv(trial_csv_path)
    if "state" in df_out.columns:
        df_out = df_out[df_out["state"] == "COMPLETE"]
    if df_out.empty:
        raise ValueError("No COMPLETE trials found in CSV.")
    best_row = df_out.nsmallest(1, "value").iloc[0]
    best_lr = float(best_row.get("params_lr", best_row.get("lr", 1e-3)))
    best_dropout = float(best_row.get("params_dropout", best_row.get("dropout", 0.1)))
    best_hidden_size = int(best_row.get("params_hidden_size", best_row.get("hidden_size", 32)))
    best_ff_size = int(best_row.get("params_ff_size", best_row.get("ff_size", 64)))
    best_num_blocks = int(best_row.get("params_num_blocks", best_row.get("num_blocks", 2)))
    print(
        f"Loaded best params from CSV: lr={best_lr}, dropout={best_dropout}, "
        f"hidden_size={best_hidden_size}, ff_size={best_ff_size}, num_blocks={best_num_blocks}"
    )

    # Train ensemble models with different seeds using the loaded params
    base_path = Path(args.output_model)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.ensemble_runs):
        seed_i = args.seed + i
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)

        model_path = base_path.with_name(f"{base_path.stem}_{i+1}{base_path.suffix}")
        model_name = f"TSMixer_seed_{i+1}"

        pl_trainer_kwargs = {
            "accelerator": accelerator,
            "devices": devices,
            "default_root_dir": args.log_dir,
            "enable_progress_bar": False,
            "gradient_clip_val": args.grad_clip,
            "precision": "bf16-mixed" if accelerator == "gpu" else 32,
            "callbacks": [EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")],
        }

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
            random_state=seed_i,
            optimizer_kwargs={"lr": best_lr},
            add_encoders=None,
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name=model_name,
            force_reset=True,
            save_checkpoints=True,
        )

        final_model.fit(
            series=train_targets,
            past_covariates=train_covs,
            val_series=val_targets,
            val_past_covariates=val_covs,
            epochs=args.epochs,
            dataloader_kwargs={"batch_size": args.batch_size},
            verbose=True,
        )

        final_model = TSMixerModel.load_from_checkpoint(model_name=model_name, best=True)
        final_model.save(str(model_path))
        print(f"Saved ensemble model {i+1} to {model_path}")


if __name__ == "__main__":
    main()
