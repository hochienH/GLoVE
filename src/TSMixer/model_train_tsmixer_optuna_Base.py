import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts.models import TSMixerModel
from model_train import WeightedLoss
from tsmixer_optuna_runner import TSMixerOptunaRunner


def _cast_series_list(series_list):
    casted = []
    for ts in series_list:
        casted.append(ts.astype(np.float32) if ts is not None else None)
    return casted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuner for TSMixer (Base, reusable).")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per trial.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Lambda for WeightedLoss.")
    parser.add_argument("--lr_min", type=float, default=1e-4, help="Lower bound for lr search (log scale).")
    parser.add_argument("--lr_max", type=float, default=1e-2, help="Upper bound for lr search (log scale).")
    parser.add_argument("--search_num_blocks", nargs="+", type=int, default=[1, 2, 4], help="Search space for num_blocks.")
    parser.add_argument("--search_dropout", nargs="+", type=float, default=[0.1, 0.2, 0.3], help="Search space for dropout.")
    parser.add_argument("--search_hidden_size", nargs="+", type=int, default=[16, 32, 64], help="Search space for hidden_size.")
    parser.add_argument("--search_ff_size", nargs="+", type=int, default=[16, 32, 64], help="Search space for ff_size.")
    parser.add_argument("--input_chunk_length", type=int, default=None, help="Optional override for input_chunk_length.")
    parser.add_argument("--optuna_timeout_sec", type=float, default=14400, help="Optuna search timeout in seconds (None to disable).")
    parser.add_argument("--log_dir", default="logs", help="Lightning log dir.")
    parser.add_argument("--output_model", default="models/base/tsmixer.pth", help="Path to save best model.")
    parser.add_argument("--trial_csv", default="outputs/optuna_lr_trials.csv", help="Where to save trial history CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--pruner_startup_trials", type=int, default=10, help="Optuna MedianPruner n_startup_trials.")
    parser.add_argument("--pruner_warmup_steps", type=int, default=0, help="Optuna MedianPruner n_warmup_steps (epochs).")
    parser.add_argument("--patience", type=int, default=5, help="EarlyStopping patience.")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient clipping value (same as model_train.py).")
    parser.add_argument("--covariate_mode", choices=["none", "alpha"], default="none", help="訓練時要不要使用協變數。預設 none（不使用alpha及營收資料訓練）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(Path(args.data), "rb") as f:
        dataset = pickle.load(f)

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets = _cast_series_list(dataset["val"]["target"])
    train_covs = _cast_series_list(dataset["train"]["cov"])
    val_covs = _cast_series_list(dataset["val"]["cov"])
    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)
    use_static = dataset.get("static_mode", "none") != "none"

    runner = TSMixerOptunaRunner(
        args,
        train_targets=train_targets,
        val_targets=val_targets,
        train_covs=train_covs,
        val_covs=val_covs,
        input_chunk_length=input_chunk_length,
        use_static=use_static,
        covariate_mode=args.covariate_mode
    )
    study, best_params = runner.optimize()
    runner.save_trials(study)

    # Final training with train/val using EarlyStopping
    pl_trainer_kwargs = {
        "accelerator": runner.accelerator,
        "devices": runner.devices,
        "default_root_dir": args.log_dir,
        "enable_progress_bar": False,
        "gradient_clip_val": args.grad_clip,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")],
        "precision": "bf16-mixed" if runner.accelerator == "gpu" else 32,
    }

    final_model = TSMixerModel(
        input_chunk_length=runner.input_chunk_length,
        output_chunk_length=1,
        hidden_size=best_params["hidden_size"],
        ff_size=best_params["ff_size"],
        dropout=best_params["dropout"],
        num_blocks=best_params["num_blocks"],
        use_static_covariates=runner.use_static,
        use_reversible_instance_norm=True,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        optimizer_kwargs={"lr": best_params["lr"]},
        add_encoders=None,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="TSMixer_final",
        force_reset=True,
        save_checkpoints=True,
    )

    if runner.covariate_mode == "alpha":
        past_covs = runner.train_covs
        val_past_covs = runner.val_covs
    else:
        past_covs = None
        val_past_covs = None

    final_model.fit(
        series=runner.train_targets,
        past_covariates=past_covs,
        val_series=runner.val_targets,
        val_past_covariates=val_past_covs,
        epochs=args.epochs,
        dataloader_kwargs={"batch_size": args.batch_size},
        verbose=True,
    )
    final_model = TSMixerModel.load_from_checkpoint(model_name="TSMixer_final", best=True)
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    final_model.save(args.output_model)
    print(f"Saved best model to {args.output_model}")


if __name__ == "__main__":
    main()
