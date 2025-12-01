import argparse
import pathlib
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from darts.models import TSMixerModel


class WeightedLoss(nn.Module):
    """Two-component loss: true vs. prediction and GARCH vs. prediction."""

    def __init__(self, lambda_weight: float):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [batch, time, components]
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        pred_vol = pred[..., 0]
        true_vol = target[..., 0]
        garch_vol = target[..., 1]
        loss_true = self.mse(pred_vol, true_vol)
        loss_garch = self.mse(pred_vol, garch_vol)
        return self.lambda_weight * loss_true + (1 - self.lambda_weight) * loss_garch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TSMixer with RevIN and static covariates.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--lambda", dest="lambda_weight", type=float, default=0.01, help="Lambda for WeightedLoss.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_dir", default="logs", help="Work directory for TensorBoard logs.")
    parser.add_argument("--model_path", default="models/tsmixer.pth", help="Path to save the trained model.")
    parser.add_argument(
        "--input_chunk_length",
        type=int,
        default=None,
        help="Override input_chunk_length (defaults to dataset metadata).",
    )
    return parser.parse_args()


def _prepare_covariates(covs: List[Optional[object]]) -> List[Optional[object]]:
    return covs if covs is not None else []


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    train_targets = dataset["train"]["target"]
    val_targets = dataset["val"]["target"]
    train_covs = _prepare_covariates(dataset["train"]["cov"])
    val_covs = _prepare_covariates(dataset["val"]["cov"])

    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)
    use_static = dataset.get("static_mode", "none") != "none"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if accelerator == "cpu" else "auto"

    model = TSMixerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        use_static_covariates=use_static,
        use_reversible_instance_norm=True,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        log_tensorboard=True,
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": devices,
            "default_root_dir": args.log_dir,
            "enable_progress_bar": False,
        },
    )

    has_val = any(ts is not None for ts in val_targets)
    has_val_cov = any(cov is not None for cov in val_covs)

    model.fit(
        series=train_targets,
        past_covariates=train_covs,
        val_series=val_targets if has_val else None,
        val_past_covariates=val_covs if has_val_cov else None,
        epochs=args.epochs,
        dataloader_kwargs={"batch_size": args.batch_size},
        verbose=False,
    )

    model_path = pathlib.Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
