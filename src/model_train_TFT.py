import argparse
import pathlib
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from darts.models import TFTModel


class WeightedLoss(nn.Module):
    """Two-component loss: true vs. prediction and GARCH vs. prediction."""

    def __init__(self, lambda_weight: float):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        pred_vol = pred[..., 0]
        true_vol = target[..., 0]
        garch_vol = target[..., 1]
        loss_true = self.mse(pred_vol, true_vol)
        loss_garch = self.mse(pred_vol, garch_vol)
        return self.lambda_weight * loss_true + (1 - self.lambda_weight) * loss_garch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFT model with custom loss.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--lambda", dest="lambda_weight", type=float, default=0.6, help="Lambda for WeightedLoss.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_dir", default="logs", help="Work directory for TensorBoard logs.")
    parser.add_argument("--model_path", default="models/tft_model.pth", help="Path to save the trained model.")
    parser.add_argument(
        "--input_chunk_length",
        type=int,
        default=None,
        help="Override input_chunk_length (defaults to dataset metadata).",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.5,
        help=(
            "Clip value for gradient norm to prevent exploding gradients.  Gradient "
            "clipping is recommended in PyTorch Lightning to stabilise training and "
            "avoid large parameter updates; the Lightning documentation shows how "
            "setting `gradient_clip_val` to 0.5 limits the gradient norm【895475883760702†L125-L150】."
        ),
    )
    parser.add_argument("--hidden_size", type=int, default=64, help="TFT hidden size.")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    return parser.parse_args()

def _prepare_covariates(covs: List[Optional[object]]) -> List[Optional[object]]:
    return covs if covs is not None else []

# 讀取資料集後，對 TimeSeries 轉成 float32
def _cast_series_list(series_list):
    casted = []
    for ts in series_list:
        if ts is None:
            casted.append(None)
        else:
            # 轉型為 float32，避免 BF16 混合精度時出現 Double/BFloat16 衝突
            casted.append(ts.astype(np.float32))
    return casted


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets   = _cast_series_list(dataset["val"]["target"])
    train_covs    = _cast_series_list(_prepare_covariates(dataset["train"]["cov"]))
    val_covs      = _cast_series_list(_prepare_covariates(dataset["val"]["cov"]))

    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)
    use_static = dataset.get("static_mode", "none") != "none"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = "auto" if accelerator == "gpu" else 1

    if args.lr_scheduler == "exponential":
        lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        lr_scheduler_kwargs = {"gamma": args.lr_gamma}
    else:
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None


    from pytorch_lightning.callbacks import EarlyStopping

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        mode="min",
    )

    # Progress bar callback for PyTorch Lightning. Enabling the progress bar
    # provides feedback during training without cluttering the console.
    # If the TFMProgressBar is unavailable, a default progress bar will be used.
    try:
        from darts.utils.callbacks import TFMProgressBar  # type: ignore
        progress_bar = TFMProgressBar(
            enable_sanity_check_bar=False, enable_validation_bar=False
        )
        callbacks = [early_stopper, progress_bar]
    except Exception:
        callbacks = [early_stopper]

    # Build PyTorch Lightning trainer configuration. Gradient clipping is
    # activated via `gradient_clip_val` to mitigate gradient explosion【462202303965851†L260-L263】.
    pl_trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "default_root_dir": args.log_dir,
        "enable_progress_bar": True,
        "callbacks": callbacks,
        "gradient_clip_val": args.grad_clip,
        "max_epochs": args.epochs,
        "precision" : "bf16-mixed",
        }

    # Instantiate the TSMixer model with configurable hyperparameters.  The
    # `add_encoders` argument adds cyclic encodings of the day of the week and
    # month as future covariates, which helps the model capture calendar
    # seasonality【462202303965851†L279-L283】.  Hidden size, feed‑forward size,
    # number of blocks and dropout are exposed via command‑line arguments.
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        loss_fn=WeightedLoss(args.lambda_weight),
        batch_size=args.batch_size,
        use_static_covariates=use_static,
        random_state=args.seed,
        log_tensorboard=True,
        optimizer_kwargs={"lr": args.learning_rate},
        pl_trainer_kwargs=pl_trainer_kwargs
    )

    has_val = any(ts is not None for ts in val_targets)
    has_val_cov = any(cov is not None for cov in val_covs)

    model.fit(
        series=train_targets,
        past_covariates=train_covs,
        val_series=val_targets if has_val else None,
        val_past_covariates=val_covs if has_val_cov else None,
        epochs=args.epochs,
        verbose=False,
    )

    model_path = pathlib.Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"TFT model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
