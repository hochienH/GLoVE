import argparse
import pathlib
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from darts.models import TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping

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
    """
    Parse command‑line arguments for training the TSMixer model.

    In addition to the original arguments, this parser exposes several
    hyperparameters that can materially affect model performance. These
    include the learning rate and scheduler parameters, gradient clipping,
    and the internal architecture of the TSMixer (hidden size, feed‑forward
    size, number of mixer blocks and dropout). Exposing these values as
    arguments makes it easier to experiment with different settings without
    modifying the code directly.

    Returns
    -------
    argparse.Namespace
        Parsed arguments accessible as attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train a Darts TSMixer model with configurable hyperparameters."
    )
    parser.add_argument(
        "--data",
        default="Dataset/ts_data.pkl",
        help="Path to dataset pickle from dataset_builder.py",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_weight",
        type=float,
        default=0.01,
        help="Lambda coefficient for the weighted loss combining the true and GARCH targets.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (overridden by early stopping if enabled).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="Directory where TensorBoard logs and checkpoints are stored.",
    )
    parser.add_argument(
        "--model_path",
        default="models/tsmixer.pth",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--input_chunk_length",
        type=int,
        default=None,
        help="Override input_chunk_length (defaults to dataset metadata).",
    )
    # Hyper‑parameter tuning arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help=(
            "Initial learning rate for the Adam optimizer.  Empirical studies on the "
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        choices=["none", "exponential"],
        default="exponential",
        help="Learning rate scheduler type. 'exponential' applies ExponentialLR; 'none' keeps the learning rate constant.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.99,
        help="Decay factor gamma for the exponential learning rate scheduler.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.5,
        help=(
            "Clip value for gradient norm to prevent exploding gradients. "
        ),
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=32,
        help=(
            "Hidden state size of the TSMixer (dimension of the second feed‑forward layer in the feature mixing MLP). "
            "The official TSMixer paper uses a hidden feature dimension of 32 and an expansion (ff) size of 64 with "
        ),
    )
    parser.add_argument(
        "--ff_size",
        type=int,
        default=64,
        help=(
            "Size of the first feed‑forward layer in the feature mixing MLP.  The expansion feature dimension is twice "
        ),
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=4,
        help=(
            "Number of mixer blocks in the TSMixer architecture"
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    return parser.parse_args()


# 讀取資料集後，對 TimeSeries 轉成 float32
def _cast_series_list(series_list):
    casted = []
    for ts in series_list:
        casted.append(ts.astype(np.float32) if ts is not None else None)
    return casted

def main() -> None:

    # 在 main 函式裡，設定 PyTorch 預設 dtype
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)

    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets   = _cast_series_list(dataset["val"]["target"])
    train_covs    = _cast_series_list(dataset["train"]["cov"])
    val_covs      = _cast_series_list(dataset["val"]["cov"])

    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)
    use_static = dataset.get("static_mode", "none") != "none"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if accelerator == "cpu" else "auto"

    if args.lr_scheduler == "exponential":
        lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        lr_scheduler_kwargs = {"gamma": args.lr_gamma}
    else:
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None


    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-5,
        mode="min",
    )

    try:
        from darts.utils.callbacks import TFMProgressBar  # type: ignore
        progress_bar = TFMProgressBar(
            enable_sanity_check_bar=False, enable_validation_bar=False
        )
        callbacks = [early_stopper, progress_bar]
    except Exception:
        callbacks = [early_stopper]

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

    model = TSMixerModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        hidden_size=args.hidden_size,
        ff_size=args.ff_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        use_static_covariates=use_static,
        use_reversible_instance_norm=True,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        log_tensorboard=True,
        optimizer_kwargs={"lr": args.lr},
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        add_encoders={"cyclic": {"future": ["dayofweek"]}},
        pl_trainer_kwargs=pl_trainer_kwargs,
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
