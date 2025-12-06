import argparse
import pathlib
import pickle
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from darts import TimeSeries
from darts.models import RNNModel


class WeightedLoss(nn.Module):
    """兩段式損失：模型輸出同時貼近真實波動與 GARCH 波動。"""

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
    parser = argparse.ArgumentParser(
        description="Train a Darts LSTM (RNNModel) with the weighted volatility loss."
    )
    parser.add_argument(
        "--data",
        default="Dataset/ts_data.pkl",
        help="dataset_builder.py 輸出的 TimeSeries pickle 路徑",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_weight",
        type=float,
        default=0.01,
        help="True vs. GARCH loss 的權重：lambda*true + (1-lambda)*garch。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="訓練 epoch 數，若啟用 early stopping 會提前停止。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="DataLoader 批次大小。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子。",
    )
    parser.add_argument(
        "--log_dir",
        default="logs",
        help="TensorBoard 與 checkpoint 目錄。",
    )
    parser.add_argument(
        "--model_path",
        default="models/lstm.pth",
        help="儲存訓練後模型的路徑。",
    )
    parser.add_argument(
        "--input_chunk_length",
        type=int,
        default=None,
        help="輸入長度，預設沿用 dataset metadata (通常為 90)。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="AdamW 學習率。",
    )
    parser.add_argument(
        "--lr_scheduler",
        choices=["none", "exponential"],
        default="exponential",
        help="學習率排程器。",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.99,
        help="ExponentialLR 衰減係數。",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.5,
        help="梯度裁剪上限。",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="LSTM 每層 hidden dimension。",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="LSTM 疊層數。",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="層間 dropout。",
    )
    parser.add_argument(
        "--covariate_mode",
        choices=["none", "lagged"],
        default="none",
        help="LSTM 在訓練時要不要使用協變數。預設 none（完全不用，以免洩漏）；"
        "選 lagged 會把 cov 退後 lag 天再當 future cov 使用，只提供歷史資訊。",
    )
    parser.add_argument(
        "--covariate_lag",
        type=int,
        default=1,
        help="lagged 模式下要延後幾天的協變數（預設 1，表示用 t-1 的 cov 預測 t）。",
    )
    return parser.parse_args()


def _prepare_covariates(covs: List[Optional[object]]) -> List[Optional[object]]:
    return covs if covs is not None else []


def _build_lagged_covariates(
    series_list: List[Optional[TimeSeries]], lag: int
) -> List[Optional[TimeSeries]]:
    lagged: List[Optional[TimeSeries]] = []
    for ts in series_list:
        if ts is None or lag <= 0:
            lagged.append(ts)
            continue
        values = ts.values(copy=True)
        if values.shape[0] <= lag:
            lagged.append(None)
            continue
        lagged_vals = np.zeros_like(values)
        lagged_vals[lag:] = values[:-lag]
        lagged_vals[:lag] = 0.0
        lagged.append(ts.with_values(lagged_vals))
    return lagged


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)

    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    def _cast_series_list(series_list):
        casted = []
        for ts in series_list:
            if ts is None:
                casted.append(None)
            else:
                casted.append(ts.astype(np.float32))
        return casted

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets = _cast_series_list(dataset["val"]["target"])
    train_covs = _cast_series_list(_prepare_covariates(dataset["train"]["cov"]))
    val_covs = _cast_series_list(_prepare_covariates(dataset["val"]["cov"]))

    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)

    # 裝置自動偵測：
    # - 若有 CUDA，使用 GPU
    # - 否則若有 Apple Silicon MPS，使用 MPS
    # - 以上皆無則退回 CPU
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = "auto"
    else:
        accelerator = "cpu"
        devices = 1

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

    try:
        from darts.utils.callbacks import TFMProgressBar  # type: ignore

        progress_bar = TFMProgressBar(
            enable_sanity_check_bar=False, enable_validation_bar=False
        )
        callbacks = [early_stopper, progress_bar]
    except Exception:
        callbacks = [early_stopper]

    if accelerator == "cpu":
        precision = 32
    else:
        precision = "bf16-mixed"

    pl_trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "default_root_dir": args.log_dir,
        "enable_progress_bar": True,
        "callbacks": callbacks,
        "gradient_clip_val": args.grad_clip,
        "max_epochs": args.epochs,
        "precision": precision,
    }

    model = RNNModel(
        model="LSTM",
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        hidden_dim=args.hidden_size,
        n_rnn_layers=args.num_layers,
        dropout=args.dropout,
        training_length=input_chunk_length,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        log_tensorboard=True,
        optimizer_kwargs={"lr": args.lr},
        optimizer_cls=torch.optim.AdamW,
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        add_encoders={"cyclic": {"future": ["dayofweek"]}},
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    has_val = any(ts is not None for ts in val_targets)

    future_covariates = None
    val_future_covariates = None
    if args.covariate_mode == "lagged":
        future_covariates = _build_lagged_covariates(train_covs, args.covariate_lag)
        val_future_covariates = (
            _build_lagged_covariates(val_covs, args.covariate_lag) if has_val else None
        )

    fit_kwargs = {
        "series": train_targets,
        "val_series": val_targets if has_val else None,
        "epochs": args.epochs,
        "dataloader_kwargs": {"batch_size": args.batch_size},
        "verbose": False,
    }
    if future_covariates is not None:
        fit_kwargs["future_covariates"] = future_covariates
    if val_future_covariates is not None:
        fit_kwargs["val_future_covariates"] = val_future_covariates

    model.fit(**fit_kwargs)

    model_path = pathlib.Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"LSTM model trained and saved to {model_path}")


if __name__ == "__main__":
    main()

