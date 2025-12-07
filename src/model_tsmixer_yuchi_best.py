import argparse
import pathlib
import pickle
from typing import List, Optional
import time
import numpy as np
import torch
from torch import nn
import darts
from darts.models import TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_lightning.profiler import AdvancedProfiler

# 若有 CUDA 可用，設定更高的矩陣運算精度以提升 Tensor Cores 效能
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


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
    Parse command-line arguments for training the TSMixer model.

    Returns
    -------
    argparse.Namespace
        Parsed arguments accessible as attributes.
    """
    parser = argparse.ArgumentParser(description="Train a Darts TSMixer model with configurable hyperparameters.")
    # 資料與模型路徑
    parser.add_argument("--data",default="data/ts_data.pkl",help="Path to dataset pickle from dataset_builder.py",)
    parser.add_argument("--model_path",default="models/tsmixer.pth",help="Path to save the trained model.",)
    parser.add_argument("--log_dir",default="logs",help="Directory where TensorBoard logs and checkpoints are stored.",)
    # 訓練參數
    parser.add_argument("--epochs",type=int,default=200,help="Number of training epochs (overridden by early stopping if enabled).",)
    parser.add_argument("--batch_size",type=int,default=32,help="Batch size for the dataloader.",)
    parser.add_argument("--seed",type=int,default=42,help="Random seed for reproducibility.",)
    parser.add_argument("--lambda_weight",type=float,default=0.01,help="Lambda coefficient for the weighted loss combining the true and GARCH targets.",)
    parser.add_argument("--early_stopping_patience",type=int,default=10,help="Number of epochs with no improvement after which training will be stopped.",)
    # 模型架構參數
    parser.add_argument("--input_chunk_length",type=int,default=None,help="Override input_chunk_length (defaults to dataset metadata).",)
    parser.add_argument("--hidden_size",type=int,default=32,help="Hidden state size of the TSMixer.",)
    parser.add_argument("--ff_size",type=int,default=64,help="Size of the first feed-forward layer in the feature mixing MLP.",)
    parser.add_argument("--num_blocks",type=int,default=4,help="Number of mixer blocks in the TSMixer architecture.",)
    parser.add_argument("--dropout",type=float,default=0.1,help="Dropout probability for regularisation within mixer blocks.",) 
    # 優化器與學習率排程器
    parser.add_argument("--lr",type=float,default=3e-4,help="Initial learning rate for the Adam optimizer.",)
    parser.add_argument("--lr_scheduler",choices=["none", "exponential"],default="exponential",help="Learning rate scheduler type.",)
    parser.add_argument("--lr_gamma",type=float,default=0.99,help="Decay factor gamma for the exponential learning rate scheduler.",)
    parser.add_argument("--grad_clip",type=float,default=0.5,help="Clip value for gradient norm to prevent exploding gradients.",)
    # 資料處理選項
    parser.add_argument("--covariate_mode",choices=["none", "lagged"],default="none",help="訓練時要不要使用協變數。預設 none（不使用 alpha 及營收資料訓練）",)
    parser.add_argument("--combine_train_val",action="store_true",help="If set, combine train and validation sets for final training.",)
    # 其他選項
    parser.add_argument("--save_checkpoints",action="store_true",help="Enable model checkpointing during training.",)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu","mps"],
        default="auto",
        required=True,
        help="Device to use for training: auto / cpu / gpu / mps",
    )
    parser.add_argument("--verbose",action="store_true",default=2,help="Enable verbose output during training.",)
    return parser.parse_args()


def _prepare_covariates(covs: List[Optional[object]]) -> List[Optional[object]]:
    """準備協變數資料，若為 None 則返回空列表"""
    return covs if covs is not None else []


def _cast_series_list(series_list: List[Optional[object]]) -> List[Optional[object]]:
    """將 TimeSeries 轉成 float32 避免 BF16 混合精度時的型別衝突"""
    casted = []
    for ts in series_list:
        if ts is None:
            casted.append(None)
        else:
            casted.append(ts.astype(np.float32))
    return casted


def main() -> None:
    args = parse_args()
    
    # 設定隨機種子與預設資料型別
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    
    # 載入資料集
    print(f"Loading dataset from {args.data}...")
    with open(args.data, "rb") as f:
        dataset = pickle.load(f)
    
    # 讀取並轉換資料
    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets = _cast_series_list(dataset["val"]["target"])
    train_covs = _cast_series_list(_prepare_covariates(dataset["train"]["cov"]))
    val_covs = _cast_series_list(_prepare_covariates(dataset["val"]["cov"]))
    
    # 取得資料集設定
    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)
    use_static = dataset.get("static_mode", "none") != "none"
    
    # 決定是否合併訓練集和驗證集
    if args.combine_train_val:
        print("Combining train and validation sets for training...")
        combined_targets = []
        combined_covs = []
        
        for idx, ts in enumerate(train_targets):
            full_ts = ts
            full_cov = train_covs[idx] if idx < len(train_covs) else None
            
            # 合併驗證集資料
            if idx < len(val_targets) and val_targets[idx] is not None:
                full_ts = full_ts.append(val_targets[idx])
                if full_cov is not None and idx < len(val_covs) and val_covs[idx] is not None:
                    full_cov = full_cov.append(val_covs[idx])
                elif full_cov is None and idx < len(val_covs) and val_covs[idx] is not None:
                    full_cov = val_covs[idx]
            
            combined_targets.append(full_ts)
            combined_covs.append(full_cov)
        
        fit_targets = combined_targets
        fit_covs = combined_covs
        fit_val_targets = None
        fit_val_covs = None
    else:
        fit_targets = train_targets
        fit_covs = train_covs
        fit_val_targets = val_targets
        fit_val_covs = val_covs
    
    # # 自動偵測裝置：CUDA > MPS > CPU
    # if torch.cuda.is_available():
    #     accelerator = "gpu"
    #     devices = "auto"
    #     precision = "bf16-mixed"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     accelerator = "mps"
    #     devices = "auto"
    #     precision = 32
    # else:
    #     accelerator = "cpu"
    #     devices = 1
    #     precision = 32
    # 選擇裝置：使用 args.device 覆蓋自動偵測
    if args.device == "cpu":
        accelerator = "cpu"
        devices = 1
        precision = 32
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available.")
        accelerator = "gpu"
        devices = 1   # 為了公平比較，先固定單卡
        precision = "bf16-mixed"
    elif args.device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but is not available.")
        accelerator = "mps"
        devices = "auto"
        precision = 32
    else:  # auto
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
            precision = "bf16-mixed"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
            devices = "auto"
            precision = 32
        else:
            accelerator = "cpu"
            devices = 1
            precision = 32

    
    print(f"Using accelerator: {accelerator}, precision: {precision}")
    
    # 設定學習率排程器
    if args.lr_scheduler == "exponential":
        lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        lr_scheduler_kwargs = {"gamma": args.lr_gamma}
    else:
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None
    
    # 設定回調函數
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=args.early_stopping_patience,
        min_delta=1e-5,
        mode="min",
    )
    
    callbacks = [early_stopper]
    
    # 嘗試加入進度條（如果可用）
    try:
        from darts.utils.callbacks import TFMProgressBar
        progress_bar = TFMProgressBar(
            enable_sanity_check_bar=False,
            enable_validation_bar=False
        )
        callbacks.append(progress_bar)
    except ImportError:
        pass
    
    # PyTorch Lightning 訓練器參數
    pl_trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "default_root_dir": args.log_dir,
        "enable_progress_bar": True,
        "callbacks": callbacks,
        "gradient_clip_val": args.grad_clip,
        "max_epochs": args.epochs,
        "precision": precision,
        "profiler": AdvancedProfiler(filename="pl_profiler.txt"),
    }

    
    # 建立模型
    print(f"Creating TSMixer model with:")
    print(f"  - input_chunk_length: {input_chunk_length}")
    print(f"  - hidden_size: {args.hidden_size}")
    print(f"  - ff_size: {args.ff_size}")
    print(f"  - num_blocks: {args.num_blocks}")
    print(f"  - dropout: {args.dropout}")
    print(f"  - learning_rate: {args.lr}")
    print(f"  - lambda_weight: {args.lambda_weight}")
    
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
        save_checkpoints=args.save_checkpoints,
        force_reset=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )
    
    # 檢查是否有驗證集
    has_val = fit_val_targets is not None and any(ts is not None for ts in fit_val_targets)
    has_val_cov = fit_val_covs is not None and any(cov is not None for cov in fit_val_covs)
    
    # 訓練模型
    print("\nStarting training...")
    
    activities = [ProfilerActivity.CPU]
    if accelerator == "gpu":
        activities.append(ProfilerActivity.CUDA)

    fit_kwargs = {
        "series": fit_targets,
        "val_series": fit_val_targets if has_val else None,
        "epochs": args.epochs,
        "dataloader_kwargs": {"batch_size": args.batch_size},
        "verbose": args.verbose,
    }

    if args.covariate_mode == "lagged":
        # Using covariates（alpha + feature）to train
        fit_kwargs["past_covariates"] = fit_covs
        fit_kwargs["val_past_covariates"]=fit_val_covs if has_val_cov else None
        
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        start_time = time.perf_counter()

        model.fit(**fit_kwargs)

        end_time = time.perf_counter()
    # ===== Profiler end =====

    # 儲存模型
    elapsed = end_time - start_time
    print(f"\nTraining finished in {elapsed:.2f} seconds on device={accelerator}")

    model_path = pathlib.Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel trained and saved to {model_path}")
    # 在 CPU / GPU 上印出最慢的 20 個 op
    print("\n=== Top 30 ops by CUDA time (if any) ===")
    try:
        print(prof.key_averages().table(
            sort_by="cuda_time_total" if accelerator == "gpu" else "cpu_time_total",
            row_limit=30,
        ))
    except Exception as e:
        print(f"Profiler table error: {e}")
    prof.export_chrome_trace("tsmixer_profile.json")
    print("Chrome trace exported to tsmixer_profile.json (open with chrome://tracing)")


if __name__ == "__main__":
    main()
