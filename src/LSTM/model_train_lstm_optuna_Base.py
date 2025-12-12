import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping

from darts.models import RNNModel
from model_train_lstm import WeightedLoss
from lstm_optuna_runner import LSTMOptunaRunner


# ---------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuner for LSTM/RNNModel.")
    parser.add_argument("--data", required=True)

    # Optuna搜尋次數
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    # 搜尋空間
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--search_hidden_dim", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--search_num_layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--search_dropout", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])

    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--input_chunk_length", type=int, default=None, help="輸入長度，預設沿用 dataset metadata (通常為 90)。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_scheduler", choices=["none", "exponential"], default="exponential", help="學習率排程器。")
    parser.add_argument("--lr_gamma", type=float, default=0.99, help="ExponentialLR 衰減係數。")
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--output_model", default="models/lstm_optuna.pth")
    
    parser.add_argument("--optuna_timeout_sec", type=float, default=14400)
    parser.add_argument("--trial_csv", default="outputs/lstm_optuna_trials.csv")

    # covariate mode
    parser.add_argument("--covariate_mode", choices=["none", "alpha"], default="none")
    parser.add_argument("--covariate_lag", type=int, default=1)

    return parser.parse_args()


# ---------------------------------------------------------
# Utility：lag covariates
# ---------------------------------------------------------
def build_lagged_covariates(series_list, lag):
    out = []
    for ts in series_list:
        if ts is None:
            out.append(None)
            continue
        v = ts.values(copy=True)
        if len(v) <= lag:
            out.append(None)
            continue
        lagged = np.zeros_like(v)
        lagged[lag:] = v[:-lag]
        out.append(ts.with_values(lagged))
    return out


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --------------------------------------
    # 讀 dataset
    # --------------------------------------
    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    def cast_list(lst):
        return [ts.astype(np.float32) if ts is not None else None for ts in lst]

    train_targets = cast_list(dataset["train"]["target"])
    val_targets = cast_list(dataset["val"]["target"])
    train_covs = cast_list(dataset["train"]["cov"])
    val_covs = cast_list(dataset["val"]["cov"])

    input_chunk_length = args.input_chunk_length or dataset.get("input_chunk_length", 90)

    # --------------------------------------
    # covariates
    # --------------------------------------
    if args.covariate_mode == "alpha":
        future_cov_train = build_lagged_covariates(train_covs, args.covariate_lag)
        future_cov_val = build_lagged_covariates(val_covs, args.covariate_lag)
    else:
        future_cov_train = None
        future_cov_val = None

    # --------------------------------------
    # 呼叫 Optuna Runner
    # --------------------------------------
    if args.lr_scheduler == "exponential":
        lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        lr_scheduler_kwargs = {"gamma": args.lr_gamma}
    else:
        lr_scheduler_cls = None
        lr_scheduler_kwargs = None

    
    runner = LSTMOptunaRunner(
        args,
        train_targets=train_targets,
        val_targets=val_targets,
        train_covs=train_covs,
        val_covs=val_covs,
        input_chunk_length=input_chunk_length,
        covariate_mode=args.covariate_mode,
        future_covariates_train=future_cov_train,
        future_covariates_val=future_cov_val,
    )

    study, best_params = runner.optimize()
    df_out = runner.save_trials(study)

    # --------------------------------------
    # Final training using the best params
    # --------------------------------------
    early_stopper = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    precision = "bf16-mixed" if torch.cuda.is_available() else 32

    pl_trainer_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto" if torch.cuda.is_available() else 1,
        "default_root_dir": args.log_dir,
        "callbacks": [early_stopper],
        "precision": precision,
        "gradient_clip_val": args.grad_clip,
    }

    final_model = RNNModel(
        model="LSTM",
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        hidden_dim=best_params["hidden_dim"],
        n_rnn_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        training_length=input_chunk_length,
        loss_fn=WeightedLoss(args.lambda_weight),
        random_state=args.seed,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": best_params["lr"]},
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    fit_kwargs = {
        "series": train_targets,
        "val_series": val_targets,
        "epochs": args.epochs,
        "dataloader_kwargs": {"batch_size": args.batch_size},
        "verbose": False,
    }
    if args.covariate_mode == "alpha":
        fit_kwargs["future_covariates"] = future_cov_train
        fit_kwargs["val_future_covariates"] = future_cov_val

    final_model.fit(**fit_kwargs)

    # 保存最終模型
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    final_model.save(args.output_model)
    print(f"Saved best LSTM model to {args.output_model}")


if __name__ == "__main__":
    main()
