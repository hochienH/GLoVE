import argparse
import csv
import pathlib
import pickle
from typing import List, Optional, Tuple
import torch
import numpy as np

from darts import TimeSeries
from darts.models import TSMixerModel

# Ensure custom loss is importable for model deserialization.
from model_train import WeightedLoss  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained TSMixer on the test split.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--output", default=None, help="Optional folder path for metrics.")
    parser.add_argument(
        "--invert_train_scale",
        choices=["none", "mean", "zscore"],
        default="none",
        help="If dataset was scaled, invert using stored train stats: 'mean' (multiply by mean) or 'zscore' (x*std+mean).",
    )
    parser.add_argument(
        "--use_log_target",
        action="store_true",
        help="Set if training used log1p targets (applies expm1 to predictions).",
    )
    parser.add_argument(
        "--covariate_mode",
        choices=["none", "alpha"],
        default="none",
        help="要不要使用協變數。預設 none（不使用alpha及營收資料）"
    )
    return parser.parse_args()


def _combine_segments(
    train_list: List[Optional[TimeSeries]],
    val_list: List[Optional[TimeSeries]],
    test_item: TimeSeries,
    idx: int,
) -> TimeSeries:
    series = train_list[idx]
    if series is None:
        raise ValueError(f"Training segment missing for series index {idx}.")
    if idx < len(val_list) and val_list[idx] is not None:
        series = series.append(val_list[idx])
    series = series.append(test_item)
    return series


def _combine_covariates(
    train_covs: List[Optional[TimeSeries]],
    val_covs: List[Optional[TimeSeries]],
    test_cov: Optional[TimeSeries],
    idx: int,
) -> Optional[TimeSeries]:
    cov = train_covs[idx] if idx < len(train_covs) else None
    if idx < len(val_covs) and val_covs[idx] is not None:
        cov = cov.append(val_covs[idx]) if cov is not None else val_covs[idx]
    if test_cov is not None:
        cov = cov.append(test_cov) if cov is not None else test_cov
    return cov


def _invert_log(values: np.ndarray, use_log_target: bool) -> np.ndarray:
    if not use_log_target:
        return values
    eps = 1
    return np.exp(values) - eps


def main() -> None:
    args = parse_args()
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

    def _prepare_covariates(covs):
        return covs if covs is not None else []

    # 這裡全部都變成 float32 的 TimeSeries
    train_targets: List[Optional[TimeSeries]] = _cast_series_list(dataset["train"]["target"])
    val_targets: List[Optional[TimeSeries]]   = _cast_series_list(dataset["val"]["target"])
    test_targets: List[Optional[TimeSeries]]  = _cast_series_list(dataset["test"]["target"])
    train_covs: List[Optional[TimeSeries]] = _cast_series_list(_prepare_covariates(dataset["train"]["cov"]))
    val_covs: List[Optional[TimeSeries]]   = _cast_series_list(_prepare_covariates(dataset["val"]["cov"]))
    test_covs: List[Optional[TimeSeries]]  = _cast_series_list(_prepare_covariates(dataset["test"]["cov"]))
    tickers: List[str] = dataset.get("tickers", [])
    target_means = dataset.get("target_means", {}) if isinstance(dataset, dict) else {}
    target_stds = dataset.get("target_stds", {}) if isinstance(dataset, dict) else {}
    trading_calendar = dataset.get("trading_calendar", [])
    date_index = dataset.get("date_index", {})

    # 給 matmul 一點空間，不一定必要，但順手加
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")


    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [0]
        precision = "bf16-mixed"   # 有 GPU 再用混合精度
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        precision = "bf16-mixed"
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32


    model = TSMixerModel.load(
        args.model,
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
        },
    )

    results: List[Tuple[str, float, float, float, float]] = []

    if args.output:
        out_dir = pathlib.Path(args.output)
    else:
        out_dir = pathlib.Path("outputs/Base")
    plots_dir = out_dir / "plots"
    rolling_plots_dir = out_dir / "rolling_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rolling_plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, test_ts in enumerate(test_targets):
        if test_ts is None or len(test_ts) == 0:
            continue
        ticker = tickers[idx] if idx < len(tickers) else f"Series_{idx}"

        history_series = _combine_segments(train_targets, val_targets, test_ts, idx)

        if args.covariate_mode == "alpha":
            history_covariates = _combine_covariates(train_covs, val_covs, test_covs[idx], idx)
        else:
            history_covariates = None

        start_time = test_ts.start_time()
        preds = model.historical_forecasts(
            series=history_series,
            past_covariates=history_covariates,
            start=start_time,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False,
            last_points_only=True,
            # predict_kwargs={"dataloader_kwargs": {"num_workers": 1, "persistent_workers":True}}
        )

        pred_vals = preds.to_dataframe().iloc[:, 0].to_numpy()
        truth_df = test_ts.to_dataframe()
        true_vals = truth_df.iloc[:, 0].to_numpy()
        garch_vals = truth_df.iloc[:, 1].to_numpy()

        # 返還縮放（mean 或 zscore）
        if args.invert_train_scale != "none":
            code_key = ticker
            if code_key not in target_means and str(code_key) in target_means:
                code_key = str(code_key)
            mean_scale = float(target_means.get(code_key, 1.0))
            std_scale = float(target_stds.get(code_key, 1.0))
            if args.invert_train_scale == "mean":
                pred_vals = pred_vals * mean_scale
                true_vals = true_vals * mean_scale
                garch_vals = garch_vals * mean_scale
            elif args.invert_train_scale == "zscore":
                pred_vals = pred_vals * std_scale + mean_scale
                true_vals = true_vals * std_scale + mean_scale
                garch_vals = garch_vals * std_scale + mean_scale

        pred_vals = _invert_log(pred_vals, args.use_log_target)
        true_vals = _invert_log(true_vals, args.use_log_target)
        garch_vals = _invert_log(garch_vals, args.use_log_target)

        mae_model = float(np.mean(np.abs(pred_vals - true_vals)))
        rmse_model = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
        mae_garch = float(np.mean(np.abs(garch_vals - true_vals)))
        rmse_garch = float(np.sqrt(np.mean((garch_vals - true_vals) ** 2)))

        # QLIKE 評估（只在真實值與預測皆為正時計算）
        eps = 1e-12
        mask_model = (true_vals > eps) & (pred_vals > eps)
        mask_garch = (true_vals > eps) & (garch_vals > eps)
        if mask_model.any():
            qlike_model = float(
                np.mean(
                    np.log(pred_vals[mask_model]) + true_vals[mask_model] / pred_vals[mask_model]
                )
            )
        else:
            qlike_model = float("nan")
        if mask_garch.any():
            qlike_garch = float(
                np.mean(
                    np.log(garch_vals[mask_garch]) + true_vals[mask_garch] / garch_vals[mask_garch]
                )
            )
        else:
            qlike_garch = float("nan")


        results.append((ticker, mae_model, rmse_model, mae_garch, rmse_garch, qlike_model, qlike_garch))

        print(
            f"{ticker}: "
            f"MAE_model={mae_model:.6f}, RMSE_model={rmse_model:.6f}, "
            f"MAE_garch={mae_garch:.6f}, RMSE_garch={rmse_garch:.6f}, "
            f"QLIKE_model={qlike_model:.6f}, QLIKE_garch={qlike_garch:.6f}"
        )


        # Plot predictions vs actuals and GARCH baseline
        import matplotlib.pyplot as plt  # local import to keep dependency scope narrow
        
        # 將 t_idx 轉回實際日期以便繪圖
        if trading_calendar:
            cal = np.array(trading_calendar)
            dates = cal[test_ts.time_index.astype(int)]
            dates_pred = cal[preds.time_index.astype(int)]
        else:
            dates = test_ts.time_index
            dates_pred = preds.time_index

        plt.figure(figsize=(10, 4))
        plt.plot(dates, np.clip(true_vals,  1e-8, None), label="True", linewidth=1.5)
        plt.plot(dates_pred, np.clip(pred_vals,  1e-8, None), label="TSMixer", linewidth=1.2)
        plt.plot(dates, np.clip(garch_vals, 1e-8, None), label="GARCH", linewidth=1.0, linestyle="--")
        plt.title(f"{ticker} - Test Forecasts")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.yscale('log', base=10)
        plt.ylim(1e-2, 1e2)
        plt.yticks([1e-2, 1e-1, 1, 1e1, 1e2], [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{ticker}.png", dpi=150)
        plt.close()

        window = 5
        kernel = np.ones(window) / window
        true_roll  = np.convolve(true_vals,  kernel, mode='valid')
        pred_roll  = np.convolve(pred_vals,  kernel, mode='valid')
        garch_roll = np.convolve(garch_vals, kernel, mode='valid')
        dates_roll = dates[window-1:]

        plt.figure(figsize=(10, 4))
        plt.plot(dates_roll, np.clip(true_roll,  1e-8, None), label="True", linewidth=1.5)
        plt.plot(dates_roll, np.clip(pred_roll,  1e-8, None), label="TSMixer", linewidth=1.2)
        plt.plot(dates_roll, np.clip(garch_roll, 1e-8, None), label="GARCH", linewidth=1.0, linestyle="--")
        plt.title(f"{ticker} - Test Forecasts")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.yscale('log', base=10)
        plt.ylim(1e-2, 1e2)
        plt.yticks([1e-2, 1e-1, 1, 1e1, 1e2], [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(rolling_plots_dir /f"{ticker}.png", dpi=150)
        plt.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["code", "MAE_model", "RMSE_model", "MAE_garch", "RMSE_garch", "QLIKE_model", "QLIKE_garch"])
        for row in results:
            writer.writerow(row)
    print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == "__main__":
    import time
    then = time.time()
    main()
    print(round(time.time() - then, 2))