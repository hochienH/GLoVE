import argparse
import csv
import pathlib
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from darts import TimeSeries
from darts.models import RNNModel

# Ensure WeightedLoss is registered for model deserialization.
from model_train_lstm import WeightedLoss  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM (RNNModel) on validation/test splits.")
    parser.add_argument("--data", required=True, help="dataset_builder.py 輸出的資料集 pickle 路徑。")
    parser.add_argument("--model", required=True, help="LSTM 訓練後儲存的 .pth 模型。")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="要評估的資料切分；預設使用 test。",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=1,
        help="historical_forecasts 的 horizon（預設 1 代表逐日滾動預測）。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="historical_forecasts 的 stride。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="輸出csv跟圖片的資料夾",
    )
    parser.add_argument(
        "--use_log_target",
        action="store_true",
        help="若訓練時 target 做過 log1p，推論後會自動 expm1 還原。",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="啟用後輸出預測 vs. 實際/GARCH 的折線圖與 rolling mean 圖。",
    )
    parser.add_argument(
        "--covariate_mode",
        choices=["none", "alpha"],
        default="none",
        help="協變數使用方式：none 代表完全不用 cov；alpha 則會把 cov 往前 lag 天再傳給模型，避免使用未來資訊。",
    )
    parser.add_argument(
        "--covariate_lag",
        type=int,
        default=1,
        help="alpha 模式下要往前取幾天的 cov（預設 1）。",
    )
    parser.add_argument(
        "--invert_train_scale",
        choices=["none", "mean", "zscore"],
        default="none",
        help="If dataset was scaled, invert using stored train stats: 'mean' (multiply by mean) or 'zscore' (x*std+mean).",
    )
    return parser.parse_args()


def _cast_series_list(series_list: Sequence[Optional[TimeSeries]]) -> List[Optional[TimeSeries]]:
    casted: List[Optional[TimeSeries]] = []
    for ts in series_list:
        if ts is None:
            casted.append(None)
        else:
            casted.append(ts.astype(np.float32))
    return casted


def _concat_series(parts: Sequence[Optional[TimeSeries]]) -> Optional[TimeSeries]:
    combined: Optional[TimeSeries] = None
    for ts in parts:
        if ts is None:
            continue
        combined = ts if combined is None else combined.append(ts)
    return combined


def _build_lagged_covariates(
    series_list: Sequence[Optional[TimeSeries]], lag: int
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


def _invert_log(values: np.ndarray, use_log_target: bool) -> np.ndarray:
    return np.expm1(values) if use_log_target else values


def _detect_accelerator() -> Tuple[str, object]:
    if torch.cuda.is_available():
        return "gpu", "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "auto"
    return "cpu", 1


def main() -> None:
    args = parse_args()
    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets = _cast_series_list(dataset["val"]["target"])
    test_targets = _cast_series_list(dataset["test"]["target"])
    train_covs = _cast_series_list(dataset["train"]["cov"])
    val_covs = _cast_series_list(dataset["val"]["cov"])
    test_covs = _cast_series_list(dataset["test"]["cov"])
    tickers: List[str] = dataset.get("tickers", [])
    target_means = dataset.get("target_means", {}) if isinstance(dataset, dict) else {}
    target_stds = dataset.get("target_stds", {}) if isinstance(dataset, dict) else {}
    trading_calendar = dataset.get("trading_calendar", [])
    date_index = dataset.get("date_index", {})

    accelerator, devices = _detect_accelerator()

    load_kwargs = {
        "pl_trainer_kwargs": {
            "accelerator": accelerator,
            "devices": devices,
            "enable_progress_bar": False,
        }
    }
    model = RNNModel.load(args.model, **load_kwargs)

    if args.save_plots:
        dir_path = pathlib.Path(args.output)
        dir_path.mkdir(parents=True, exist_ok=True)
        plots_dir = pathlib.Path(args.output) / "plots"
        rolling_dir = pathlib.Path(args.output) / "rolling_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        rolling_dir.mkdir(parents=True, exist_ok=True)
    else:
        plots_dir = rolling_dir = None

    if args.covariate_mode == "alpha":
        train_covs_processed = _build_lagged_covariates(train_covs, args.covariate_lag)
        val_covs_processed = _build_lagged_covariates(val_covs, args.covariate_lag)
        test_covs_processed = _build_lagged_covariates(test_covs, args.covariate_lag)
    else:
        train_covs_processed = [None] * len(train_covs)
        val_covs_processed = [None] * len(val_covs)
        test_covs_processed = [None] * len(test_covs)

    if args.split == "val":
        eval_targets = val_targets
        eval_covs = val_covs_processed
        history_builder = lambda idx: _concat_series([train_targets[idx], val_targets[idx]])
        cov_builder = lambda idx: _concat_series([train_covs_processed[idx], val_covs_processed[idx]])
    else:
        eval_targets = test_targets
        eval_covs = test_covs_processed
        history_builder = lambda idx: _concat_series([train_targets[idx], val_targets[idx], test_targets[idx]])
        cov_builder = lambda idx: _concat_series([train_covs_processed[idx], val_covs_processed[idx], test_covs_processed[idx]])

    metrics: List[Tuple[str, float, float, float, float]] = []

    for idx, eval_ts in enumerate(eval_targets):
        if eval_ts is None or len(eval_ts) == 0:
            continue
        ticker = tickers[idx] if idx < len(tickers) else f"Series_{idx}"

        history_series = history_builder(idx)
        history_covariates = cov_builder(idx) if args.covariate_mode == "alpha" else None
        if history_series is None:
            print(f"[警告] {ticker} 缺少歷史資料，跳過。")
            continue

        start_time = eval_ts.start_time()
        preds = model.historical_forecasts(
            series=history_series,
            future_covariates=history_covariates,
            start=start_time,
            forecast_horizon=args.forecast_horizon,
            stride=args.stride,
            retrain=False,
            verbose=False,
            last_points_only=True,
        )

        pred_vals = preds.to_dataframe().iloc[:, 0].to_numpy()
        truth_df = eval_ts.to_dataframe()
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

        metrics.append((ticker, mae_model, rmse_model, mae_garch, rmse_garch, qlike_model, qlike_garch))
        
        print(
            f"{ticker}: "
            f"MAE_model={mae_model:.6f}, RMSE_model={rmse_model:.6f}, "
            f"MAE_garch={mae_garch:.6f}, RMSE_garch={rmse_garch:.6f}, "
            f"QLIKE_model={qlike_model:.6f}, QLIKE_garch={qlike_garch:.6f}"
        )

        if args.save_plots:
            import matplotlib.pyplot as plt  # noqa: PLC0415

            # 將 t_idx 轉回實際日期以便繪圖
            if trading_calendar:
                cal = np.array(trading_calendar)
                dates = cal[eval_ts.time_index.astype(int)]
                dates_pred = cal[preds.time_index.astype(int)]
            else:
                dates = eval_ts.time_index
                dates_pred = preds.time_index

            plt.figure(figsize=(10, 4))
            plt.plot(dates, np.clip(true_vals, 1e-8, None), label="True", linewidth=1.5)
            plt.plot(dates_pred, np.clip(pred_vals, 1e-8, None), label="LSTM", linewidth=1.2)
            plt.plot(dates, np.clip(garch_vals, 1e-8, None), label="GARCH", linewidth=1.0, linestyle="--")
            plt.title(f"{ticker} - {args.split.upper()} Forecasts")
            plt.xlabel("Date")
            plt.ylabel("Volatility")
            plt.yscale("log", base=10)
            plt.ylim(1e-2, 1e2)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"{ticker}.png", dpi=150)
            plt.close()

            window = 5
            kernel = np.ones(window) / window
            true_roll = np.convolve(true_vals, kernel, mode="valid")
            pred_roll = np.convolve(pred_vals, kernel, mode="valid")
            garch_roll = np.convolve(garch_vals, kernel, mode="valid")
            dates_roll = dates[window - 1 :]

            plt.figure(figsize=(10, 4))
            plt.plot(dates_roll, np.clip(true_roll, 1e-8, None), label="True", linewidth=1.5)
            plt.plot(dates_roll, np.clip(pred_roll, 1e-8, None), label="LSTM", linewidth=1.2)
            plt.plot(dates_roll, np.clip(garch_roll, 1e-8, None), label="GARCH", linewidth=1.0, linestyle="--")
            plt.title(f"{ticker} - Rolling Mean ({args.split.upper()})")
            plt.xlabel("Date")
            plt.ylabel("Volatility")
            plt.yscale("log", base=10)
            plt.ylim(1e-2, 1e2)
            plt.legend()
            plt.tight_layout()
            plt.savefig(rolling_dir / f"{ticker}.png", dpi=150)
            plt.close()

    if args.output:
        output_path = pathlib.Path(args.output) / "metrics.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["code", "MAE_model", "RMSE_model", "MAE_garch", "RMSE_garch", "QLIKE_model", "QLIKE_garch"])
            writer.writerows(metrics)
        print(f"指標已輸出至 {output_path}")


if __name__ == "__main__":
    main()

