import argparse
import csv
import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np
from darts import TimeSeries
from darts.models import TSMixerModel

# Ensure custom loss is importable for model deserialization.
from model_train import WeightedLoss  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained TSMixer on the test split.")
    parser.add_argument("--data", required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--output", default=None, help="Optional CSV path for metrics.")
    parser.add_argument(
        "--use_log_target",
        action="store_true",
        help="Set if training used log1p targets (applies expm1 to predictions).",
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
    return np.expm1(values) if use_log_target else values


def main() -> None:
    args = parse_args()
    with open(args.data, "rb") as f:
        dataset = pickle.load(f)

    test_targets: List[Optional[TimeSeries]] = dataset["test"]["target"]
    test_covs: List[Optional[TimeSeries]] = dataset["test"]["cov"]
    train_targets: List[Optional[TimeSeries]] = dataset["train"]["target"]
    val_targets: List[Optional[TimeSeries]] = dataset["val"]["target"]
    train_covs: List[Optional[TimeSeries]] = dataset["train"]["cov"]
    val_covs: List[Optional[TimeSeries]] = dataset["val"]["cov"]
    tickers: List[str] = dataset.get("tickers", [])

    model = TSMixerModel.load(args.model)

    results: List[Tuple[str, float, float, float, float]] = []

    plots_dir = pathlib.Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    rolling_plots_dir = pathlib.Path("outputs/rolling_plots")
    rolling_plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, test_ts in enumerate(test_targets):
        if test_ts is None or len(test_ts) == 0:
            continue
        ticker = tickers[idx] if idx < len(tickers) else f"Series_{idx}"

        history_series = _combine_segments(train_targets, val_targets, test_ts, idx)
        history_covariates = _combine_covariates(train_covs, val_covs, test_covs[idx], idx)

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
        )

        pred_vals = preds.to_dataframe().iloc[:, 0].to_numpy()
        truth_df = test_ts.to_dataframe()
        true_vals = truth_df.iloc[:, 0].to_numpy()
        garch_vals = truth_df.iloc[:, 1].to_numpy()

        pred_vals = _invert_log(pred_vals, args.use_log_target)
        true_vals = _invert_log(true_vals, args.use_log_target)
        garch_vals = _invert_log(garch_vals, args.use_log_target)

        mae_model = float(np.mean(np.abs(pred_vals - true_vals)))
        rmse_model = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
        mae_garch = float(np.mean(np.abs(garch_vals - true_vals)))
        rmse_garch = float(np.sqrt(np.mean((garch_vals - true_vals) ** 2)))

        results.append((ticker, mae_model, rmse_model, mae_garch, rmse_garch))
        print(
            f"{ticker}: MAE_model={mae_model:.6f}, RMSE_model={rmse_model:.6f}, "
            f"MAE_garch={mae_garch:.6f}, RMSE_garch={rmse_garch:.6f}"
        )

        # Plot predictions vs actuals and GARCH baseline
        import matplotlib.pyplot as plt  # local import to keep dependency scope narrow

        dates = test_ts.time_index

        plt.figure(figsize=(10, 4))
        plt.plot(dates, np.clip(true_vals,  1e-8, None), label="True", linewidth=1.5)
        plt.plot(dates, np.clip(pred_vals,  1e-8, None), label="TSMixer", linewidth=1.2)
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

        window = 20
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


    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["code", "MAE_model", "RMSE_model", "MAE_garch", "RMSE_garch"])
            for row in results:
                writer.writerow(row)
        print(f"Evaluation metrics saved to {output_path}")


if __name__ == "__main__":
    main()
