import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries


def _cast_series_list(series_list):
    out = []
    for ts in series_list:
        out.append(ts.astype(np.float32) if ts is not None else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot target series (var_true_90) for each ticker from ts_data.pkl.")
    parser.add_argument("--data", default="Dataset/ts_data.pkl", help="Path to ts_data.pkl")
    parser.add_argument("--output", default="outputs/target_plots", help="Directory to save plots")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with data_path.open("rb") as f:
        dataset = pickle.load(f)

    train_targets = _cast_series_list(dataset["train"]["target"])
    val_targets = _cast_series_list(dataset["val"]["target"])
    test_targets = _cast_series_list(dataset["test"]["target"])
    tickers = dataset.get("tickers", [])
    trading_calendar = dataset.get("trading_calendar", [])

    # iterate each ticker
    for idx, train_ts in enumerate(train_targets):
        ticker = tickers[idx] if idx < len(tickers) else f"Series_{idx}"
        series_parts = []
        for part in (train_targets[idx], val_targets[idx], test_targets[idx]):
            if part is not None:
                series_parts.append(part)
        if not series_parts:
            continue
        full_ts = series_parts[0]
        for extra in series_parts[1:]:
            full_ts = full_ts.append(extra)

        df = full_ts.to_dataframe()
        values = df.iloc[:, 0].to_numpy()
        idx_time = full_ts.time_index

        if trading_calendar:
            cal = np.array(trading_calendar)
            dates = cal[idx_time.astype(int)]
        else:
            dates = idx_time

        plt.figure(figsize=(10, 4))
        plt.plot(dates, values, linewidth=1.2)
        plt.title(f"{ticker} - var_true_90")
        plt.xlabel("Date")
        plt.ylabel("var_true_90")
        plt.tight_layout()
        plt.savefig(out_dir / f"{ticker}.png", dpi=150)
        plt.close()
        print(f"Saved plot for {ticker}")


if __name__ == "__main__":
    main()
