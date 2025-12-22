import os
import sys
from typing import List

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from joblib import Parallel, delayed
from typing import Tuple


# Project root and shared paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from pipelines.run_alpha101_from_ohlc import (  # noqa: E402
    build_alpha_input,
    load_ohlc,
)


def load_prices() -> pd.DataFrame:
    """Load adjusted close prices using the same mapping as Alpha101.

    Returns a long-format DataFrame with columns: date, code, close.
    """
    raw = load_ohlc()
    alpha_input = build_alpha_input(raw)
    prices = alpha_input[["date", "code", "close"]].copy()
    return prices


def add_logreturns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily close-to-close log returns per code.
    Unit: percentage
    """
    df = prices.sort_values(["code", "date"]).copy()
    df["log_return"] = df.groupby("code")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df["log_return"] = df["log_return"] * 100
    print("log-return calculation completed")
    return df


def add_ar_mean(
    df: pd.DataFrame,
    window: int = 90
) -> pd.DataFrame:
    """
    The AR model forecasts the average
    predicted daily log returns ˆμt for the day t on a rolling basis
    using the daily stock log returns rt for the past 90 days
    (rt−90, rt−89, rt−88, ..., rt−1).
    """
    df = df.copy()
    df[f"u_hat_{window}"] = np.nan

    for code, sub in df.groupby("code"):
        r = sub["log_return"].values
        idx = sub.index
        n = len(r)

        for t in range(window, n):
            train_slice = r[t-window:t]
            if not np.isfinite(train_slice).all():
                continue
            
            model = AutoReg(train_slice, lags=1, old_names=False).fit()
            mu_hat = model.predict(start=len(train_slice), end=len(train_slice))[0]
            df.loc[idx[t], f"u_hat_{window}"] = mu_hat
    print("AR calculation completed")
    return df
    

def add_gt_var(
    df: pd.DataFrame,
    window: int = 90
) -> pd.DataFrame:
    """
    The ground truth varuance.
    sigma_t^2 = (r_t - ^u_t)^2
    """
    df = df.copy()
    df[f"var_true_{window}"] = (df["log_return"] - df[f"u_hat_{window}"]) ** 2
    print("ground-truth var calculation completed")
    return df


def _compute_garch_for_one_code(
    code: str,
    sub: pd.DataFrame,
    window: int,
    model_type: str,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Calculate rolling GARCH for signal code.
    return: (code, index_array, preds_array)
    """
    r = sub["log_return"].astype(float)
    n = len(r)
    preds = np.full(n, np.nan)

    if n <= window:
        # 全部 NaN，直接回傳
        return code, sub.index.to_numpy(), preds

    for t in range(window, n):
        train_slice = r.iloc[t - window : t].dropna()
        if len(train_slice) < window * 0.8:
            continue

        # GARCH type
        if(model_type == "garch"):
            am = arch_model(
                train_slice,
                mean="AR",
                lags=1,
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
        elif model_type == "gjrgarch":
            am = arch_model(
                train_slice,
                mean="AR",
                lags=1,
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                dist="normal",
                rescale=False,
            )
        elif model_type == "tgarch":
            am = arch_model(
                train_slice,
                mean="AR",
                lags=1,
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                power=1.0,
                dist="normal",
                rescale=False,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        res = am.fit(disp="off")

        fcast = res.forecast(horizon=1, reindex=False)
        var_next = float(fcast.variance.values[-1, 0])
        preds[t] = var_next

    return code, sub.index.to_numpy(), preds


def add_AR_GARCH(
    df: pd.DataFrame,
    window: int = 90,
    model_type: str = "garch",
    n_jobs: int = -1,   # 預設用全部核心
) -> pd.DataFrame:
    df = df.copy()
    col_name = f"{model_type}_var_{window}"
    df[col_name] = np.nan

    # 先把 groupby 結果 materialize，避免在 worker 裡面再做 groupby
    groups = list(df.groupby("code"))

    # 並行計算每個 code 的 preds
    results = Parallel(n_jobs=os.cpu_count()-1)(
        delayed(_compute_garch_for_one_code)(code, sub, window, model_type)
        for code, sub in groups
    )

    # 把結果貼回 df
    for code, idx_array, preds in results:
        df.loc[idx_array, col_name] = preds
        print(code, "calculation completed")

    print(f"{model_type.upper()} calculation completed")
    return df


def main() -> None:
    window = 90

    prices = load_prices()
    prices = add_logreturns(prices)
    features = add_ar_mean(prices, window)
    features = add_gt_var(features, window)
    features = add_AR_GARCH(features, window, "garch")
    features = add_AR_GARCH(features, window, "gjrgarch")
    features = add_AR_GARCH(features, window, "tgarch")

    out_parquet = os.path.join(DATA_DIR, "volatility_labels_features.parquet")
    out_csv = os.path.join(DATA_DIR, "volatility_labels_features.csv")

    features.to_parquet(out_parquet)
    features.to_csv(out_csv, index=False)

    print(f"Saved volatility labels/features to {out_parquet}")
    print(f"Saved volatility labels/features to {out_csv}")


if __name__ == "__main__":
    main()
