import os
import sys
from typing import List

import pandas as pd

# Project root: .../alpha_101
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_DIR, "data", "OHLC.parquet")

# Ensure we import the local alpha101 package from src/alpha101
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from alpha101 import Alpha101  # noqa: E402


def load_ohlc(path: str = DATA_PATH) -> pd.DataFrame:
    """Load OHLC parquet file produced by the TEJ pipeline."""
    return pd.read_parquet(path)


def build_alpha_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform raw OHLC data into the long-format schema expected by Alpha101.

    Expected logical columns (by position) in raw:

        0: code
        1: date
        2: industry
        3: open
        4: high
        5: low
        6: close
        7: adjustment factor
        8: volume
        9: traded amount (often in thousands)
        10: market cap (often in thousands)

    Adjusted prices with suffix ``_調整後`` will be used if present; otherwise
    prices are multiplied by the adjustment factor when available.
    """
    cols: List[str] = list(raw.columns)
    if len(cols) < 11:
        raise ValueError("OHLC.parquet 應至少包含 11 個欄位（代碼、日期、產業、開高低收、調整係數、成交量、成交金額、市值）。")

    (
        code_col,
        date_col,
        industry_col,
        open_col,
        high_col,
        low_col,
        close_col,
        adj_col,
        volume_col,
        amount_col,
        cap_col,
    ) = cols[:11]

    open_adj_name = f"{open_col}_調整後"
    high_adj_name = f"{high_col}_調整後"
    low_adj_name = f"{low_col}_調整後"
    close_adj_name = f"{close_col}_調整後"

    def to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def get_price(series_name: str, adjusted_name: str) -> pd.Series:
        if adjusted_name in raw.columns:
            return to_numeric(raw[adjusted_name])
        if adj_col in raw.columns:
            return to_numeric(raw[series_name]) * to_numeric(raw[adj_col])
        return to_numeric(raw[series_name])

    open_series = get_price(open_col, open_adj_name)
    high_series = get_price(high_col, high_adj_name)
    low_series = get_price(low_col, low_adj_name)
    close_series = get_price(close_col, close_adj_name)

    def maybe_scale_thousands(series: pd.Series, name: str) -> pd.Series:
        # If the column name suggests units in thousands, rescale to units.
        if "千" in str(name):
            return to_numeric(series) * 1000.0
        return to_numeric(series)

    volume_amount = maybe_scale_thousands(raw[amount_col], amount_col)
    cap = maybe_scale_thousands(raw[cap_col], cap_col)

    # If an adjustment factor is available, apply it to amount and cap so
    # that VWAP and capitalization are on the same scale as adjusted prices.
    if adj_col in raw.columns:
        adj = to_numeric(raw[adj_col])
        volume_amount = volume_amount * adj
        cap = cap * adj

    alpha_input = pd.DataFrame(
        {
            "code": raw[code_col].astype(str),
            "date": pd.to_datetime(raw[date_col]),
            "open": open_series.astype(float),
            "high": high_series.astype(float),
            "low": low_series.astype(float),
            "close": close_series.astype(float),
            "volume": to_numeric(raw[volume_col]),
            "volume_amount": volume_amount,
            "cap": cap,
            "industry_code": raw[industry_col].astype(str),
        }
    )
    # Sort by date/code for consistency (Alpha101 also sorts internally).
    return alpha_input.sort_values(["date", "code"]).reset_index(drop=True)


def compute_all_alphas(alpha_engine: Alpha101) -> pd.DataFrame:
    """Compute all 101 alphas and return a long-format DataFrame."""
    alpha_series_list = []

    # Tier 1（最高優先）
    # 👉 從 **alpha102 到 alpha111
    # Tier 2（次優先）
    # 👉 從 **alpha112 到 alpha121
    # Tier 3（補充 / 其餘 Volume 因子）
    # 👉 從 **alpha122 一直到 alpha171
    for alpha_index in range(1, 172):
        method_name = f"alpha{alpha_index:03d}"
        try:
            method = getattr(alpha_engine, method_name)
        except AttributeError:
            continue
        values = method()
        stacked = values.stack().rename(method_name)
        alpha_series_list.append(stacked)

    all_long = pd.concat(alpha_series_list, axis=1).reset_index()
    all_long = all_long.rename(columns={"level_0": "date", "level_1": "code"})
    return all_long


def main() -> None:
    raw = load_ohlc()
    alpha_input = build_alpha_input(raw)

    engine = Alpha101(alpha_input)
    alphas_long = compute_all_alphas(engine)
    alphas_long = alphas_long.sort_values(by=["date", "code"]).reset_index(drop=True)

    output_parquet = os.path.join(os.path.dirname(DATA_PATH), "Alpha101_signals.parquet")
    output_csv = os.path.join(os.path.dirname(DATA_PATH), "Alpha101_signals.csv")
    alphas_long.to_parquet(output_parquet)
    alphas_long.to_csv(output_csv, index=False)
    print(f"Saved Alpha101 signals to {output_parquet}")


if __name__ == "__main__":
    main()
