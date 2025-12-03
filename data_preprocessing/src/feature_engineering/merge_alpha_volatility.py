import os
import sys

import pandas as pd


# Project root and shared paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


def load_alpha_signals(path: str | None = None) -> pd.DataFrame:
    """Load long-format Alpha101 signals (date, code, alpha001..alpha101)."""
    if path is None:
        path = os.path.join(DATA_DIR, "Alpha101_signals.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    return df


def load_volatility_features(path: str | None = None) -> pd.DataFrame:
    """Load volatility / momentum / GARCH features DataFrame."""
    if path is None:
        path = os.path.join(DATA_DIR, "volatility_labels_features.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    return df


def load_ohlc_metadata(path: str | None = None) -> pd.DataFrame:
    """Load OHLC data and keep only metadata columns needed for ML dataset."""
    if path is None:
        path = os.path.join(DATA_DIR, "OHLC.parquet")

    rename_map = {
        "股票代碼": "code",
        "日期": "date",
        "營收發布日_flag": "revenue_report",
        "財報發布日_flag": "fin_report",
        "主產業別_中文": "industry_code",
    }
    df = pd.read_parquet(path, columns=list(rename_map.keys()))
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str)
    return df.drop_duplicates(subset=["date", "code"])


def main() -> None:
    vol_df = load_volatility_features()
    alpha_df = load_alpha_signals()
    ohlc_meta = load_ohlc_metadata()

    merged = ohlc_meta.merge(vol_df, on=["date", "code"], how="left")
    merged = merged.merge(alpha_df, on=["date", "code"], how="left")

    out_parquet = os.path.join(DATA_DIR, "ml_dataset_alpha101_volatility.parquet")
    out_csv = os.path.join(DATA_DIR, "ml_dataset_alpha101_volatility.csv")

    merged.to_parquet(out_parquet)
    merged.to_csv(out_csv, index=False)

    print(f"Saved merged ML dataset to {out_parquet}")
    print(f"Saved merged ML dataset to {out_csv}")


if __name__ == "__main__":
    main()
