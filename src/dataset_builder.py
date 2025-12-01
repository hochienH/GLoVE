import argparse
import pathlib
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries


SplitDict = Dict[str, Dict[str, List[Optional[TimeSeries]]]]


def _load_dataframe(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, parse_dates=["date"])


def _compute_split_lengths(n_obs: int, val_frac: float, test_frac: float) -> Tuple[int, int]:
    def _to_length(fraction: float) -> int:
        if fraction <= 0:
            return 0
        if 0 < fraction <= 1:
            return int(np.floor(n_obs * fraction))
        return int(fraction)

    val_len = _to_length(val_frac)
    test_len = _to_length(test_frac)
    if val_len == 0 and val_frac > 0:
        val_len = 1
    if test_len == 0 and test_frac > 0:
        test_len = 1
    if val_len + test_len >= n_obs:
        # leave at least one sample for training
        overflow = val_len + test_len - (n_obs - 1)
        val_len = max(0, val_len - overflow)
        if val_len + test_len >= n_obs:
            val_len = 0
            test_len = max(0, n_obs - 1)
    return val_len, test_len


def _build_static_covariates(
    code_to_industry: Dict[str, str],
    static_mode: str,
) -> Dict[str, pd.DataFrame]:
    if static_mode == "none":
        return {}

    unique_industries = sorted({ind for ind in code_to_industry.values()})
    unique_codes = sorted(code_to_industry.keys())

    if static_mode == "industry":
        static_columns = [f"industry_{ind}" for ind in unique_industries]
    elif static_mode == "ticker":
        static_columns = [f"ticker_{code}" for code in unique_codes]
    else:  # industry_ticker
        static_columns = [f"industry_{ind}" for ind in unique_industries] + [
            f"ticker_{code}" for code in unique_codes
        ]

    covariates: Dict[str, pd.DataFrame] = {}
    for code, industry in code_to_industry.items():
        vec = np.zeros((1, len(static_columns)))
        if static_mode in {"industry", "industry_ticker"}:
            vec[0, unique_industries.index(industry)] = 1
        if static_mode in {"ticker", "industry_ticker"}:
            tick_offset = (
                unique_codes.index(code)
                if static_mode == "ticker"
                else len(unique_industries) + unique_codes.index(code)
            )
            vec[0, tick_offset] = 1
        covariates[code] = pd.DataFrame(vec, columns=static_columns)
    return covariates


def build_datasets(
    df: pd.DataFrame,
    target_col: str,
    garch_col: str,
    val_frac: float,
    test_frac: float,
    input_chunk_length: int,
    static_mode: str,
) -> Dict[str, object]:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    if garch_col not in df.columns:
        raise ValueError(f"GARCH column '{garch_col}' is required for training.")

    feature_cols = [
        c
        for c in df.columns
        if c not in {target_col, garch_col, "code", "date", "industry_code"}
    ]

    dataset: Dict[str, object] = {
        "train": {"target": [], "cov": []},
        "val": {"target": [], "cov": []},
        "test": {"target": [], "cov": []},
        "tickers": [],
        "feature_cols": feature_cols,
        "target_col": target_col,
        "garch_col": garch_col,
        "static_mode": static_mode,
        "input_chunk_length": input_chunk_length,
    }

    code_to_industry = {
        str(code): str(group["industry_code"].iloc[0]) if "industry_code" in group else "unknown"
        for code, group in df.groupby("code")
    }
    static_covs = _build_static_covariates(code_to_industry, static_mode=static_mode)

    min_train_len = input_chunk_length + 1
    for code, group in df.groupby("code"):
        group = group.sort_values("date").set_index("date")
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="B")
        group = group.reindex(full_index)
        group = group.ffill()
        group["code"] = str(code)
        if "industry_code" in group.columns:
            first_industry = group["industry_code"].dropna().iloc[0] if group["industry_code"].notna().any() else "unknown"
            group["industry_code"] = group["industry_code"].fillna(first_industry)
        group = group.reset_index().rename(columns={"index": "date"})
        group = group.dropna(subset=[target_col, garch_col])
        n_obs = len(group)
        if n_obs < min_train_len:
            continue

        val_len, test_len = _compute_split_lengths(n_obs, val_frac, test_frac)
        train_end = n_obs - val_len - test_len
        if train_end < min_train_len:
            continue

        train_df = group.iloc[:train_end]
        val_df = group.iloc[train_end : train_end + val_len] if val_len > 0 else None
        test_df = group.iloc[train_end + val_len :] if test_len > 0 else None

        target_value_cols = [target_col, garch_col]
        ts_kwargs = {
            "time_col": "date",
            "value_cols": target_value_cols,
            "fill_missing_dates": True,
            "freq": "B",
        }
        train_target_ts = TimeSeries.from_dataframe(train_df, **ts_kwargs)
        val_target_ts = (
            TimeSeries.from_dataframe(val_df, **ts_kwargs)
            if val_df is not None and len(val_df) > 0
            else None
        )
        test_target_ts = (
            TimeSeries.from_dataframe(test_df, **ts_kwargs)
            if test_df is not None and len(test_df) > 0
            else None
        )

        cov_kwargs = {"time_col": "date", "value_cols": feature_cols, "fill_missing_dates": True, "freq": "B"}
        cov_ts_train = TimeSeries.from_dataframe(train_df, **cov_kwargs) if feature_cols else None
        cov_ts_val = (
            TimeSeries.from_dataframe(val_df, **cov_kwargs)
            if feature_cols and val_df is not None and len(val_df) > 0
            else None
        )
        cov_ts_test = (
            TimeSeries.from_dataframe(test_df, **cov_kwargs)
            if feature_cols and test_df is not None and len(test_df) > 0
            else None
        )

        if static_mode != "none":
            static_df = static_covs[str(code)]
            train_target_ts = train_target_ts.with_static_covariates(static_df)
            if val_target_ts is not None:
                val_target_ts = val_target_ts.with_static_covariates(static_df)
            if test_target_ts is not None:
                test_target_ts = test_target_ts.with_static_covariates(static_df)

        dataset["tickers"].append(str(code))
        dataset["train"]["target"].append(train_target_ts)
        dataset["train"]["cov"].append(cov_ts_train)
        dataset["val"]["target"].append(val_target_ts)
        dataset["val"]["cov"].append(cov_ts_val)
        dataset["test"]["target"].append(test_target_ts)
        dataset["test"]["cov"].append(cov_ts_test)

    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Darts TimeSeries datasets for TSMixer.")
    parser.add_argument("--input", required=True, help="Path to cleaned data (CSV or PKL).")
    parser.add_argument("--output", required=True, help="Path to save the dataset pickle.")
    parser.add_argument("--target_col", default="var_true_90", help="Target column name.")
    parser.add_argument("--garch_col", default="garch_var_90", help="GARCH baseline column name.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction or count.")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction or count.")
    parser.add_argument(
        "--input_chunk_length",
        type=int,
        default=90,
        help="Input chunk length; series shorter than this are dropped.",
    )
    parser.add_argument(
        "--static_mode",
        choices=["industry", "industry_ticker", "ticker", "none"],
        default="industry_ticker",
        help="Static covariates mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    df = _load_dataframe(input_path)
    dataset = build_datasets(
        df=df,
        target_col=args.target_col,
        garch_col=args.garch_col,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        input_chunk_length=args.input_chunk_length,
        static_mode=args.static_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    total_series = len(dataset["tickers"])
    print(f"Dataset built with {total_series} series. Saved to {output_path}")


if __name__ == "__main__":
    main()
