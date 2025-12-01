import argparse
import pathlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _load_dataframe(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, parse_dates=["date"])


def _save_dataframe(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)


def _first_valid_position(series: pd.Series) -> Optional[int]:
    valid_positions = np.flatnonzero(series.notna().to_numpy())
    if len(valid_positions) == 0:
        return None
    return int(valid_positions[0])


def _rank_to_unit_interval(series: pd.Series) -> pd.Series:
    if series.shape[0] <= 1:
        return pd.Series(0.0, index=series.index)
    ranks = series.rank(method="average")
    return 2 * (ranks - 1) / (len(series) - 1) - 1


def _fill_and_scale_alphas(df: pd.DataFrame, alpha_cols: Sequence[str]) -> pd.DataFrame:
    if not alpha_cols:
        return df
    filled = df.groupby("date")[list(alpha_cols)].transform(lambda col: col.fillna(col.median()).fillna(0.0))
    df[list(alpha_cols)] = filled
    df[list(alpha_cols)] = df.groupby("date")[list(alpha_cols)].transform(_rank_to_unit_interval).fillna(0.0)
    return df


def _apply_burn_in(
    df: pd.DataFrame,
    target_col: str,
    garch_col: Optional[str],
    alpha_cols: Sequence[str],
    burn_in_target: int,
    burn_in_garch: int,
    burn_in_alpha: int,
) -> Tuple[pd.DataFrame, List[str]]:
    kept_frames: List[pd.DataFrame] = []
    dropped_codes: List[str] = []
    for code, group in df.groupby("code", sort=False):
        burn_candidates = [burn_in_target]
        if garch_col and garch_col in group.columns:
            burn_candidates.append(burn_in_garch)
        if alpha_cols:
            burn_candidates.append(burn_in_alpha)

        target_start = _first_valid_position(group[target_col])
        if target_start is not None:
            burn_candidates.append(target_start)
        if garch_col and garch_col in group.columns:
            garch_start = _first_valid_position(group[garch_col])
            if garch_start is not None:
                burn_candidates.append(garch_start)

        # if alpha_cols:
        #     alpha_ready = group[list(alpha_cols)].notna().all(axis=1)
        #     alpha_start = _first_valid_position(alpha_ready)
        #     if alpha_start is not None:
        #         burn_candidates.append(alpha_start)

        if alpha_cols:
            # 因為 .notna().all(axis=1) 會把所有值都轉換成True, False
            # _first_valid_position 中的 series.notna().to_numpy() 會全部都是 True, 回傳永遠都是 
            # 因此 需要重寫判斷式
            alpha_ready = group[list(alpha_cols)].notna().all(axis=1)
            true_positions = np.flatnonzero(alpha_ready.to_numpy())
            if len(true_positions) > 0:
                alpha_start = int(true_positions[0])
                burn_candidates.append(alpha_start)


        burn_in = int(max(burn_candidates)) if burn_candidates else 0
        trimmed = group.iloc[burn_in:].copy()
        trimmed = trimmed.dropna(subset=[target_col] + ([garch_col] if garch_col else []))
        if trimmed.empty:
            dropped_codes.append(str(code))
            continue
        kept_frames.append(trimmed)
    if not kept_frames:
        return pd.DataFrame(columns=df.columns), dropped_codes
    return pd.concat(kept_frames).sort_values(["code", "date"]), dropped_codes


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    garch_col: str,
    disabled_features: Sequence[str],
    use_log_target: bool,
    burn_in_target: int,
    burn_in_garch: int,
    burn_in_alpha: int,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    if disabled_features:
        drop_cols = [col for col in disabled_features if col in df.columns]
        df = df.drop(columns=drop_cols)

    alpha_cols = [c for c in df.columns if c.lower().startswith("alpha")]
    df = _fill_and_scale_alphas(df, alpha_cols)

    for col in ["revenue_report", "fin_report"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    if "industry_code" in df.columns:
        df["industry_code"] = df["industry_code"].fillna("unknown").astype(str)

    if use_log_target:
        df[target_col] = np.log1p(np.clip(df[target_col], a_min=0, a_max=None))
        if garch_col in df.columns:
            df[garch_col] = np.log1p(np.clip(df[garch_col], a_min=0, a_max=None))

    cleaned, dropped = _apply_burn_in(
        df=df,
        target_col=target_col,
        garch_col=garch_col if garch_col in df.columns else None,
        alpha_cols=alpha_cols,
        burn_in_target=burn_in_target,
        burn_in_garch=burn_in_garch,
        burn_in_alpha=burn_in_alpha,
    )

    protected_cols = {target_col, "code", "date", "industry_code", garch_col}
    all_nan_cols = [col for col in cleaned.columns if col not in protected_cols and cleaned[col].isna().all()]

    if all_nan_cols:
        cleaned = cleaned.drop(columns=all_nan_cols)

    # if target_col:
    #     cleaned[target_col] = cleaned[target_col] * 10000
    # if garch_col:
    #     cleaned[garch_col] = cleaned[garch_col] * 10000

    cleaned = cleaned.sort_values(["code", "date"]).reset_index(drop=True)
    return cleaned, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw volatility dataset.")
    parser.add_argument("--input", required=True, help="Path to raw CSV/PKL data.")
    parser.add_argument("--output", required=True, help="Path to save cleaned data (CSV or PKL).")
    parser.add_argument(
        "--disabled_features",
        nargs="+",
        default=[],
        help="Feature columns to remove before modeling.",
    )
    parser.add_argument(
        "--target_col",
        default="var_true_90",
        help="Name of the realized volatility target column.",
    )
    parser.add_argument(
        "--garch_col",
        default="garch_var_90",
        help="Column name for GARCH predictions.",
    )
    parser.add_argument(
        "--use_log_target",
        action="store_true",
        help="Apply log1p transform to target and garch features.",
    )
    parser.add_argument(
        "--burn_in_target",
        type=int,
        default=90,
        help="Burn-in days to drop per series for the target column.",
    )
    parser.add_argument(
        "--burn_in_garch",
        type=int,
        default=90,
        help="Burn-in days to drop per series for the garch column.",
    )
    parser.add_argument(
        "--burn_in_alpha",
        type=int,
        default=250,
        help="Burn-in days to drop per series for alpha features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    df_raw = _load_dataframe(input_path)
    df_raw = df_raw.sort_values(["code", "date"])

    cleaned, dropped_codes = preprocess_data(
        df=df_raw,
        target_col=args.target_col,
        garch_col=args.garch_col,
        disabled_features=args.disabled_features,
        use_log_target=args.use_log_target,
        burn_in_target=args.burn_in_target,
        burn_in_garch=args.burn_in_garch,
        burn_in_alpha=args.burn_in_alpha,
    )

    _save_dataframe(cleaned, output_path)

    print(f"Preprocessing complete. Saved to {output_path}")
    if dropped_codes:
        print(f"Skipped {len(dropped_codes)} series with insufficient data after burn-in: {dropped_codes}")
    nan_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
    if nan_cols:
        print(f"Dropped all-NaN columns: {nan_cols}")


if __name__ == "__main__":
    main()
