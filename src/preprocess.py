import pandas as pd
import numpy as np
import argparse

def preprocess_data(df: pd.DataFrame, target_col: str, disabled_features: list, 
                    garch_scaling: str, use_log_target: bool) -> pd.DataFrame:
    # 1. 移除停用的特徵欄位
    if disabled_features:
        for feat in disabled_features:
            if feat in df.columns:
                df.drop(columns=feat, inplace=True)
    # 2. Alpha 補值與排名 [-1,1] 標準化
    if 'alpha' in df.columns:
        # 以當日中位數填補缺失值
        df['alpha'] = df.groupby('date')['alpha'].transform(
            lambda x: x.fillna(x.median())  # 當日所有股票alpha的中位數
        )
        # 對每個日期計算排名並縮放到 [-1,1]
        def rank_to_scale(x: pd.Series) -> pd.Series:
            if len(x) == 1:
                # 單個值無法排名，直接設為0
                return pd.Series(0.0, index=x.index)
            ranks = x.rank(method='average')  # 平均名次（如有並列）
            scaled = (ranks - 1) / (len(x) - 1) * 2 - 1
            return scaled
        df['alpha'] = df.groupby('date')['alpha'].transform(rank_to_scale)
        # 填補可能出現的極端情況NaN（若某日全NaN），用0替代
        df['alpha'] = df['alpha'].fillna(0.0)
    # 3. GARCH 特徵縮放
    if 'garch_pred' in df.columns:
        if garch_scaling == 'global':
            mu = df['garch_pred'].mean()
            sigma = df['garch_pred'].std()
            if pd.isna(sigma) or sigma == 0:
                # 若sigma為0（理論上不太會發生除非數據恆定），避免除零
                sigma = 1e-8
            df['garch_pred'] = (df['garch_pred'] - mu) / sigma
        elif garch_scaling == 'per_series':
            def scale_group(x: pd.Series) -> pd.Series:
                mu = x.mean()
                sigma = x.std()
                if pd.isna(sigma) or sigma == 0:
                    sigma = 1e-8
                return (x - mu) / sigma
            df['garch_pred'] = df.groupby('code')['garch_pred'].transform(scale_group)
        # 填補前面幾天可能存在的NaN（例如某些模型預熱期），用鄰近值或0
        df['garch_pred'] = df.groupby('code')['garch_pred'].apply(
            lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(0)
        )
    # 4. 目標與基準對數轉換
    if use_log_target:
        # 確保目標值非負再取對數，如有負值（不太可能）須先處理
        df[target_col] = np.log1p(np.clip(df[target_col], a_min=0, a_max=None))
        if 'garch_pred' in df.columns:
            # GARCH預測可能也全為非負，本例中視作波動率
            df['garch_pred'] = np.log1p(np.clip(df['garch_pred'], a_min=0, a_max=None))

    # 5️. 營收 / 財報公布日：缺失補 0，強制轉 int
    for col in ["revenue_report", "fin_report"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # 6️6. 產業變數：轉成字串，確保後面 one-hot 時不出問題
    if "industry_code" in df.columns:
        df["industry_code"] = df["industry_code"].astype(str)

    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data for TSMixerModel.")
    parser.add_argument('--input', required=True, help="Path to raw input CSV data")
    parser.add_argument('--output', required=True, help="Path to save cleaned output (csv or pkl)")
    parser.add_argument('--disabled_features', nargs='+', help="Feature columns to remove", default=[])
    parser.add_argument('--garch_scaling', choices=['none', 'global', 'per_series'], default='none',
                        help="Scaling mode for garch_pred feature")
    parser.add_argument('--use_log_target', action='store_true', help="Apply log1p transform to target and garch_pred")
    parser.add_argument('--target_col', default='target', help="Name of the target column in data")
    args = parser.parse_args()
    # 讀取資料
    if args.input.endswith('.pkl') or args.input.endswith('.pickle'):
        df = pd.read_pickle(args.input)
    else:
        df = pd.read_csv(args.input, parse_dates=['date'])
    # 資料排序（按日期）
    df.sort_values(['code', 'date'], inplace=True)
    # 資料預處理
    df_clean = preprocess_data(df, target_col=args.target_col,
                               disabled_features=args.disabled_features,
                               garch_scaling=args.garch_scaling,
                               use_log_target=args.use_log_target)
    # 輸出結果
    if args.output.endswith('.pkl') or args.output.endswith('.pickle'):
        df_clean.to_pickle(args.output)
    else:
        df_clean.to_csv(args.output, index=False)
    print(f"Preprocessing done. Output saved to {args.output}")

if __name__ == "__main__":
    main()
