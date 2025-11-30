import pandas as pd
import numpy as np
import pickle
import argparse
from darts import TimeSeries

def build_datasets(df: pd.DataFrame, target_col: str, val_frac: float, test_frac: float,
                   input_chunk_length: int, static_mode: str) -> dict:
    # 按ticker分組並切分序列
    unique_tickers = sorted(df['ticker'].unique())
    data = {'train': {'target': [], 'cov': []},
            'val': {'target': [], 'cov': []},
            'test': {'target': [], 'cov': []},
            'tickers': [], 'feature_cols': []}
    # 確認特徵欄位列表（除去ticker,date,target）
    feature_cols = [c for c in df.columns if c not in [target_col, 'code', 'date', 'industry_code']]

    data['feature_cols'] = feature_cols  # 保存特徵名，用於後續識別基準欄位等
    
    # 靜態covariates設置
    unique_codes = sorted(df['code'].astype(str).unique())
    unique_industries = sorted(df['industry_code'].astype(str).unique())

    if static_mode in ['industry', 'industry_ticker']:
        static_columns = []
        # industry one-hot 部分
        industry_cols = [f"industry_{ind}" for ind in unique_industries]
        static_columns.extend(industry_cols)

        # ticker one-hot 部分（只有在 industry_ticker 模式下才用）
        if static_mode == 'industry_ticker':
            ticker_cols = [f"ticker_{code}" for code in unique_codes]
            static_columns.extend(ticker_cols)
        else:
            ticker_cols = []

        # 建立每個 code 對應的 static cov 向量
        code_to_static = {}
        for code in unique_codes:
            ind_code = df.loc[df['code'] == code, 'industry_code'].iloc[0]
            vec = np.zeros((1, len(static_columns)))
            # industry one-hot
            ind_idx = unique_industries.index(ind_code)
            vec[0, ind_idx] = 1
            # ticker one-hot (若啟用)
            if static_mode == 'industry_ticker':
                tick_idx = len(unique_industries) + unique_codes.index(code)
                vec[0, tick_idx] = 1
            static_df = pd.DataFrame(vec, columns=static_columns)
            code_to_static[code] = static_df
    else:
        code_to_static = None
        static_columns = []

    # 逐個股票處理
    for tic in unique_tickers:
        sub_df = df[df['code'] == tic].sort_values('date')
        n = len(sub_df)
        # 如果序列長度不足一個input_chunk，則跳過該股票
        if n < input_chunk_length:
            continue
        # 計算測試和驗證長度
        test_len = 0
        val_len = 0
        if test_frac is not None:
            if 0 < test_frac <= 1:
                test_len = int(n * test_frac)
            elif test_frac > 1:
                test_len = int(test_frac)
        if val_frac is not None:
            if 0 < val_frac <= 1:
                val_len = int(n * val_frac)
            elif val_frac > 1:
                val_len = int(val_frac)
        # 至少保留1點作為test/val（若frac很小）
        if test_len == 0 and test_frac and test_frac > 0:
            test_len = 1
        if val_len == 0 and val_frac and val_frac > 0:
            val_len = 1
        # 若資料不足以分割，調整val_len以確保train至少有1條
        if test_len + val_len >= n:
            if test_len < n:
                # 讓訓練至少保留1個，壓縮驗證長度
                val_len = n - test_len - 1
                if val_len < 0:
                    val_len = 0
            else:
                # test已經占滿或超出，放棄該序列
                continue
        train_end = n - val_len - test_len
        train_df = sub_df.iloc[:train_end]
        val_df = sub_df.iloc[train_end:train_end+val_len] if val_len > 0 else None
        test_df = sub_df.iloc[train_end+val_len:] if test_len > 0 else None
        # 生成 TimeSeries
        # 目標序列
        train_target_ts = TimeSeries.from_dataframe(train_df, time_col='date', value_cols=[target_col])
        val_target_ts = TimeSeries.from_dataframe(val_df, time_col='date', value_cols=[target_col]) if val_df is not None else None
        test_target_ts = TimeSeries.from_dataframe(test_df, time_col='date', value_cols=[target_col]) if test_df is not None else None
        # 協變數序列（如果有特徵）
        if feature_cols:
            train_cov_ts = TimeSeries.from_dataframe(train_df, time_col='date', value_cols=feature_cols)
            val_cov_ts = TimeSeries.from_dataframe(val_df, time_col='date', value_cols=feature_cols) if val_df is not None else None
            test_cov_ts = TimeSeries.from_dataframe(test_df, time_col='date', value_cols=feature_cols) if test_df is not None else None
        else:
            train_cov_ts = val_cov_ts = test_cov_ts = None
        # 附加靜態covariates
        if static_mode in ['industry', 'industry_ticker']:
            code_str = str(sub_df['code'].iloc[0])
            static_df = code_to_static[code_str]
            train_target_ts = train_target_ts.with_static_covariates(static_df)
            if val_target_ts is not None:
                val_target_ts = val_target_ts.with_static_covariates(static_df)
            if test_target_ts is not None:
                test_target_ts = test_target_ts.with_static_covariates(static_df)
        # 添加到集合
        data['tickers'].append(tic)
        data['train']['target'].append(train_target_ts)
        data['train']['cov'].append(train_cov_ts)
        if val_target_ts is not None:
            data['val']['target'].append(val_target_ts)
            data['val']['cov'].append(val_cov_ts)
        if test_target_ts is not None:
            data['test']['target'].append(test_target_ts)
            data['test']['cov'].append(test_cov_ts)
    return data

def main():
    parser = argparse.ArgumentParser(description="Build train/val/test TimeSeries datasets.")
    parser.add_argument('--input', required=True, help="Path to cleaned data (CSV or PKL)")
    parser.add_argument('--output', required=True, help="Path to save dataset (PKL)")
    parser.add_argument('--val_frac', type=float, default=0.1, help="Fraction (or count) for validation set per series")
    parser.add_argument('--test_frac', type=float, default=0.1, help="Fraction (or count) for test set per series")
    parser.add_argument('--input_chunk_length', type=int, default=90, help="Input chunk length (for filtering short series)")
    parser.add_argument('--target_col', default='target', help="Name of the target column")
    parser.add_argument('--static_mode', choices=['industry', 'industry_ticker', 'none'], default='industry', help='Static covariates mode: "industry", "industry_ticker", or "none"')

    args = parser.parse_args()
    # 讀取資料
    if args.input.endswith('.pkl') or args.input.endswith('.pickle'):
        df = pd.read_pickle(args.input)
    else:
        df = pd.read_csv(args.input, parse_dates=['date'])
    # 構建數據集
    dataset = build_datasets(df, target_col=args.target_col,
                            val_frac=args.val_frac, test_frac=args.test_frac,
                            input_chunk_length=args.input_chunk_length,
                            static_mode=args.static_mode)
    # 保存
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset built and saved to {args.output}.")
    if 'tickers' in dataset:
        print(f"Total series included: {len(dataset['tickers'])}")

if __name__ == "__main__":
    main()
