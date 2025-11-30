import pickle
import argparse
import numpy as np
from darts import TimeSeries
from darts.models import TSMixerModel
# 匯入WeightedLoss類，以確保在載入模型時存在類定義
from model_train import WeightedLoss

def main():
    parser = argparse.ArgumentParser(description="Make predictions on test set and evaluate the results.")
    parser.add_argument('--data', required=True, help="Path to dataset pickle from dataset_builder.py")
    parser.add_argument('--model', required=True, help="Path to trained model file (.pth)")
    parser.add_argument('--output', help="Path to output CSV for results")
    parser.add_argument('--use_log_target', action='store_true', help="Set if target was log-transformed (to invert predictions)")
    args = parser.parse_args()
    # 載入資料集
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
    test_targets = dataset['test']['target']
    test_covs = dataset['test']['cov']
    train_targets = dataset['train']['target']
    val_targets = dataset.get('val', {}).get('target', [])
    train_covs = dataset['train']['cov']
    val_covs = dataset.get('val', {}).get('cov', [])
    tickers = dataset.get('tickers', None)
    feature_cols = dataset.get('feature_cols', [])
    # 載入模型
    model = TSMixerModel.load(args.model, work_dir='.')  # work_dir可忽略，因為我們有完整路徑
    # 找出 garch_pred 在特徵中的索引（如果存在）
    baseline_idx = None
    if 'garch_pred' in feature_cols:
        baseline_idx = feature_cols.index('garch_pred')
    results = []
    # 逐檔股票產生預測並計算誤差
    for i, target_ts in enumerate(test_targets):
        if target_ts is None:
            continue
        # 該股票名稱
        ticker = tickers[i] if tickers and i < len(tickers) else f"Series_{i}"
        # 拼接訓練+驗證序列作為歷史輸入（已包含靜態cov）
        if i < len(train_targets):
            history_ts = train_targets[i]
        else:
            continue  # 理論上不會發生，保險起見
        if i < len(val_targets) and val_targets[i] is not None:
            history_ts = history_ts.append(val_targets[i])
        # 拼接協變數歷史
        hist_cov_ts = None
        if train_covs and i < len(train_covs):
            hist_cov_ts = train_covs[i]
        if val_covs and i < len(val_covs) and val_covs[i] is not None:
            hist_cov_ts = hist_cov_ts.append(val_covs[i]) if hist_cov_ts is not None else val_covs[i]
        # 預測（歷史序列的最後一點之後開始）
        start_time = target_ts.start_time()
        pred_ts = model.historical_forecasts(series=history_ts, past_covariates=hist_cov_ts, 
                                            start=start_time, forecast_horizon=1, stride=1, 
                                            retrain=False, verbose=False, last_points_only=True)
        # 提取實際與預測值序列（numpy 陣列）
        actual_vals = target_ts.univariate_values()  # shape (N,)
        pred_vals = pred_ts.univariate_values()      # shape (N,)
        # 基準 GARCH 預測值序列
        baseline_vals = None
        if baseline_idx is not None and test_covs and i < len(test_covs) and test_covs[i] is not None:
            # 提取協變數中的 baseline 分量
            baseline_component = test_covs[i].univariate_component(baseline_idx)
            baseline_vals = baseline_component.univariate_values()
        # 對數還原
        if args.use_log_target:
            actual_vals = np.expm1(actual_vals)
            pred_vals = np.expm1(pred_vals)
            if baseline_vals is not None:
                baseline_vals = np.expm1(baseline_vals)
        # 計算 MAE 和 RMSE
        mae_model = np.mean(np.abs(pred_vals - actual_vals))
        rmse_model = np.sqrt(np.mean((pred_vals - actual_vals) ** 2))
        if baseline_vals is not None:
            mae_garch = np.mean(np.abs(baseline_vals - actual_vals))
            rmse_garch = np.sqrt(np.mean((baseline_vals - actual_vals) ** 2))
        else:
            mae_garch = rmse_garch = float('nan')
        results.append((ticker, mae_model, rmse_model, mae_garch, rmse_garch))
        print(f"{ticker}: MAE_model={mae_model:.6f}, RMSE_model={rmse_model:.6f}, "
              f"MAE_garch={mae_garch:.6f}, RMSE_garch={rmse_garch:.6f}")
    # 保存結果至CSV（如指定）
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["code", "MAE_model", "RMSE_model", "MAE_garch", "RMSE_garch"])
            for row in results:
                writer.writerow(row)
        print(f"Evaluation metrics saved to {args.output}")

if __name__ == "__main__":
    main()
