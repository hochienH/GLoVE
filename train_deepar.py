import json
import glob
import os
import numpy as np
import torch

_original_torch_load = torch.load

def _unsafe_global_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 替換官方函式
torch.load = _unsafe_global_load

from pathlib import Path
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator

# ignore specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-tuple sequence for multidimensional indexing.*")

# ==========================================
# 1. 參數設定 (Configuration)
# ==========================================
INPUT_FOLDER = "./gluonTS_Dataset/input/"
MODEL_OUTPUT_DIR = "./trained_model"

FREQ = "D"
PREDICTION_LENGTH = 5
CONTEXT_LENGTH = 30
EPOCHS = 20
BATCH_SIZE = 32
NUM_LAYERS = 2
HIDDEN_SIZE = 40
LR = 1e-3

# ==========================================
# 2. 資料載入與切分 (Loader & Splitter)
# ==========================================
def load_and_split_datasets(json_folder, prediction_length, freq):
    file_paths = glob.glob(os.path.join(json_folder, "*.json"))
    file_paths.sort()
    
    print(f"🔍 Found {len(file_paths)} json files in {json_folder}")
    
    train_list = []
    test_list = []
    max_ids_per_layer = {}
    skipped_count = 0
    
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)

        # 自動相容模式: 判斷有無 instances 層
        if "instances" in data:
            iterator = data["instances"]
        else:
            iterator = [data]
            
        for inst in iterator:
            target = inst["target"]
            start = inst["start"]
            statics = inst["feat_static_cat"]
            
            # 更新 Static Feature 的最大 ID
            for i, val in enumerate(statics):
                current_max = max_ids_per_layer.get(i, -1)
                if val > current_max:
                    max_ids_per_layer[i] = val

            dynamics = np.array(inst["feat_dynamic_real"]) 

            # [Fix Data Leakage]
            # Shift dynamic features by 1 timestep.
            # Original: dynamics[t] contains info from time t (e.g. Close[t]).
            # Target: target[t] is derived from time t.
            # We must use dynamics[t-1] to predict target[t].
            shifted_dynamics = np.zeros_like(dynamics)
            shifted_dynamics[:, 1:] = dynamics[:, :-1]
            shifted_dynamics[:, 0] = 0  # Pad with 0 for the first step
            dynamics = shifted_dynamics
            total_len = len(target)
            
            min_length = CONTEXT_LENGTH + prediction_length
            if total_len <= min_length: 
                skipped_count += 1
                continue

            cut_idx = total_len - prediction_length
            
            # Test Entry
            test_list.append({
                "start": start,
                "target": target,
                "feat_static_cat": statics,
                "feat_dynamic_real": dynamics.tolist()
            })
            
            # Train Entry
            train_list.append({
                "start": start,
                "target": target[:cut_idx],
                "feat_static_cat": statics,
                "feat_dynamic_real": dynamics[:, :cut_idx].tolist()
            })

    print(f"✅ Loaded {len(train_list)} time series for training.")
    print(f"🚫 Skipped {skipped_count} files (too short).")
    
    # 計算 Cardinality
    num_static_layers = len(max_ids_per_layer)
    cardinality = [max_ids_per_layer[i] + 1 for i in range(num_static_layers)]
    if not cardinality: cardinality = [1]

    # [新增] 計算 Dynamic Feature 數量 (回傳給模型用)
    num_dynamic_features = len(train_list[0]["feat_dynamic_real"])
    
    print(f"📊 Dataset Info: Dynamic Features={num_dynamic_features}")
    print(f"📊 Static Features: {cardinality}")
    
    return (
        ListDataset(train_list, freq=freq), 
        ListDataset(test_list, freq=freq),
        cardinality,
        num_dynamic_features # [🔥 新增回傳值]
    )

if __name__ == "__main__":
    
    # 執行載入
    print("--- 1. Loading Data ---")
    train_ds, test_ds, cardinality_config, num_dyn_feats = load_and_split_datasets(INPUT_FOLDER, PREDICTION_LENGTH, FREQ)

    # 定義模型
    print("--- 2. Building Model ---")
    estimator = DeepAREstimator(
        freq=FREQ,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        
        num_feat_dynamic_real=num_dyn_feats,
        num_feat_static_cat=len(cardinality_config),
        cardinality=cardinality_config,
        
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=0.1,
        
        batch_size=BATCH_SIZE,
        trainer_kwargs={
            "max_epochs": EPOCHS,
            "accelerator": "cpu",
            "devices": 1
        }
    )

    # 開始訓練
    print(f"--- 3. Start Training (Epochs={EPOCHS}) ---")
    predictor = estimator.train(train_ds)
    print("✅ Training Completed!")

    # 保存模型
    print(f"--- 4. Saving Model to {MODEL_OUTPUT_DIR} ---")
    Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    predictor.serialize(Path(MODEL_OUTPUT_DIR))
    print("✅ Model Saved.")

    # 回測評估
    print("--- 5. Evaluating (Backtest) ---")
    
    # 這裡會啟動多進程，有了 if __name__ 保護，子進程就不會再重新跑上面的 Loading Data 了
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  
        predictor=predictor,   
        num_samples=100
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("\n====== 🏆 Evaluation Metrics ======")
    print(f"MSE: {agg_metrics['MSE']:.4f}")
    print(f"RMSE: {agg_metrics['RMSE']:.4f}")
    print(f"Mean wQuantileLoss: {agg_metrics['mean_wQuantileLoss']:.4f}")
    print(f"Quantile Loss [0.5] (Median): {agg_metrics['QuantileLoss[0.5]']:.4f}")
    print(f"Quantile Loss [0.9] (Tail Risk): {agg_metrics['QuantileLoss[0.9]']:.4f}")
    print("===================================")