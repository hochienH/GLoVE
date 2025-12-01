import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import ListDataset

# 1. 環境設定 (Mac MPS Fallback)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 2. 載入模型
MODEL_PATH = Path("./trained_model")
print(f"Loading model from {MODEL_PATH}...")
predictor = PyTorchPredictor.deserialize(MODEL_PATH)

# 3. 準備檔案列表
INPUT_FOLDER = Path("./gluonTS_Dataset/input/")
files = list(INPUT_FOLDER.glob("*.json"))
np.random.shuffle(files) # 洗牌，隨機排序

print(f"Found {len(files)} files. Looking for valid samples...")

# draw all the figure out and store in a folder
TARGET_COUNT = len(files)  # Set to total files to plot all valid ones
valid_count = 0

# create a directory to save plots
OUTPUT_PLOT_DIR = Path("./visualization_plots")
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

for json_file in files:
    if valid_count >= TARGET_COUNT:
        break
        
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # 相容性讀取
        inst = data if "instances" not in data else data["instances"][0]
        
        target = inst["target"]
        start = inst["start"]
        statics = inst["feat_static_cat"]
        raw_dynamics = inst["feat_dynamic_real"]

        # [Fix Data Leakage]
        # Shift dynamic features by 1 timestep.
        dynamics_arr = np.array(raw_dynamics)
        shifted_dynamics = np.zeros_like(dynamics_arr)
        shifted_dynamics[:, 1:] = dynamics_arr[:, :-1]
        shifted_dynamics[:, 0] = 0
        raw_dynamics = shifted_dynamics.tolist()

        # ==========================================
        # 🛡️ 防呆檢查 (Sanity Check)
        # ==========================================
        # 1. 檢查動態特徵是否存在
        if not raw_dynamics or not isinstance(raw_dynamics, list):
            print(f"⚠️ Skipping {json_file.name}: Bad format")
            continue
            
        # 2. 檢查矩陣是否方正 (Jagged Array Check)
        # 這是造成你剛剛報錯的主因
        expected_len = len(raw_dynamics[0])
        if any(len(f) != expected_len for f in raw_dynamics):
            print(f"⚠️ Skipping {json_file.name}: Jagged array detected (Feature lengths mismatch)")
            continue
            
        # 3. 檢查特徵長度是否等於 Target
        if expected_len != len(target):
            print(f"⚠️ Skipping {json_file.name}: Feature len {expected_len} != Target len {len(target)}")
            continue
            
        # ==========================================
        # 繪圖邏輯
        # ==========================================
        print(f"✅ Plotting {json_file.name}...")
        
        # 我們想要預測最後 5 天 (Prediction Length)
        # 所以輸入給模型的 Target 應該要扣掉最後 5 天
        # 但 Dynamic Features 必須包含這 5 天 (因為模型需要未來的 Feature 來預測)
        prediction_length = 5
        
        input_target = target[:-prediction_length]
        # raw_dynamics 保持原長度 (L)，這樣相對於 input_target (L-5)，它就多出了 5 天的未來特徵
        
        # 建構測試資料集
        test_ds = ListDataset([{
            "start": start,
            "target": input_target,
            "feat_static_cat": statics,
            "feat_dynamic_real": raw_dynamics
        }], freq="D")

        # 進行預測
        forecast_it = predictor.predict(test_ds)
        forecast = list(forecast_it)[0]
        
        # 準備繪圖數據
        plot_len = 120 # 畫過去 120 天 + 未來 5 天
        target_np = np.array(target)
        
        # 建立畫布
        plt.figure(figsize=(12, 6))
        
        # 1. 畫真實值 (歷史 + 未來)
        # 我們只畫最後 plot_len 天
        # 注意：這裡的 target_np 是完整的 (包含最後 5 天)
        history_x = np.arange(len(target_np) - plot_len, len(target_np))
        plt.plot(history_x, target_np[-plot_len:], color='black', label='True Log Volatility', linewidth=1.5)
        
        # 2. 畫預測值
        # DeepAR 的 forecast 物件是從 input_target 的最後一個時間點開始預測
        # 也就是從 len(target) - 5 開始，長度為 5
        pred_len = prediction_length
        pred_x = np.arange(len(target_np) - pred_len, len(target_np))
        
        # P50 (中位數)
        plt.plot(pred_x, forecast.quantile(0.5), color='#FF0000', label='P50 Prediction', linewidth=2)
        
        # P10-P90 (80% 信賴區間 - 淺色)
        plt.fill_between(pred_x, 
                         forecast.quantile(0.1), 
                         forecast.quantile(0.9), 
                         color='green', alpha=0.2, label='10%-90% Confidence')
        
        # P30-P70 (40% 信賴區間 - 深色，讓圖看起來更有層次)
        plt.fill_between(pred_x, 
                         forecast.quantile(0.3), 
                         forecast.quantile(0.7), 
                         color='green', alpha=0.4, label='30%-70% Confidence')

        # 分界線 (昨天 vs 今天)
        plt.axvline(x=len(target_np) - pred_len - 0.5, color='gray', linestyle='--', alpha=0.5)
        plt.text(len(target_np) - pred_len - 0.5, plt.ylim()[1], ' Forecast Start', rotation=90, verticalalignment='top')

        plt.title(f"DeepAR Volatility Forecast: {json_file.name}\n(Alpha-101 Enhanced)", fontsize=14)
        plt.xlabel("Time Steps")
        plt.ylabel("Log Volatility")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the plot to the output directory
        output_plot_path = OUTPUT_PLOT_DIR / f"{json_file.stem}_forecast.png"
        plt.savefig(output_plot_path)
        plt.close()
        
        valid_count += 1

    except Exception as e:
        print(f"❌ Error processing {json_file.name}: {e}")
        continue

if valid_count == 0:
    print("❌ No valid files found for visualization!")