# GLoVE: GARCH-Loss Based Volatility Forcasting Experiment

## How to reenact the result and further comparison
### 0. Set up Python virtual environment
``` 
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Download newest dataset from **Releases**
- unzip and rename the folder to "Dataset"

### 2. Run following code to complete data preprocessing
```
python src/preprocess.py     --input Dataset/ml_dataset_alpha101_volatility.csv     --output Dataset/clean.pkl     --disabled_features close log_return u_hat_90 gjrgarch_var_90 tgarch_var_90    --use_log_target     --target_col var_true_90     --garch_col garch_var_90 
```
- `--use_log_target`: 將target_col及garch_col數據取ln(1+p)，後續Eval Model時設定需與此相同
- Final data is in Dataset/clean.pkl

### 3. Build dataset (30 secs)
```
python src/dataset_builder.py --input Dataset/clean.pkl  --output Dataset/ts_data.pkl --val_frac 0.2 --test_frac 0.1 --input_chunk_length 90 --static_mode ticker --target_col var_true_90 --garch_col garch_var_90 --scale_target zscore
```
- `--scale_target`: 預設為`none`，可調整為`mean`, `zscore`，對target_col及garch_col標準化，後續Eval Model時設定需與此相同

### 4. Execute deep learning scripts
腳本會自動化進行以下步驟:
1. 以optuna搜尋最佳超參數
2. 對各種lambda分別訓練模型及預測
3. 將各模型結果整合並繪製成圖表

#### TSMixer
```
bash tsmixer_pipeline.sh --parallel 4 --use_log_target --covariate_mode alpha --invert_train_scale zscore
```
#### LSTM
```
bash lstm_pipeline.sh --parallel 4 --use_log_target --covariate_mode alpha --invert_train_scale zscore
```
- `--parallel`: 平行化幾個
- `--use_log_target`: 若Step 2.處理資料時有進行log轉換，則需要在此轉換回來
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`alpha`會使用alpha資料
- `--invert_train_scale`: 若Step 3.處理資料時有進行標準化，則需要在此返還波動度為真實值
- 其餘參數可在.sh檔調整

---
後面還沒整理，但不重要

# Other files
### model_train_tsmixer_optuna_Base.py
- 呼叫 `tsmixer_optuna_runner.py` 搜尋超參數
- 至少要訓練幾次 -> trials
- 輸出最優的超參數 -> trial_csv
- 超參數範圍 -> lr_min, lr_max, search_num_blocks, search_dropout, search_hidden_size, search_ff_size
- 至少要有幾次完整的訓練才開始Pruned裁斷 -> pruner_startup_trials
- 至少要訓練幾個Epoch才開始Pruned裁斷 -> pruner_warmup_steps
- 最多搜尋多久超參數 -> optuna_timeout_sec
```
python src/model_train_tsmixer_optuna_Base.py `
  --data Dataset/ts_data.pkl `
  --trials 1 `
  --epochs 10 `
  --batch_size 32 `
  --lambda_weight 1 `
  --lr_min 1e-4 `
  --lr_max 1e-2 `
  --input_chunk_length 90 `
  --log_dir logs `
  --output_model models/base/tsmixer.pth `
  --trial_csv outputs/optuna_lr_trials.csv `
  --seed 42 `
  --search_num_blocks 1 2 4 `
  --search_dropout 0.1 0.2 0.3 `
  --search_hidden_size 16 32 64 `
  --search_ff_size 16 32 64 `
  --pruner_startup_trials 10 `
  --pruner_warmup_steps 0 ` 
  --patience 2 `
  --optuna_timeout_sec 3600
```

### model_train_tsmixer_optuna_DeepEnsemble.py
- 讀取最優的超參數一組超參數，重新用不同RandomSeed訓練
- trial_csv -> 超參數的檔案位置
- ensemble_runs -> 重新訓練幾次
```
python src/model_train_tsmixer_optuna_DeepEnsemble.py `
  --data Dataset/ts_data.pkl `
  --epochs 10 `
  --batch_size 32 `
  --lambda_weight 1 `
  --input_chunk_length 90 `
  --log_dir logs `
  --output_model models/DeepEnsemble/tsmixer_optuna.pth `
  --trial_csv outputs/optuna_lr_trials.csv `
  --ensemble_runs 10 `
  --seed 42 `
  --patience 2
```

### model_train_tsmixer_optuna_ParamsEnsemble.py
- 功能和 `model_train_tsmixer_optuna_DeepEnsemble.py` 相似
- 讀取最優的超參數`ensemble_runs`組超參數，各訓練一個模型，最終Ensemble
- trial_csv -> 超參數的檔案位置
- ensemble_runs -> 訓練幾組超參數
```
python src/model_train_tsmixer_optuna_ParamsEnsemble.py `
  --data Dataset/ts_data.pkl `
  --epochs 10 `
  --batch_size 32 `
  --lambda_weight 1 `
  --input_chunk_length 90 `
  --log_dir logs `
  --output_model models/ParamsEnsemble/tsmixer_optuna.pth `
  --trial_csv outputs/optuna_lr_trials.csv `
  --ensemble_runs 10 `
  --seed 42 `
  --patience 2
  ```

### model_predict_eval_ensemble.py
- 功能和 `model_predict_eval.py` 相似
- 讀取 `model_base` 裡面的 `ensemble_runs` 個模型，加權平均 Eval 模型結果
- 目前沒有 QLIKE
```
python src/model_predict_eval_ensemble.py `
  --data Dataset/ts_data.pkl `
  --model_base models/DeepEnsemble/tsmixer_optuna.pth `
  --ensemble_runs 10 `
  --output outputs/DeepEnsemble `
  --invert_train_scale zscore `
  --use_log_target 
```

### 4. 訓練模型，自訂部分超參數 (一個iter 100秒)
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`alpha`會使用alpha資料訓練

#### TSMixer
```
python src/TSMixer/model_train.py --data Dataset/ts_data.pkl --lambda 0 --epochs 6 --lr 3e-4 --lr_scheduler exponential --lr_gamma 0.99  --grad_clip 0.5 --hidden_size 32 --ff_size 64 --num_blocks 4 --dropout 0.1 --model_path models/tsmixer_lambda0.pth
```
#### LSTM
```
python src/LSTM/model_train_lstm.py  --data Dataset/ts_data.pkl --lambda 0 --epochs 6 --lr 2e-4 --lr_scheduler exponential --lr_gamma 0.99  --grad_clip 0.5 --hidden_size 32 --dropout 0.1 --model_path models/lstm_lambda0.pth
```

### 5. Predict，輸出結果
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`alpha`會使用alpha資料  
- `--invert_train_scale`: 可調整為`mean`, `none`, `zscore`，返還波動度為真實值
- `--use_log_target`: 若處理資料時有進行log轉換，則需要在此轉換回來
#### TSMixer
```
python src/model_predict_eval.py --data Dataset/ts_data.pkl --model models/tsmixer_lambda0_reviselr.pth --output outputs/lambda_0_1/
```
#### LSTM
```
python src/predict_lstm.py --data Dataset/ts_data.pkl --model models/lstm.pth --split test --output outputs/lambda_0_1 --save_plots
```

### 6. 資料視覺化和資訊整理 
5. 輸出結果 python src/model_predict_eval.py --data Dataset/ts_data.pkl --model models/tsmixer_lambda0_reviselr.pth --output outputs/lambda_0_1/


or lstm prediction
python src/predict_lstm.py \
  --data Dataset/ts_data.pkl \
  --model models/lstm.pth \
  --split test \
  --output outputs/lambda_0_1
  --save_plots

6. 資料視覺化和資訊整理 
python src/data_visualization.py \
  --input outputs/lambda_0_1/metrics.csv \
  --input outputs/lambda_0_1/metrics.png \
python src/data_visualization.py \
  --input outputs/lamb0-iter2-new/metrics_tsmixer_lambda0.csv \
  --output outputs/lamb0-iter2-new/metrics_tsmixer_lambda0.png


# 查看日誌檔案
tail -f logs/tsmixer_lambda0.log

# 查看所有模型檔案
ls -lh models/

# 查看執行中的 Python 進程
ps aux | grep python

7. 用compare_all_model.py來看到底誰更好
python src/compare_all_model.py --input outputs/all_metrics_combined.csv 
以下是每次比較的資料結果
Lambda = 0, garch+tsmixer, epoch=2, lr=2e-4:
--- Average Metrics ---
|            |   Average Value |
|:-----------|----------------:|
| MAE_model  |        0.818576 |
| RMSE_model |        0.981375 |
| MAE_garch  |        0.827502 |
| RMSE_garch |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           27 |           19 |      0 |
| RMSE     |           39 |            7 |      0 |

(.venv) blackwingedkite@keyouqideMacBook-Air ADL-Final % python src/data_visualization.py --input outputs/lamb0-iter10/metrics_tsmixer_lambda0.csv


Lambda = 0, garch+tsmixer, epoch=10, lr=3e-4
--- Average Metrics ---
|            |   Average Value |
|:-----------|----------------:|
| MAE_model  |        0.808775 |
| RMSE_model |        0.97595  |
| MAE_garch  |        0.827502 |
| RMSE_garch |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           34 |           12 |      0 |
| RMSE     |           40 |            6 |      0 |

Lambda = 0, lstm+garch, epoch=10, lr=3e-4
--- Average Metrics ---
| Metric   | Method   |   Average Value |
|:---------|:---------|----------------:|
| MAE      | LSTM     |        0.806965 |
| MAE      | GARCH    |        0.827502 |
| RMSE     | LSTM     |        0.980534 |
| RMSE     | GARCH    |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           45 |            1 |      0 |
| RMSE     |           45 |            1 |      0 |

Lambda = 0, tsmixer+garch, epoch=2, lr=3e-4
--- Average Metrics ---
| Metric   | Method   |   Average Value |
|:---------|:---------|----------------:|
| MAE      | model    |        0.802012 |
| MAE      | garch    |        0.827502 |
| RMSE     | model    |        0.967023 |
| RMSE     | garch    |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           36 |           10 |      0 |
| RMSE     |           41 |            5 |      0 |

Lambda = 0, lstm+tsmixer, epoch=2, lr=3e-4

FAQ:
model在哪裡？
->可以自己訓練或者跟我要我上傳雲端 因為有點多
