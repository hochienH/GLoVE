# How to reenact the result and further comparison
### 0. source .venv/bin/activate
new requirements is built by pipreqs to further allocate pip dependency. 
``` 
pip install pipreqs  
pipreqs ./src  
```
資料集我不提供到github上因為太大了

### 1. download newest dataset from **Releases**

### 2. run following code to complete data preprocessing
```
python src/preprocess.py     --input Dataset/data/ml_dataset_alpha101_volatility.csv     --output clean.pkl     --disabled_features close log_return u_hat_90 gjrgarch_var_90 tgarch_var_90    --use_log_target     --target_col var_true_90     --garch_col garch_var_90 
```
- `--use_log_target`: 將target_col及garch_col數據取ln(1+p)  
Final data is in Dataset/clean.pkl

### 3. build dataset (30 secs)
```
python src/dataset_builder.py --input Dataset/clean.pkl  --output Dataset/ts_data.pkl --val_frac 0.2 --test_frac 0.1 --input_chunk_length 90 --static_mode ticker --target_col var_true_90 --garch_col garch_var_90 --scale_target zscore
```
- `--scale_target`: 可調整為`mean`, `none`, `zscore`，對target_col及garch_col標準化，Eval Model時也需要打開這個功能。

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
#### bash批次訓練
```
bash train_all_lambdas.sh --parallel 4 --nohup
```
- 腳本訓練多組lambda (lambda=0: All GARCH; lambda=1: No GARCH)
- 超參數需至bash檔設定
- 原始的方法如果只用cpi要跑8小時1個epoch我叫gpt幫我生成mac晶片加速跟gpu加速： 結果光是用電腦內建的mac晶片就可以壓到10分鐘內一個epoch

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
#### bash批次預測
```
bash eval_all_lambdas.sh --parallel 4
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

### 如果用bash:使用 ./eval_all_lambdas.sh --parallel 4
6. 資料視覺化和資訊整理 
python src/data_visualization.py \
  --input outputs/lambda_0_1/metrics.csv \
  --input outputs/lambda_0_1/metrics.png \
python src/data_visualization.py \
  --input outputs/lamb0-iter2-new/metrics_tsmixer_lambda0.csv \
  --output outputs/lamb0-iter2-new/metrics_tsmixer_lambda0.png


如何使用自動測試最佳化的腳本？ ＊我有使用平行化加速、背景處理、nohup  
./train_all_lambdas.sh --parallel 4 --nohup  
./train_all_lambdas.sh --parallel 4 --log-dir my_logs
監控訓練進度
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
