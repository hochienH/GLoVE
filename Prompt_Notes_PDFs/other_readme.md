# Other Single Tests
## Training
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`alpha`會使用alpha資料訓練

### TSMixer
```
python src/TSMixer/model_train.py --data Dataset/ts_data.pkl --lambda 0 --epochs 6 --lr 3e-4 --lr_scheduler exponential --lr_gamma 0.99  --grad_clip 0.5 --hidden_size 32 --ff_size 64 --num_blocks 4 --dropout 0.1 --model_path models/tsmixer_lambda0.pth
```
### LSTM
```
python src/LSTM/model_train_lstm.py --data Dataset/ts_data.pkl --lambda 0 --epochs 6 --lr 2e-4 --lr_scheduler exponential --lr_gamma 0.99 --grad_clip 0.5 --hidden_size 32 --dropout 0.1 --model_path models/lstm_lambda0.pth
```
### model_train_tsmixer_optuna_Base.py
- 呼叫 `tsmixer_optuna_runner.py` 搜尋超參數
- 至少要訓練幾次 -> trials
- 輸出最優的超參數 -> trial_csv
- 超參數範圍 -> lr_min, lr_max, search_num_blocks, search_dropout, search_hidden_size, search_ff_size
- 至少要有幾次完整的訓練才開始Pruned裁斷 -> pruner_startup_trials
- 至少要訓練幾個Epoch才開始Pruned裁斷 -> pruner_warmup_steps
- 最多搜尋多久超參數 -> optuna_timeout_sec
```
python src/TSMixer/model_train_tsmixer_optuna_Base.py `
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
## Predict
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`alpha`會使用alpha資料  
- `--invert_train_scale`: 可調整為`mean`, `none`, `zscore`，返還波動度為真實值
- `--use_log_target`: 若處理資料時有進行log轉換，則需要在此轉換回來
### TSMixer
```
python src/TSMixer/model_predict_eval.py --data Dataset/ts_data.pkl --model models/tsmixer_lambda0_reviselr.pth --output outputs/lambda_0_1/
```
### LSTM
```
python src/LSTM/predict_lstm.py --data Dataset/ts_data.pkl --model models/lstm.pth --split test --output outputs/lambda_0_1 --save_plots
```

## Data Visualization 
對單一模型計算所有股票平均誤差，並輸出視覺化圖表
```
python src/data_visualization.py --input {model_output_path}/metrics.csv 
```
對不同lambda模型比較，並輸出視覺化圖表
```
python src/compare_all_model.py --input {outputs}/all_metrics_combined.csv 
```

## Ensemble Strategy
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