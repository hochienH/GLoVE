0. run python -m pip install -r requirements.txt


pip install "u8darts[notorch]"

1. 下載dataset(line-> v3 version)
2. unzip ->get dataset
3. run python3 gluon_data_preprocessor.py ./Dataset/data/ml_dataset_alpha101_volatility.csv --standardize -> get gluonTS_Dataset
4. ginn0_dataset.py  把eval_compare的 load_garch_data 抽乾淨，變成「回傳每檔股票的 time series」。這個檔案的重點就是：跟 DeepAR 的 evaluation code 脫鉤，專心只管 GARCH + ground truth。

接下來根據蔡老闆的指示做一些檢查與架構重構
你貼的這段超關鍵，我直接翻成「接下來 Darts/GNN-0 要怎麼設計」：
# 檢查標準化流程
- Alphas 截面Rank[-1, 1]
- Alphas + GARCH(Past) 時間序列Normalize[0, 1]
- RevIN (Darts的套件有), 主要用來做時間序列的輸入、輸出標準化

🔧 3.1 Feature & Scaling 設計
對 46 檔股票、每天𝑡:
 - Cross-sectional rank for alphas：
 - 同一天對所有股票的每一個 alpha 做截面 rank → 映射到[−1,1]。
這一步是「橫向」：同一天看全部股票。

Time-series normalize for each series：
對每一檔股票、每個 feature（包含 alphas + GARCH past）做 per-series normalization（例如 min-max 到 [0,1]，或 z-score）。
這步是「縱向」：每條時間序列各自正規化。

RevIN：
- 用 Darts 的 RevIN transformer 包住模型前後：輸入先標準化、模型輸出再逆轉標準化。
- 好處：可以對付 non-stationary / scale shift 問題。

結論：原始 GINN 是 univariate + AR/GARCH，
你們可以做成「多變量 features + RevIN + GINN-0」，
算是架構上的 upgrade，但論文精神還在。

對應的code: in sanity_check/











5. train_ginn0.py:    assert mode in {"lstm", "ginn0", "garch"}, "mode must be one of: lstm | ginn0 | garch"
訓練模型的地方 有三種模式，garch不用動腦 
# 單獨跑 GARCH baseline
python train_ginn0.py --mode garch --save_path results_garch.csv

# 跑 LSTM baseline
python train_ginn0.py --mode lstm  --save_path results_lstm.csv

# 跑 GINN-0
python train_ginn0.py --mode ginn0 --save_path results_ginn0.csv
