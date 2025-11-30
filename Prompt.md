# 模型概述與預測任務

你是一名熟悉 **Python、PyTorch、Darts**，並對 **金融時間序列與波動度建模** 有實務經驗的工程師兼研究員。

你的任務是：根據下列明確規格，為一個 **多資產 realized volatility 預測專案**，設計並實作 **完整的建模 pipeline**（含資料前處理與模型訓練），使用 **Darts 的 `TSMixerModel` 並開啟 RevIN**。

請在回答時：

- 以 **繁體中文** 解釋與說明設計；
- 提供 **可執行的 Python 程式碼**（程式碼與變數用英文，附註解）；
- 嚴格遵守下列資料前處理與模型設定規範；
- 如有任何無法確定的地方，先列出你的假設再依此實作。

## 問題定義：
我們針對 48 檔股票的波動度進行多資產時間序列預測，每日預測下一日的波動度。為此，我們採用 Darts 時間序列庫的 TSMixerModel（時間序列混合器）結合 Reversible Instance Normalization (RevIN) 技術。一個單一的 TSMixer 模型可同時訓練多條序列，透過結合歷史目標值、協變數（covariates）及靜態特徵來進行預測。模型在此設定為全域模型，即一個模型學習所有 48 檔股票的模式。

## TSMixer 特性：
TSMixer 是一種全連接神經網路架構，可處理時間維度與特徵維度的混合。它允許使用過去協變數（past covariates）和未來協變數（future covariates）以及靜態協變數等多種輸入。在我們的任務中，我們將歷史的價格與衍生特徵作為動態協變數，同時為每檔股票設置靜態標識特徵。模型設定輸入序列長度為 90（即用過去 90 天資料預測未來），預測輸出長度為 1（單步預測未來 1 天）。因此每次訓練樣本包含前 90 天的資料來預測下一日的波動度 (horizon=1)。

## **RevIN 技術：**
為降低不同時期資料分布偏移對模型的影響，我們在模型中啟用了 Reversible Instance Normalization (RevIN)。RevIN 會對每條序列的目標值在輸入模型前後進行標準化及還原，協助模型在訓練過程中對抗分布變化問題。這對金融時間序列（如波動度）尤為重要，可提升模型在不同市場階段的準確度。啟用 RevIN 只會對目標序列特徵應用正規化，不改變協變數。

# 請假設專案資料夾結構如下，並依此結構生成程式碼：
Dataset/data: 存放 raw csv
src/: 存放 .py 腳本 (preprocess.py, dataset_builder.py, model_train.py, model_predict_eval.py)
logs/: 存放 tensorboard logs
models/: 存放 .pth 模型

# 資料欄位與特徵工程流程

原始資料假設來自每日收盤後的計算，包含以下主要欄位：

## 📦 資料欄位說明
| 欄位 | 說明 |
|------|------|
| `code` | 股票代碼（48 檔，後續可用於 ticker one-hot static cov） |
| `date` | 日期 |
| `var_true_90` | target：真實 90 日波動度/變異數 |
| `garch_var_90` | GARCH 模型預測的 90 日波動度/變異數 |
| `revenue_report` | 營收公布日 dummy（0/1，當日有營收公告為 1） |
| `fin_report` | 財報公布日 dummy（0/1，當日有財報公告為 1） |
| `industry_code` | 產業代碼（約 20 個產業，會被轉成 static one-hot） |
| `alpha001` ~ `alpha171` | 171 個 Alpha 因子（截面 rank-normalized 至 [-1,1]） |
| `close` | 收盤價（可選特徵，預設關閉） |
| `log_return` | 日對數報酬（可選特徵，預設關閉） |
| `u_hat_90` | 其他 90 日相關指標（可選特徵，預設關閉） |

資料預處理需對上述欄位進行清洗和特徵轉換。我們的 preprocess.py 腳本執行以下步驟：

**移除停用特徵：**根據使用者設定的 disabled_features 清單，從資料中刪除不希望納入模型的特徵欄位。例如，可以選擇不使用 close、log_return 或 u_hat_90 等欄位，以觀察不同特徵組合對模型的影響。移除後的資料只保留需要的特徵。

**特徵與標籤的 Burn-in / Warm-up 處理：**

由於部份特徵（covariates）以及目標 `target_col` 本身是由 rolling window 或模型估計而來，
最前面一段會是 burn-in / warm-up 期，這段期間的值通常是：

- `NaN`（例如 rolling 尚未滿足視窗長度）；或
- 數量太少、不穩定、不希望拿來訓練。

為了避免模型學到這些不穩定區間，我們需要對所有股票做一致的 burn-in 切齊處理，規則如下：

1. **對每一個使用中的欄位（包含 covariates 與 `target_col`），先確定它的 burn-in 長度**  
   - 例如：
     - `garch_var_90` 是用 90 天 window 計算 → burn-in = 90；
     - 某些 `alphaXXX` 需要 250 天 rolling → burn-in = 250；
     - `var_true_90` 若本身也是 90 天 realized vol → burn-in = 90。
   - 這可以透過「設定檔 + 常數」或「實際掃描第一個非 NaN index」來得到。

2. **計算全體最大 burn-in 長度 $T$**  

   設所有會用到的欄位集合為 $\mathcal{F}$，每個欄位的 burn-in 長度為 $T_f$，則定義：
   $T = \max_{f \in \mathcal{F}} T_f$
   這個 $T$ 代表：要讓「所有特徵與 `target_col` 都已經穩定可用」，至少要先丟掉的天數。

3. **對每一檔股票 `code`，從自己的起始日往後丟掉前 $T$ 天的資料**

   對於每檔股票：

   - 先把該股票的資料依 `date` 排序；
   - 設該股票的第一筆觀測日期為 $t_0$，則只保留從 $t_0 + T$ 之後（含當天）的資料；
   - 等價寫法：對每檔股票，根據列索引丟掉前 $T$ 個樣本。

   如此處理後：

   - 不論是哪一個 covariate 或 `target_col`，在保留下來的區間中**都不應該再有「因為 burn-in 而產生的缺值」**；
   - 例如：若 `garch_var_90` 有 90 天 NaN、某個 `alpha` 有 250 天 NaN，那 $T = 250$，
     每檔股票都從自己起始日的第 250 天開始算作「可訓練區間」。

4. **晚上市或資料較短的股票處理**

   - 若某檔股票上市較晚、歷史樣本較短，在切掉前 $T$ 天之後，實際剩餘長度不足
     `input_chunk_length + 1`（例如小於 91 天，無法組出一個 90→1 的樣本），  
     則這檔股票**不納入訓練 / 評估**。
   - 這個判斷在 `dataset_builder.py` 裡實作：對每檔股票先做 burn-in 切齊，再檢查剩餘長度是否達門檻。

5. **實作原則**

   - `preprocess.py` / `dataset_builder.py` 對 burn-in 區間 **不做插補（ffill/bfill）**；
   - 直接依上述規則切掉 burn-in 區間，保證送進模型的每一行資料，在所有 covariates 與 `target_col` 上都已過 burn-in、為有效值。

**Alpha 因子補值與排名標準化：**針對 alpha 因子欄位，先對缺失值進行補值，再進行橫截面排名標準化。補值方面，每日將該日所有股票的 alpha 值以當日現有值的中位數代替缺失值（若當日所有股票均缺失則以 0 代替）。接著，對每個交易日，將當日所有股票的 alpha 值按照數值排序，轉換為 0～1 區間的排名百分比，最後再線性映射到 [-1, 1] 區間。映射公式為： $\text{alpha\_scaled} = 2 \cdot \frac{\text{rank}-1}{N-1} - 1$, 其中 $N$ 是當日有報告數值的股票總數。經此轉換，每日 alpha 最高的股票得到 +1，最低得到 -1，中位數接近 0。這種橫截面標準化有助於消除 alpha 值在不同日期和股票之間的量綱差異，使模型更關注相對高低而非絕對大小。

**營收公布日/財報公布日：**（`revenue_report`, `fin_report`）
- `revenue_report` 與 `fin_report` 為 **0/1 dummy**，代表當日是否有營收或財報公告。
- 這兩個欄位：
  - 在 `preprocess.py` 中只做型別與缺失處理（缺失補 0，轉成 `int`），**不做任何 scaling**。
  - 在 `dataset_builder.py` 中被當成 **past covariates** 的一部分，隨時間變化餵給 TSMixer。
- 模型的解讀就是：特定日子前後，這兩個 dummy 的 pattern 會影響下一日波動度。

### 產業變數（`industry_code`）與 ticker static covariates

- `industry_code` 表示該股票所屬產業（約 20 類），是不隨時間變化的類別特徵。
- `code`（或 `ticker`）是個股 ID，本身也是一個類別，但主要代表「單一股票」，沒有明確產業結構。

在本 pipeline 中，static covariates 的設計支援兩種模式，由 `STATIC_MODE` 控制：

1. `STATIC_MODE = "industry"`  
   - 每檔股票的 static covariate = **industry_code one-hot**  
   - 例如 industry 有 20 類，就產生長度 20 的 one-hot 向量 `[industry_bank, industry_tech, ...]`。
   - 模型透過這個向量知道「這條時間序列屬於哪個產業」，學習產業層級的長期結構差異。

2. `STATIC_MODE = "industry_ticker"`  
   - 每檔股票的 static covariate = **industry_code one-hot + ticker one-hot** 串接在一起  
   - 例如 industry 有 20 類、股票有 48 檔，static 向量長度 = 20 + 48：  
     - 前 20 維：產業 one-hot  
     - 後 48 維：ticker one-hot  
   - 這樣模型既能看到產業層級結構（前 20 維），也能對每一檔股票學一些 idiosyncratic 的偏移（後 48 維）。

3. `STATIC_MODE = "none"`  
   - 不使用 static covariates，模型只靠過去目標與 covariates 自行區分不同股票。

在 `dataset_builder.py` 中會根據 `STATIC_MODE`：
- 收集所有 `industry_code` → 建立 industry one-hot；
- 收集所有 `code` → 建立 ticker one-hot（只有在 `"industry_ticker"` 模式下才會用）；
- 將組合後的 static 向量透過 `with_static_covariates()` 附加到每條 target TimeSeries 上。

**GARCH 特徵縮放：**對 GARCH 預測波動度特徵 (garch_pred) 進行標準化處理。此處提供兩種縮放選項：

**Global Scaling：**以所有資產整體數據計算均值和標準差，將 garch_pred 在全域範圍內標準化。即 $z = (x - \mu_{\text{all}}) / \sigma_{\text{all}}$，將所有股票的 GARCH 值轉換成同一尺度。

**Per-series Scaling：**以每檔股票自身的歷史均值和標準差進行標準化。**歷史均值和標準差需要嚴格遵守避免未來資料洩漏的問題，歷史均值、標準差應以合理的方式計算**。每檔股票的 GARCH 序列會被調整為該股票上的 $z$ 分數（平均為0，方差為1）。使用者可透過參數 --garch_scaling 選擇 "global" 或 "per_series" 模式（或 "none" 表示不縮放）。適當的縮放可以避免不同股票之間 GARCH 數值量級差異過大，影響模型訓練的穩定性。

**對數目標轉換：**由於波動度（或方差）資料經常具有右偏分佈且範圍跨越數量級，我們可以啟用 USE_LOG_TARGET=True 對目標值進行對數轉換。具體做法是在資料中將目標欄位取 $\log(1+x)$（自然對數），以緩解極端值和偏態對模型的不利影響。對應地，若有使用 GARCH 預測值為基準比較，也對 GARCH 預測值做相同的 $\log(1+x)$ 轉換，確保兩者處於同一尺度上。日後在模型預測輸出時，再使用反轉換 $\exp(x) - 1$ 將結果還原回原始尺度。透過對數化處理，模型更容易學習線性關係，同時確保預測值為非負。

preprocess.py 可接受 --use_log_target 參數啟用此步驟。啟用時會對目標欄位和 garch_pred 欄位都執行 log1p 轉換，並記錄此設定以便在預測階段反轉。

完成上述處理後，preprocess.py 會輸出清理轉換後的資料表（可選擇輸出為 CSV 或 pickle 格式）。其中每列包含：date、code、目標欄位（可能已取對數）、以及經處理後的協變數欄位（移除了停用特徵、填補並縮放了 alpha、縮放了 garch 等）。例如：

date，code，target（可能為 log(vol)），alpha（已轉換為 [-1,1]）、garch_pred（縮放後，可能亦已取 log1p）、以及保留的其他特徵（如 log_return 等）。

## preprocess.py 模組參數與使用方式

以下是 preprocess.py 腳本的主要參數及使用示例：

--input：輸入原始資料檔案路徑（CSV）。(必填)

--output：輸出處理後資料的檔案路徑（建議用 .pkl 保存 DataFrame，或 .csv）。(必填)

--disabled_features：可選，多個欄位名稱，指定需從資料中移除的特徵。舉例：--disabled_features close log_return u_hat_90 將移除收盤價、對數報酬和 u_hat_90 特徵。

--garch_scaling：可選，none 或 global 或 per_series，設定 GARCH 特徵的縮放方式。預設為不縮放或原樣使用。

--use_log_target：flag 開關，若提供則對目標值及 GARCH 特徵套用對數轉換。

--target_col：可選，目標欄位名稱（預設為 "target"）。若資料中的目標欄位名稱不同，透過此參數指定。

使用範例：

python preprocess.py \
    --input raw_data.csv \
    --output cleaned_data.pkl \
    --disabled_features close log_return u_hat_90 \
    --garch_scaling per_series \
    --use_log_target \
    --target_col realized_vol


上述命令將讀取 raw_data.csv，移除 close、log_return、u_hat_90 三個欄位，對每檔股票各自標準化其 garch_pred，對 realized_vol 目標和 garch_pred 取對數，處理 alpha 後，將結果保存為 cleaned_data.pkl。

## 時間序列資料集構建 (dataset_builder.py)

經過預處理的資料將用於構建 Darts 模型所需的時間序列物件。dataset_builder.py 腳本負責將整理後的資料轉換為 Darts 的 TimeSeries 結構，並按照訓練、驗證、測試區間進行劃分。

1. 載入清理後資料：
腳本接受前一階段輸出的資料檔（CSV 或 pickle），讀入為 DataFrame。資料中每條記錄帶有 date、code、目標值和多個特徵欄位。建議將 date 解析為日期時間型別，並確保各股票序列按照日期排序。

2. 靜態與動態協變數：
Darts 允許每條序列攜帶靜態協變數（時間不變的特徵）和動態協變數（隨時間變化的特徵）。在本案例中：

靜態協變數：我們為每檔股票引入一組靜態特徵，用於標識該股票或產業。例如，我們使用股票身份的一位有效編碼（one-hot encoding）向量作為靜態特徵：48 檔股票即對應長度為 48 的向量，每檔股票在自己對應的位置為1，其餘為0。這相當於為模型提供股票的類別資訊。使用靜態特徵需要在模型中設定use_static_covariates=True，模型在訓練時會確保所有目標序列具有相同維度的靜態向量並利用它們。靜態協變數可以幫助模型識別不同序列的特性，例如區分不同股票或不同產業的波動水準。

動態協變數：則包括我們在預處理後保留下來的所有時間序列特徵（除了目標值本身）。例如，alpha（已標準化）、garch_pred（已縮放/取對數）、以及可能保留的 log_return 等均作為協變變數隨時間變動。這些特徵會被作為過去協變數（past covariates）提供給模型，也就是說，在預測下一步時，模型可使用最近 90 天這些特徵的歷史值作為輸入。預設情況下沒有使用未來協變數，因為未來協變數須為已知的未來資訊（例如節假日、計畫公告等）；我們的特徵如價格或收益在未來未知，因此不作 future covariate。如 GARCH 預測值因為代表下一日的已知預測，理論上可作為 future covariate，但此處我們直接將其與目標對齊作為過去特徵使用（相當於每一天用GARCH對隔日的預測值作為當日特徵）。模型在預測時不需要未來協變數的延伸。

**3. 序列劃分：**腳本將每檔股票的時間序列按照訓練集、驗證集和測試集進行拆分：

訓練集 (train)：用於模型訓練的期間資料。

驗證集 (val)：在訓練過程中進行模型性能評估調整的資料，不參與模型權重更新，只用來調參或早停。

測試集 (test)：完全獨立的資料，用於最終評價模型預測效果。

用戶可以透過參數指定驗證集和測試集的比例或長度（例如 --val_frac、--test_frac）。默認情況下腳本會將每條序列的最後 10\% 作為測試，往前的 20\% 作為驗證，其餘最前 70\% 作訓練（可根據需求調整）。劃分時確保時間順序：驗證集緊接在訓練集之後，測試集為最後的部分，且各部分不重疊。例如，若某股票有 1000 筆日資料，則默認劃分約為前 700 筆為訓練、接下來 200 筆驗證、最後 100 筆測試。

註意：如某些股票資料很短（短於模型所需的最小序列長度，如 90 天），或扣除驗證/測試後訓練資料不足，這些序列將自動跳過不納入模型訓練。確保每個用於訓練的序列至少有 input_chunk_length 長度的樣本可提取。

4. 建立 Darts TimeSeries：對於每檔股票，我們利用 Darts 的 TimeSeries.from_dataframe 方法將整個序列轉換為 TimeSeries 物件，然後根據上述切點切分。對於每檔股票我們產生:

target TimeSeries：僅包含目標值（單變量序列），並帶有該股票的靜態特徵向量。透過 with_static_covariates() 方法將靜態 DataFrame 附加到 TimeSeries。靜態特徵在所有拆出的子序列中相同。例如，股票 A 的訓練序列和驗證序列都攜帶同一靜態 one-hot 向量，模型因而識別它們同屬股票 A 或對應產業。

past covariates TimeSeries：包含該股票的所有協變特徵，多變量序列（每個時間點有多個特徵值）。如果在預處理中移除了所有額外特徵，則此部分為空或 None。對於每檔股票，協變數序列與目標序列在時間索引上對齊。這些協變數僅用作過去資料輸入，並不會在未來提供超出觀測範圍的值。

腳本將每檔股票切分後的部分加入對應的列表中：train_targets、val_targets、test_targets 以及 train_covariates、val_covariates、test_covariates 列表。若某部分沒有資料則相應地略過或使用空列表表示。

完成後，我們得到數據集中包含多個序列的列表。例如，train_targets 是一個長度最多 48 的列表，其中每個元素是一檔股票訓練區間的目標 TimeSeries；train_covariates 則是對應的協變數 TimeSeries 列表。這些列表將在模型訓練時傳入模型的 fit() 方法。為了方便後續使用，腳本會將上述結構（列表或者字典）序列化保存（如保存為 .pkl pickle 檔）。

dataset_builder.py 模組參數與使用方式

--input：預處理後資料的路徑（CSV 或 PKL）。(必填)

--output：輸出資料集物件的路徑（建議 .pkl）。(必填)

--val_frac / --test_frac：驗證集和測試集所佔序列長度比例（0~1之間浮點數）。也可使用 --val_size / --test_size 指定固定的天數長度。若兩者皆提供，按比例優先。預設例如 --val_frac 0.2 --test_frac 0.1。

--input_chunk_length：模型輸入序列長度，用於過濾太短的序列（預設 90，同模型設定）。

--static：靜態特徵編碼方式，one-hot（預設）或 none。若為 one-hot，則每檔股票會生成對應的 one-hot 靜態向量；若 none，則不使用靜態協變數。

其他：--target_col（如需要覆蓋目標欄位名），--use_log_target（與預處理一致，用於記錄在資料集中以便預測時知道需要反轉）。

使用範例：

python dataset_builder.py \
    --input cleaned_data.pkl \
    --output dataset.pkl \
    --val_frac 0.2 \
    --test_frac 0.1 \
    --input_chunk_length 90


此命令將讀取 cleaned_data.pkl，對每檔股票按最後10\%做測試、次後20\%做驗證，其餘為訓練，構建對應的 TimeSeries 列表，並保存為 dataset.pkl 以供後續訓練使用。

## 模型訓練 (model_train.py)

在此步驟，我們載入先前構建的時間序列資料集，並開始訓練 TSMixer 模型。主要工作包括模型初始化、定義自訂損失函數、設置訓練參數，然後執行訓練過程。

**1. 載入資料集：**從 dataset.pkl 讀取先前生成的 train 和 val 序列列表。我們會得到諸如 train_targets（列表內每項為一檔股票的目標 TimeSeries）、train_covariates（列表內每項為對應股票的協變數 TimeSeries）。同樣地，如果有驗證集，會有 val_targets 及 val_covariates 列表。在此基礎上，我們準備進行全局模型訓練：即將 train_targets 列表中所有序列同時提供給模型訓練（模型會視不同序列為樣本的不同instances）。

**2. 模型初始化：**我們建立一個 Darts 的 TSMixerModel 實例：

**input_chunk_length=90：**設定模型輸入長度為90天，對應我們使用過去90個時間步長的資料作為輸入。

**output_chunk_length=1：**設定單步輸出。這意味著模型一次只產生1天的預測，對應 horizon=1 的情況。由於我們將進行滾動預測，此設置允許模型以自回歸方式多步預測。

**use_static_covariates=True：**啟用靜態協變數使用。模型會將我們嵌入在 TimeSeries 中的靜態one-hot向量作為每條序列的固定特徵，使模型可以學習不同股票與產業之間的差異。

**use_reversible_instance_norm=True：**啟用可逆實例標準化 (RevIN)。這將在模型內自動對每條序列的目標值進行尺度標準化並在預測輸出時還原，以降低訓練和測試分布不一致的影響。

**loss_fn=WeightedLoss(lambda)：**我們自定義了損失函數，將其傳入模型。預設情況下 TSMixer 使用均方誤差損失；我們在此定義 WeightedLoss 來實現組合損失：
$\mathcal{L}= \lambda \cdot \mathrm{MSE}(y_{\text{true}},\, y_{\text{pred}})+\;(1-\lambda)\cdot \mathrm{MSE}(y_{\text{GARCH}},\, y_{\text{pred}})$, 其中 $y_{\text{pred}}$ 是模型預測的波動度，$y_{\text{true}}$ 是實際真值波動度，$y_{\text{GARCH}}$ 是傳統 GARCH 模型對該日的預測值。我們透過係數 $\lambda \in [0,1]$ 在貼近真實值和貼近 GARCH 基準之間做權衡。當 $\lambda=1$ 時損失退化為普通 MSE，只對齊真實值；$\lambda=0$ 則模型完全按 GARCH 預測校準。適當選擇 $\lambda$（如 0.5）可讓模型在學習數據趨勢的同時，不偏離現有GARCH模型的穩健性。

**實現方式：**我們將 GARCH 預測值視作另一條“目標”序列，在損失計算時同樣計入。訓練時模型仍然只預測波動度，但損失函數的第二項相當於對預測與baseline之間的偏差加以懲罰，促使模型預測不偏離 GARCH 太遠。這一自訂損失可以通過自定義 torch.nn.Module 並傳給 Darts 模型的 loss_fn 參數實現。

**關於 Weighted Loss 的實作關鍵細節：** 由於 Darts 的 loss_fn 形式限制，為了讓 Loss 函數能存取 GARCH 基準值，請採用 Multivariate Target 策略：

1. 在 dataset_builder.py 中，建構 train_targets 時，請將 target_col (真實波動度) 與 garch_col (GARCH預測值) 合併，形成一個 Dimension=2 的 TimeSeries (例如：Component 0 為 True Vol, Component 1 為 GARCH)。

2. 若啟用了 use_log_target，這兩個分量都必須做 log1p 轉換。

3. 在 model_train.py 的 WeightedLoss 中，計算邏輯如下：
    - 假設 input (y_pred) 和 target (y_true) 形狀皆為 (batch, n_timesteps, 2)
    - 只優化第一個分量 (模型預測值)，忽略模型對 GARCH 的預測(若有的話)
    - pred_vol = input[:, :, 0]
    - true_vol = target[:, :, 0]
    - ref_garch = target[:, :, 1]
    - loss = lambda_coeff * mse(pred_vol, true_vol) + (1 - lambda_coeff) * mse(pred_vol, ref_garch)
4. 在 model_predict_eval.py 中，取得預測結果後，只取 Component 0 作為最終預測出的波動度。

**其他超參數：**我們可設定隱層大小、dropout等（使用默認值或根據需要調整）。另外，我們傳入 random_state 確保可重現性，並啟用 log_tensorboard=True 讓訓練過程記錄至 TensorBoard 日誌。在 work_dir 中指定日誌目錄（例如 --log_dir logs/），Darts 會將日誌寫入 {work_dir}/darts_logs/{model_name}/logs/。我們也可指定 model_name 以控制檔名。

為了使用 GPU 加速訓練，我們利用 PyTorch Lightning 的設定：pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True} 來讓 Darts 在有 GPU 時自動使用。如無 GPU，則回退使用 CPU。這確保程式可在支援 CUDA 的環境下利用 GPU 訓練。

**3. 模型訓練：**呼叫 model.fit() 開始訓練。訓練時：

傳入參數 series=train_targets 列表，以及 past_covariates=train_covariates（如果存在協變數）。

傳入 val_series=val_targets 及 val_past_covariates=val_covariates 作為驗證資料。模型將在訓練過程中每個 epoch 後對驗證集計算損失以供參考，並可用於模型選擇或早停。

指定訓練 epoch 數（如 --epochs 100）、batch size（如 --batch_size 32）等。TSMixer 本質上是 batch 訓練，每個 batch 會隨機採樣多條序列的片段進行梯度更新。

設定隨機種子以固定初始化和批次順序。

模型會使用我們定義的 WeightedLoss 作為優化目標。隨著訓練，模型將學習到同時逼近真實波動度並貼近 GARCH 基準。訓練過程中，可以透過 TensorBoard 查看損失的下降趨勢以及訓練集和驗證集上不同損失項的變化。

**4. 模型保存：**訓練完成後，將模型權重與結構保存至檔案（例如 model.pth）。這允許之後載入模型進行預測而無需重訓。Darts 提供 model.save(path) 方法保存模型（底層使用 Python pickle）。我們需確保在保存和加載模型時自訂的損失類別是可被識別的（已在執行環境中定義），否則在加載時會遇到錯誤。為此，在預測腳本中我們會重新定義或引用同一 WeightedLoss 類。

model_train.py 模組參數與使用方式

--data：資料集檔案路徑（dataset.pkl）。(必填)

--lambda：損失函數中的權重係數 $\lambda$，介於0~1。決定真實值MSE所佔權重。

--epochs：訓練週期數。

--batch_size：訓練批次大小。

--seed：隨機種子，用於初始化權重及隨機過程。

--log_dir：TensorBoard 日誌目錄。若不提供則默認當前路徑。

--model_path：模型保存路徑。預設為當前目錄下 tsmixer_model.pth。

使用範例：
python model_train.py \
    --data dataset.pkl \ 
    --lambda 0.5 \
    --epochs 50 \
    --batch_size 32 \ 
    --seed 42 \
    --log_dir logs \
    --model_path tsmodel.pth


上述命令將讀取 dataset.pkl，以 $\lambda=0.5$ 訓練模型 50 個 epoch，每批次32條序列片段，使用隨機種子42，在 logs/ 下記錄TensorBoard日誌，並將模型保存為 tsmodel.pth。

## 模型預測與評估 (model_predict_eval.py)

訓練完成後，我們使用測試集資料來評估模型的預測性能，並與 GARCH 基準進行比較。model_predict_eval.py 腳本負責載入訓練好的模型，對每檔股票在測試期間進行逐日滾動預測，還原對數變換，計算誤差指標，並將結果輸出。

1. 載入模型與資料：

腳本將先透過 TSMixerModel.load(model_path) 載入訓練好的模型權重。如前所述，需要確保載入環境中定義了與訓練時相同的 WeightedLoss 類，以順利重建模型（本腳本會匯入或定義相同的損失類）。接著，載入先前的資料集 dataset.pkl 以取得測試所需的序列。如果為簡化流程，我們也可以只載入預處理後的完整資料檔，再次構建完整序列，不過直接使用已分割的資料集更方便。我們將提取每檔股票的完整目標序列（包含訓練+驗證+測試全段），或者至少包含到測試開始前的序列段和測試段，以供生成預測。

2. 逐日滾動預測：

對於每檔股票，在測試集的第一天之前，模型都已經觀察到了該股票訓練+驗證期間的所有實際數據（因為我們每一天預測隔日，並在隔日實際值出現後再繼續預測下一天）。我們採用 Darts 提供的 historical_forecasts() 方法對每條序列進行多步滾動預測：設定 start 為該股票測試區間的開始日期，forecast_horizon=1，stride=1，且 retrain=False（不在滾動時反覆重訓模型）。在 historical_forecasts 過程中，模型會從 start 時刻開始，每次向前滾動一步進行 1 天預測。由於我們使用 last_points_only=True（預設值），模型在每個滾動步都僅產生該步的一點預測並將其串連成預測序列。實際上，這相當於每天收盤後利用當天實際數據更新模型輸入，預測下一天波動度。也就是說：

模型在預測測試期第1天時，用到了訓練+驗證期最後90天的真實數據（模型未看過第1天的實際值）。

預測出第1天的波動度後，第1天結束時我們會知道實際的第1天波動度（假設可由日內數據計算得到），模型在預測第2天時即可將第1天實際值納入歷史序列。

以此類推，模型在預測測試期每一天時，都有該股票前一天及更早的真實波動度可供參考，這符合真實預測情境（我們每天都能利用最新實際資料來預測下一天）。

如此，historical_forecasts 產生的預測序列將和測試集實際序列一一對齊。在實現中，我們給 historical_forecasts 傳入每檔股票完整的目標序列（包含靜態特徵）以及完整的協變數序列，再指定 start=測試開始日期，它會返回從該日期起模型對每一步的預測結果 TimeSeries。**注意：**如果協變數包含未來未知的量（例如未來的價格、報酬），理論上在滾動預測中不應使用。由於我們僅把協變數限制在過去值（模型每一步只用過去90天協變數），因此在historical_forecasts中，我們仍傳入相同的 past covariates 序列即可。Darts 會自動在需要時截取相應窗口；對於每一步，未來的協變數不會被用到或模型會跳過缺失的未來協變數。因此模型的預測是根據已知的歷史數據進行的，不會偷看未來真值。

3. 預測結果反轉與評估：

如果在預處理時對目標做了對數變換，我們需要對模型預測值做逆變換才能與原始尺度的真實值比較。具體而言，對每條序列預測出的 TimeSeries pred_series，我們對其中的值套用 $\exp(x)-1$（即 np.expm1）轉回原始波動度。同樣地，真實目標值序列以及 GARCH 基準序列若也是以對數存儲的（在 dataset.pkl 中可能為 log尺度），也一併反轉。完成後，我們獲得原尺度上的預測值、實際值和基準值。

隨後，我們計算誤差指標來評估模型準確度：

平均絕對誤差 (MAE)： $\frac{1}{N}\sum_{t=1}^{N} |y^{\text{pred}}_t - y^{\text{true}}_t|$

均方根誤差 (RMSE)： $\sqrt{\frac{1}{N}\sum_{t=1}^{N}(y^{\text{pred}}_t - y^{\text{true}}_t)^2}$

我們會為每檔股票各計算模型預測與實際值之間的 MAE 和 RMSE。同時也計算 GARCH 預測與實際之間的 MAE、RMSE 作為基準比較。這能反映模型是否優於傳統基準。計算結果將輸出至終端，並彙總保存到 CSV 檔，其中每行包含股票代號、模型 MAE、模型 RMSE、GARCH MAE、GARCH RMSE。

4. 分析結果：

在結果中，若模型優於基準，我們預期模型的 MAE、RMSE 會較 GARCH 的低。同時，由於我們在訓練中加入了 GARCH 誤差項，模型預測可能在趨勢上貼近 GARCH，但希望能稍微修正 GARCH 的系統性偏差而取得更好準確度。透過調整 $\lambda$，我們可以權衡偏向真實數據 vs. 偏向基準模型的程度。例如，$\lambda=0.5$ 給予兩者同等權重；若 $\lambda$ 較大，模型會更自由地偏離 GARCH 以貼合真值，誤差較小但風險較高；$\lambda$ 較小則模型預測更接近 GARCH，可能降低誤差但也限制了模型學習數據特性的能力。

model_predict_eval.py 模組參數與使用方式

--data：資料集檔案路徑（與訓練用相同的 dataset.pkl，其中包含測試集）。(必填)

--model：訓練後模型檔案路徑（如 model.pth）。(必填)

--output：輸出結果CSV檔路徑，用於保存每檔股票的誤差指標。 (選填)

--use_log_target：flag，若預處理/訓練時使用了對數目標，此處需提供以執行預測值的反轉換。

使用範例：

python model_predict_eval.py \
    --data dataset.pkl \
    --model tsmodel.pth \
    --output results.csv \
    --use_log_target


此命令將載入 dataset.pkl 和模型權重 tsmodel.pth，對每檔股票的測試集進行逐日預測，將預測結果與實際值及 GARCH 進行比較，計算 MAE、RMSE，列印結果並寫入 results.csv。CSV 中每行格式如：Ticker, MAE_model, RMSE_model, MAE_garch, RMSE_garch。

完成上述步驟後，我們即可得到模型在測試集上的表現報告。例如，輸出可能顯示模型在大多數股票上的 MAE/RMSE 明顯低於 GARCH，表示模型捕捉到了額外的信息提升了預測性能。同時，由於我們引入了 RevIN 正規化和靜態特徵，模型在不同股票、不同波動度水準下都能較好地泛化。整體而言，透過 Darts TSMixer + RevIN 的架構，以及精心的特徵工程與損失設計，我們在多資產波動度預測任務中取得了比傳統方法更高的準確性。接下來可進一步根據這些結果調整模型參數或架構，以期獲得更優的預測效果。