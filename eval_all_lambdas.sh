#!/bin/bash

# ============================================================================
# 大量對所有 lambda 模型做 prediction + 視覺化（支援平行化）
# ============================================================================
# 目標：
#   1. 對 TSMixer / LSTM 各 21 個 lambda 模型做推論，輸出 metrics.csv
#   2. 對每個 metrics.csv 呼叫 data_visualization.py 產生對應 PNG
#   3. 最後彙整所有結果為一個總表 CSV：outputs/all_metrics_combined.csv
#
# 前置假設：
#   - TSMixer 模型位於：models/tsmixer_lambda*.pth
#   - LSTM   模型位於：models/lstm_lambda*.pth
#   - 評估程式：
#       * TSMixer: python src/model_predict_eval.py
#       * LSTM   : python src/predict_lstm.py
#   - 視覺化程式：
#       * python src/data_visualization.py
#
# 使用方式：
#   單線程執行（較慢，但穩定）
#     ./eval_all_lambdas.sh
#
#   平行執行 4 個任務（推薦，可依照 CPU / GPU 能力調整）
#     ./eval_all_lambdas.sh --parallel 4
#
#   查看說明
#     ./eval_all_lambdas.sh --help
# ============================================================================

set -euo pipefail

# -------------------- 解析參數 --------------------
PARALLEL_JOBS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方式: $0 [選項]"
            echo ""
            echo "選項："
            echo "  --parallel N   平行執行 N 個 prediction 任務（預設: 1）"
            echo "  -h, --help     顯示此說明"
            echo ""
            echo "範例："
            echo "  $0"
            echo "  $0 --parallel 4"
            exit 0
            ;;
        *)
            echo "未知參數: $1"
            echo "使用 $0 --help 查看說明"
            exit 1
            ;;
    esac
done

# -------------------- 基本設定 --------------------
DATA_PATH="Dataset_reenact_yuchi/ts_data.pkl"
TSMIXER_DIR="models/"
LSTM_DIR="models/"
OUTPUT_ROOT="outputs"

# 要跑的 lambda 值（需與你訓練時一致）
LAMBDAS=(0 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 1)

mkdir -p "${OUTPUT_ROOT}"

TOTAL_TASKS=$((${#LAMBDAS[@]} * 2))

echo "=========================================="
echo "開始對所有 lambda 做 prediction + 視覺化"
echo "TSMixer 模型路徑 : ${TSMIXER_DIR}"
echo "LSTM   模型路徑 : ${LSTM_DIR}"
echo "輸出根目錄      : ${OUTPUT_ROOT}"
echo "總任務數        : ${TOTAL_TASKS}"
if [ "${PARALLEL_JOBS}" -gt 1 ]; then
    echo "執行模式        : 平行 (${PARALLEL_JOBS})"
else
    echo "執行模式        : 單線程"
fi
echo "=========================================="
echo ""

# -------------------- 建立任務列表 --------------------
declare -a TASKS
idx=0
for lambda_val in "${LAMBDAS[@]}"; do
    lambda_str=$(echo "${lambda_val}" | tr '.' '_')
    TASKS[idx]="tsmixer:${lambda_val}:${lambda_str}"
    idx=$((idx + 1))
    TASKS[idx]="lstm:${lambda_val}:${lambda_str}"
    idx=$((idx + 1))
done

# -------------------- 單一任務函式 --------------------
run_one_task() {
    local model_type="$1"    # tsmixer 或 lstm
    local lambda_val="$2"
    local lambda_str="$3"
    local task_idx="$4"
    local total_tasks="$5"

    if [[ "${model_type}" == "tsmixer" ]]; then
        local model_path="${TSMIXER_DIR}/tsmixer_lambda${lambda_str}.pth"
        local out_dir="${OUTPUT_ROOT}/tsmixer_lambda${lambda_str}"
        local metrics_csv="${out_dir}/metrics.csv"

        echo "[${task_idx}/${total_tasks}] TSMixer lambda=${lambda_val}"
        echo "  模型:   ${model_path}"
        echo "  輸出:   ${out_dir}"

        if [[ ! -f "${model_path}" ]]; then
            echo "  ✗ 找不到模型檔案：${model_path}，跳過"
            return 1
        fi

        mkdir -p "${out_dir}"

        # 1) 跑 prediction
        python src/model_predict_eval.py \
            --data "${DATA_PATH}" \
            --model "${model_path}" \
            --output "${out_dir}"

        # 2) 視覺化 metrics.csv
        if [[ -f "${metrics_csv}" ]]; then
            python src/data_visualization.py \
                --input "${metrics_csv}" \
                --output "${out_dir}/metrics.png"
        else
            echo "  ⚠ 找不到 ${metrics_csv}，無法繪圖"
        fi

        echo "  ✓ 完成 TSMixer lambda=${lambda_val}"

    elif [[ "${model_type}" == "lstm" ]]; then
        local model_path="${LSTM_DIR}/lstm_lambda${lambda_str}.pth"
        local out_dir="${OUTPUT_ROOT}/lstm_lambda${lambda_str}"
        local metrics_csv="${out_dir}/metrics.csv"

        echo "[${task_idx}/${total_tasks}] LSTM lambda=${lambda_val}"
        echo "  模型:   ${model_path}"
        echo "  輸出:   ${out_dir}"

        if [[ ! -f "${model_path}" ]]; then
            echo "  ✗ 找不到模型檔案：${model_path}，跳過"
            return 1
        fi

        mkdir -p "${out_dir}"

        # 1) 跑 prediction（使用 test split，並畫圖）
        python src/predict_lstm.py \
            --data "${DATA_PATH}" \
            --model "${model_path}" \
            --split test \
            --output "${out_dir}" \
            --save_plots

        # 2) 視覺化 metrics.csv
        if [[ -f "${metrics_csv}" ]]; then
            python src/data_visualization.py \
                --input "${metrics_csv}" \
                --output "${out_dir}/metrics.png"
        else
            echo "  ⚠ 找不到 ${metrics_csv}，無法繪圖"
        fi

        echo "  ✓ 完成 LSTM lambda=${lambda_val}"
    fi
}

# -------------------- 執行所有任務（支援平行） --------------------
START_TIME=$(date +%s)

if [[ ${PARALLEL_JOBS} -le 1 ]]; then
    # 單線程依序執行
    t_idx=0
    for task in "${TASKS[@]}"; do
        t_idx=$((t_idx + 1))
        IFS=':' read -r model_type lambda_val lambda_str <<< "${task}"
        run_one_task "${model_type}" "${lambda_val}" "${lambda_str}" "${t_idx}" "${TOTAL_TASKS}"
        echo ""
    done
else
    # 簡單的背景平行排程
    PIDS=()
    declare -a PIDS
    t_total=${#TASKS[@]}
    next_task=0
    finished=0

    while [[ ${finished} -lt ${t_total} ]]; do
        # 啟動新任務（直到到達並行上限）
        while [[ ${#PIDS[@]} -lt ${PARALLEL_JOBS} && ${next_task} -lt ${t_total} ]]; do
            task="${TASKS[next_task]}"
            IFS=':' read -r model_type lambda_val lambda_str <<< "${task}"
            task_idx=$((next_task + 1))

            run_one_task "${model_type}" "${lambda_val}" "${lambda_str}" "${task_idx}" "${TOTAL_TASKS}" &
            pid=$!
            PIDS+=("${pid}")
            echo "  啟動背景任務 PID=${pid} (${model_type}, lambda=${lambda_val})"

            next_task=$((next_task + 1))
            sleep 1
        done

        # 檢查已有任務是否結束
        new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                new_pids+=("${pid}")
            else
                wait "${pid}" || true
                finished=$((finished + 1))
            fi
        done
        PIDS=("${new_pids[@]}")

        echo "  進度: 已完成 ${finished}/${t_total}，執行中 ${#PIDS[@]}，待啟動 $((t_total - next_task))"
        sleep 2
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
H=$((ELAPSED / 3600))
M=$(((ELAPSED % 3600) / 60))
S=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "所有 prediction + 視覺化 任務完成"
echo "總耗時: ${H} 小時 ${M} 分 ${S} 秒"
echo "開始彙總所有 metrics.csv ..."
echo "=========================================="

# -------------------- 彙總所有 metrics.csv 為一個大 CSV --------------------
python - << 'PY'
import csv
import pathlib

OUTPUT_ROOT = pathlib.Path("outputs")
LAMBDAS = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]

rows = []

def _lambda_str(val: float) -> str:
    s = str(val)
    return s.replace(".", "_")

for lam in LAMBDAS:
    lam_s = _lambda_str(lam)

    # TSMixer
    t_dir = OUTPUT_ROOT / f"tsmixer_lambda{lam_s}"
    t_csv = t_dir / "metrics.csv"
    if t_csv.is_file():
        with t_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    {
                        "model_type": "tsmixer",
                        "lambda": lam,
                        "code": r.get("code"),
                        "MAE_model": float(r.get("MAE_model", "nan")),
                        "RMSE_model": float(r.get("RMSE_model", "nan")),
                        "MAE_garch": float(r.get("MAE_garch", "nan")),
                        "RMSE_garch": float(r.get("RMSE_garch", "nan")),
                    }
                )

    # LSTM
    l_dir = OUTPUT_ROOT / f"lstm_lambda{lam_s}"
    l_csv = l_dir / "metrics.csv"
    if l_csv.is_file():
        with l_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    {
                        "model_type": "lstm",
                        "lambda": lam,
                        "code": r.get("code"),
                        "MAE_model": float(r.get("MAE_LSTM", "nan")),
                        "RMSE_model": float(r.get("RMSE_LSTM", "nan")),
                        "MAE_garch": float(r.get("MAE_GARCH", "nan")),
                        "RMSE_garch": float(r.get("RMSE_GARCH", "nan")),
                    }
                )

out_path = OUTPUT_ROOT / "all_metrics_combined.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model_type",
            "lambda",
            "code",
            "MAE_model",
            "RMSE_model",
            "MAE_garch",
            "RMSE_garch",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"彙總完成，共 {len(rows)} 筆記錄。已輸出至 {out_path}")
PY

echo ""
echo "✅ 全部流程完成："
echo "  - 各 lambda 的 prediction + plots + metrics.csv 已完成"
echo "  - outputs/all_metrics_combined.csv 含全部模型與 lambda 結果"
echo ""


