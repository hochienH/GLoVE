#!/bin/bash

# ============================================================================
# 大量測試所有 lambda 值的訓練腳本（支援平行化與背景執行）
# ============================================================================
# 此腳本會自動訓練 TSMixer 和 LSTM 模型，使用不同的 lambda 值
# 所有模型會儲存在 models/ 資料夾中，檔名包含對應的 lambda 值
# 
# 使用方式：
#   ./train_all_lambdas.sh                    # 順序執行
#   ./train_all_lambdas.sh --parallel 4       # 平行執行 4 個任務
#   ./train_all_lambdas.sh --nohup            # 背景執行（使用 nohup）
#   ./train_all_lambdas.sh --parallel 4 --nohup  # 平行 + 背景執行
# ============================================================================
#共花費5小時多
# 解析命令列參數
PARALLEL_JOBS=1
USE_NOHUP=false
LOG_DIR="logs"

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --nohup)
            USE_NOHUP=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方式: $0 [選項]"
            echo ""
            echo "選項："
            echo "  --parallel N     平行執行 N 個任務（預設: 1，順序執行）"
            echo "  --nohup          使用 nohup 在背景執行"
            echo "  --log-dir DIR    日誌檔案目錄（預設: logs/）"
            echo "  -h, --help       顯示此說明"
            echo ""
            echo "範例："
            echo "  $0                          # 順序執行"
            echo "  $0 --parallel 4              # 平行執行 4 個任務"
            echo "  $0 --nohup                   # 背景執行"
            echo "  $0 --parallel 4 --nohup      # 平行 + 背景執行"
            exit 0
            ;;
        *)
            echo "未知參數: $1"
            echo "使用 $0 --help 查看說明"
            exit 1
            ;;
    esac
done

# 設定資料路徑
DATA_PATH="Dataset_reenact_yuchi/ts_data.pkl"

# 定義要測試的 lambda 值列表
LAMBDAS=(0 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 1)

# 確保必要的資料夾存在
mkdir -p models
mkdir -p "${LOG_DIR}"

# 計算總共要訓練的模型數量
TOTAL_MODELS=$((${#LAMBDAS[@]} * 2))

# 建立任務列表（每個 lambda 值對應兩個任務：TSMixer 和 LSTM）
declare -a TASKS

task_id=0
for lambda_val in "${LAMBDAS[@]}"; do
    lambda_str=$(echo "$lambda_val" | tr '.' '_')
    
    # TSMixer 任務
    TASKS[$task_id]="tsmixer:${lambda_val}:${lambda_str}"
    task_id=$((task_id + 1))
    
    # LSTM 任務
    TASKS[$task_id]="lstm:${lambda_val}:${lambda_str}"
    task_id=$((task_id + 1))
done

# ============================================================================
# 訓練函數
# ============================================================================
train_model() {
    local model_type=$1
    local lambda_val=$2
    local lambda_str=$3
    local task_num=$4
    local total_tasks=$5
    
    local model_path=""
    local log_file="${LOG_DIR}/${model_type}_lambda${lambda_str}.log"
    
    if [ "$model_type" == "tsmixer" ]; then
        model_path="models/tsmixer_lambda${lambda_str}.pth"
        echo "[任務 ${task_num}/${total_tasks}] 開始訓練 TSMixer (lambda=${lambda_val})"
        echo "  模型: ${model_path}"
        echo "  日誌: ${log_file}"
        
        if [ "$USE_NOHUP" == true ]; then
            nohup python src/model_train.py \
                --data "${DATA_PATH}" \
                --lambda "${lambda_val}" \
                --epochs 6 \
                --lr 3e-4 \
                --lr_scheduler exponential \
                --lr_gamma 0.99 \
                --grad_clip 0.5 \
                --hidden_size 32 \
                --ff_size 64 \
                --num_blocks 4 \
                --dropout 0.1 \
                --model_path "${model_path}" \
                > "${log_file}" 2>&1
        else
            python src/model_train.py \
                --data "${DATA_PATH}" \
                --lambda "${lambda_val}" \
                --epochs 6 \
                --lr 3e-4 \
                --lr_scheduler exponential \
                --lr_gamma 0.99 \
                --grad_clip 0.5 \
                --hidden_size 32 \
                --ff_size 64 \
                --num_blocks 4 \
                --dropout 0.1 \
                --model_path "${model_path}" \
                > "${log_file}" 2>&1
        fi
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[任務 ${task_num}/${total_tasks}] ✓ TSMixer (lambda=${lambda_val}) 訓練完成"
            return 0
        else
            echo "[任務 ${task_num}/${total_tasks}] ✗ TSMixer (lambda=${lambda_val}) 訓練失敗 (退出碼: ${exit_code})"
            return 1
        fi
        
    elif [ "$model_type" == "lstm" ]; then
        model_path="models/lstm_lambda${lambda_str}.pth"
        echo "[任務 ${task_num}/${total_tasks}] 開始訓練 LSTM (lambda=${lambda_val})"
        echo "  模型: ${model_path}"
        echo "  日誌: ${log_file}"
        
        if [ "$USE_NOHUP" == true ]; then
            nohup python src/model_train_lstm.py \
                --data "${DATA_PATH}" \
                --lambda "${lambda_val}" \
                --epochs 6 \
                --lr 2e-4 \
                --lr_scheduler exponential \
                --lr_gamma 0.99 \
                --grad_clip 0.5 \
                --hidden_size 32 \
                --dropout 0.1 \
                --model_path "${model_path}" \
                > "${log_file}" 2>&1
        else
            python src/model_train_lstm.py \
                --data "${DATA_PATH}" \
                --lambda "${lambda_val}" \
                --epochs 6 \
                --lr 2e-4 \
                --lr_scheduler exponential \
                --lr_gamma 0.99 \
                --grad_clip 0.5 \
                --hidden_size 32 \
                --dropout 0.1 \
                --model_path "${model_path}" \
                > "${log_file}" 2>&1
        fi
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[任務 ${task_num}/${total_tasks}] ✓ LSTM (lambda=${lambda_val}) 訓練完成"
            return 0
        else
            echo "[任務 ${task_num}/${total_tasks}] ✗ LSTM (lambda=${lambda_val}) 訓練失敗 (退出碼: ${exit_code})"
            return 1
        fi
    fi
}

# ============================================================================
# 主執行邏輯
# ============================================================================
echo "=========================================="
echo "開始大量訓練：共 ${TOTAL_MODELS} 個模型"
echo "TSMixer: ${#LAMBDAS[@]} 個 lambda 值"
echo "LSTM: ${#LAMBDAS[@]} 個 lambda 值"
echo "執行模式: $([ $PARALLEL_JOBS -gt 1 ] && echo "平行執行 (${PARALLEL_JOBS} 個並發)" || echo "順序執行")"
echo "$([ "$USE_NOHUP" == true ] && echo "背景執行: 是 (使用 nohup)" || echo "背景執行: 否")"
echo "日誌目錄: ${LOG_DIR}/"
echo "=========================================="
echo ""

# 記錄開始時間
START_TIME=$(date +%s)

# 如果使用平行執行
if [ $PARALLEL_JOBS -gt 1 ]; then
    # 平行執行模式
    declare -a PIDS
    current_task=0
    completed_tasks=0
    failed_tasks=0
    
    while [ $current_task -lt ${#TASKS[@]} ] || [ ${#PIDS[@]} -gt 0 ]; do
        # 啟動新任務（如果還有任務且未達並發上限）
        while [ ${#PIDS[@]} -lt $PARALLEL_JOBS ] && [ $current_task -lt ${#TASKS[@]} ]; do
            task_info="${TASKS[$current_task]}"
            IFS=':' read -r model_type lambda_val lambda_str <<< "$task_info"
            task_num=$((current_task + 1))
            
            # 在背景執行訓練
            train_model "$model_type" "$lambda_val" "$lambda_str" "$task_num" "$TOTAL_MODELS" &
            pid=$!
            PIDS+=($pid)
            echo "  啟動背景任務 PID: $pid (${model_type}, lambda=${lambda_val})"
            
            current_task=$((current_task + 1))
            sleep 1  # 稍微延遲避免同時啟動太多任務
        done
        
        # 檢查完成的任務
        for i in "${!PIDS[@]}"; do
            pid="${PIDS[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                # 任務已完成，等待並取得退出碼
                wait "$pid"
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    completed_tasks=$((completed_tasks + 1))
                else
                    failed_tasks=$((failed_tasks + 1))
                fi
                unset PIDS[$i]
            fi
        done
        
        # 重新索引陣列
        PIDS=("${PIDS[@]}")
        
        # 顯示進度
        if [ $((completed_tasks + failed_tasks)) -gt 0 ]; then
            echo "[進度] 已完成: ${completed_tasks}, 失敗: ${failed_tasks}, 執行中: ${#PIDS[@]}, 待執行: $((${#TASKS[@]} - current_task))"
        fi
        
        sleep 2  # 避免 CPU 過度使用
    done
    
    echo ""
    echo "所有任務執行完成！"
    echo "  成功: ${completed_tasks}"
    echo "  失敗: ${failed_tasks}"
    
else
    # 順序執行模式
    current_task=0
    completed_tasks=0
    failed_tasks=0
    
    for task_info in "${TASKS[@]}"; do
        IFS=':' read -r model_type lambda_val lambda_str <<< "$task_info"
        current_task=$((current_task + 1))
        
        train_model "$model_type" "$lambda_val" "$lambda_str" "$current_task" "$TOTAL_MODELS"
        
        if [ $? -eq 0 ]; then
            completed_tasks=$((completed_tasks + 1))
        else
            failed_tasks=$((failed_tasks + 1))
        fi
        
        echo ""
    done
    
    echo "所有任務執行完成！"
    echo "  成功: ${completed_tasks}"
    echo "  失敗: ${failed_tasks}"
fi

# 計算總執行時間
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# ============================================================================
# 總結
# ============================================================================
echo ""
echo "=========================================="
echo "訓練腳本執行完成！"
echo "=========================================="
echo ""
echo "執行時間: ${HOURS} 小時 ${MINUTES} 分鐘 ${SECONDS} 秒"
echo "模型儲存位置: models/"
echo "  TSMixer 模型: models/tsmixer_lambda*.pth"
echo "  LSTM 模型: models/lstm_lambda*.pth"
echo "日誌檔案位置: ${LOG_DIR}/"
echo "  TSMixer 日誌: ${LOG_DIR}/tsmixer_lambda*.log"
echo "  LSTM 日誌: ${LOG_DIR}/lstm_lambda*.log"
echo ""
echo "可以使用以下指令查看："
echo "  ls -lh models/"
echo "  ls -lh ${LOG_DIR}/"
echo ""
