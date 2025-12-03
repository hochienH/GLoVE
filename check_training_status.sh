#!/bin/bash

# ============================================================================
# 檢查訓練狀態腳本
# ============================================================================
# 此腳本用於檢查訓練任務的執行狀態，包括：
# - 正在運行的 Python 訓練進程
# - 已完成的模型檔案
# - 訓練日誌檔案
# - 失敗的任務
# ============================================================================

echo "=========================================="
echo "訓練狀態檢查"
echo "=========================================="
echo ""

# 定義 lambda 值列表（與訓練腳本一致）
LAMBDAS=(0 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98 0.99 1)

# ============================================================================
# 1. 檢查正在運行的訓練進程
# ============================================================================
echo ">>> 1. 正在運行的訓練進程 <<<"
RUNNING_PROCESSES=$(ps aux | grep -E "(model_train\.py|model_train_lstm\.py)" | grep -v grep)

if [ -z "$RUNNING_PROCESSES" ]; then
    echo "  ⚠ 目前沒有正在運行的訓練進程"
else
    echo "$RUNNING_PROCESSES" | while IFS= read -r line; do
        echo "  $line"
    done
fi
echo ""

# ============================================================================
# 2. 檢查已完成的模型檔案
# ============================================================================
echo ">>> 2. 已完成的模型檔案 <<<"
TSMIXER_COUNT=0
LSTM_COUNT=0

for lambda_val in "${LAMBDAS[@]}"; do
    lambda_str=$(echo "$lambda_val" | tr '.' '_')
    
    # 檢查 TSMixer 模型
    if [ -f "models/tsmixer_lambda${lambda_str}.pth" ]; then
        TSMIXER_COUNT=$((TSMIXER_COUNT + 1))
    fi
    
    # 檢查 LSTM 模型
    if [ -f "models/lstm_lambda${lambda_str}.pth" ]; then
        LSTM_COUNT=$((LSTM_COUNT + 1))
    fi
done

TOTAL_EXPECTED=$((${#LAMBDAS[@]} * 2))
TOTAL_COMPLETED=$((TSMIXER_COUNT + LSTM_COUNT))
COMPLETION_RATE=$(echo "scale=1; $TOTAL_COMPLETED * 100 / $TOTAL_EXPECTED" | bc)

echo "  TSMixer 模型: ${TSMIXER_COUNT}/${#LAMBDAS[@]} 完成"
echo "  LSTM 模型: ${LSTM_COUNT}/${#LAMBDAS[@]} 完成"
echo "  總計: ${TOTAL_COMPLETED}/${TOTAL_EXPECTED} 完成 (${COMPLETION_RATE}%)"
echo ""

# ============================================================================
# 3. 列出所有模型檔案
# ============================================================================
echo ">>> 3. 模型檔案列表 <<<"
if [ -d "models" ] && [ "$(ls -A models/*.pth 2>/dev/null)" ]; then
    ls -lh models/*.pth | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "  ⚠ models/ 資料夾中沒有找到 .pth 檔案"
fi
echo ""

# ============================================================================
# 4. 檢查日誌檔案
# ============================================================================
echo ">>> 4. 最近的訓練日誌 <<<"
LOG_DIR="${LOG_DIR:-logs}"

if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR/*.log 2>/dev/null)" ]; then
    echo "  最近的 5 個日誌檔案："
    ls -t $LOG_DIR/*.log 2>/dev/null | head -5 | while read logfile; do
        filename=$(basename "$logfile")
        size=$(ls -lh "$logfile" | awk '{print $5}')
        mod_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$logfile" 2>/dev/null || stat -c "%y" "$logfile" 2>/dev/null | cut -d' ' -f1-2)
        echo "    $filename ($size, 修改時間: $mod_time)"
    done
else
    echo "  ⚠ $LOG_DIR/ 資料夾中沒有找到日誌檔案"
fi
echo ""

# ============================================================================
# 5. 檢查失敗的任務（有日誌但沒有模型）
# ============================================================================
echo ">>> 5. 可能失敗的任務 <<<"
FAILED_COUNT=0

if [ -d "$LOG_DIR" ]; then
    for logfile in $LOG_DIR/*.log; do
        if [ -f "$logfile" ]; then
            filename=$(basename "$logfile")
            # 從日誌檔名提取模型類型和 lambda
            if [[ $filename =~ ^(tsmixer|lstm)_lambda(.+)\.log$ ]]; then
                model_type="${BASH_REMATCH[1]}"
                lambda_str="${BASH_REMATCH[2]}"
                model_file="models/${model_type}_lambda${lambda_str}.pth"
                
                if [ ! -f "$model_file" ]; then
                    # 檢查日誌最後幾行是否有錯誤
                    last_lines=$(tail -3 "$logfile" 2>/dev/null)
                    if echo "$last_lines" | grep -qi "error\|exception\|failed\|traceback"; then
                        echo "  ✗ ${model_type} lambda=${lambda_str} (可能有錯誤，查看: $logfile)"
                        FAILED_COUNT=$((FAILED_COUNT + 1))
                    fi
                fi
            fi
        fi
    done
fi

if [ $FAILED_COUNT -eq 0 ]; then
    echo "  ✓ 沒有發現明顯失敗的任務"
fi
echo ""

# ============================================================================
# 6. 建議的後續動作
# ============================================================================
echo ">>> 6. 建議的後續動作 <<<"

if [ -z "$RUNNING_PROCESSES" ] && [ $TOTAL_COMPLETED -lt $TOTAL_EXPECTED ]; then
    echo "  ⚠ 訓練進程已停止，但還有未完成的模型"
    echo ""
    echo "  選項 1: 重新啟動訓練（會跳過已完成的模型）"
    echo "    ./train_all_lambdas.sh --parallel 4 --nohup"
    echo ""
    echo "  選項 2: 使用 screen 或 tmux 來管理會話（推薦）"
    echo "    # 安裝 screen: brew install screen"
    echo "    # 啟動 screen: screen -S training"
    echo "    # 在 screen 中執行: ./train_all_lambdas.sh --parallel 4"
    echo "    # 離開 screen: Ctrl+A, D"
    echo "    # 重新連接: screen -r training"
    echo ""
fi

if [ ! -z "$RUNNING_PROCESSES" ]; then
    echo "  ✓ 訓練正在進行中"
    echo "  查看即時日誌："
    echo "    tail -f $LOG_DIR/tsmixer_lambda*.log"
    echo "    tail -f $LOG_DIR/lstm_lambda*.log"
    echo ""
fi

echo "=========================================="
echo "檢查完成"
echo "=========================================="

