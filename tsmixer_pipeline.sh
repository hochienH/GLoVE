#!/bin/bash
set -euo pipefail

###############################################################################
# CONFIG
###############################################################################

DATA_PATH="Dataset/ts_data.pkl"
LAMBDAS=(0 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1)
OPTUNA_TRIALS=100
OPTUNA_LAMBDA=0.5
EPOCHS=10
BATCH_SIZE=32
LR_SCHEDULER="exponential"
LR_GAMMA=0.99
GRAD_CLIP=0.5

export EPOCHS
export BATCH_SIZE
export DATA_PATH
export LR_SCHEDULER
export LR_GAMMA
export GRAD_CLIP

mkdir -p models
mkdir -p logs
mkdir -p outputs

echo "=========================================="
echo "TSMIXER FULL PIPELINE"
echo "DATA_PATH = $DATA_PATH"
echo "=========================================="

###############################################################################
# 0.  ARGUMENT PARSING
###############################################################################

# default
PARALLEL_JOBS=1
USE_LOG_TARGET=0
COVARIATE_MODE="none"
INVERT_TRAIN_SCALE="none"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --use_log_target)
            USE_LOG_TARGET=1
            shift 1
            ;;
        --covariate_mode)
            COVARIATE_MODE="$2"
            shift 2
            ;;
        --invert_train_scale)
            INVERT_TRAIN_SCALE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export PARALLEL_JOBS
export USE_LOG_TARGET
export COVARIATE_MODE
export INVERT_TRAIN_SCALE

###############################################################################
# 0.5 CREATE TIMESTAMP RUN ID + DIRECTORY STRUCTURE
###############################################################################

RUN_ID="$(date +%Y%m%d_%H%M%S)"

# 各輸出資料夾（模型、輸出、log、config 各自分開）
MODEL_DIR="models/$RUN_ID"
OUTPUT_DIR="outputs/$RUN_ID"
LOG_DIR="logs/$RUN_ID"

mkdir -p "$MODEL_DIR/base"
mkdir -p "$OUTPUT_DIR/base"
mkdir -p "$LOG_DIR"

echo "RUN_ID = $RUN_ID"
echo "Models  → $MODEL_DIR"
echo "Outputs → $OUTPUT_DIR"
echo "Logs    → $LOG_DIR"

# 建 config.json (先放基本 flags，後面會補最佳超參數)
LAMBDAS_JSON=$(printf "%s," "${LAMBDAS[@]}" | sed 's/,$//')

cat > "$MODEL_DIR/config.json" <<EOF
{
  "run_id": "$RUN_ID",
  "model": "TSMixer",
  "covariate_mode": "$COVARIATE_MODE",
  "invert_train_scale": "$INVERT_TRAIN_SCALE",
  "use_log_target": $USE_LOG_TARGET,
  "lambdas": [$LAMBDAS_JSON]
}
EOF

###############################################################################
# 1.  OPTUNA BASELINE
###############################################################################

echo ""
echo "=========================================="
echo "STEP 1: TSMixer baseline Optuna 搜尋"
echo "=========================================="

python src/TSMixer/model_train_tsmixer_optuna_Base.py \
  --data $DATA_PATH \
  --trials $OPTUNA_TRIALS \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lambda_weight $OPTUNA_LAMBDA \
  --lr_min 1e-4 \
  --lr_max 1e-2 \
  --input_chunk_length 90 \
  --log_dir $LOG_DIR \
  --output_model $MODEL_DIR/base/tsmixer.pth \
  --trial_csv $OUTPUT_DIR/optuna_lr_trials.csv \
  --seed 42 \
  --search_num_blocks 1 2 4 \
  --search_dropout 0.1 0.2 0.3 \
  --search_hidden_size 16 32 64 \
  --search_ff_size 16 32 64 \
  --pruner_startup_trials 10 \
  --pruner_warmup_steps 0 \
  --patience 2 \
  --optuna_timeout_sec 3600 \
  --covariate_mode "$COVARIATE_MODE"

###############################################################################
# 1.5.  READ THE OPTIMAL HYPERPARAMETERS
###############################################################################

echo ""
echo "=========================================="
echo "STEP 1.5: 從 Optuna trial CSV 讀取最佳超參數"
echo "=========================================="

eval "$(python - << PY
import pandas as pd

csv_path = "$OUTPUT_DIR/optuna_lr_trials.csv"
df = pd.read_csv(csv_path)
# 只保留完整結束的 trial
df = df[df["state"] == "COMPLETE"]

# 取 value 最小的那一列
best = df.loc[df["value"].idxmin()]

def emit(name, val):
    print(f'{name}={val}')

emit("BEST_LR", best["lr"])
emit("BEST_NUM_BLOCKS", int(best["num_blocks"]))
emit("BEST_DROPOUT", best["dropout"])
emit("BEST_HIDDEN_SIZE", int(best["hidden_size"]))
emit("BEST_FF_SIZE", int(best["ff_size"]))
PY
)"

echo "最佳參數："
echo "  lr          = $BEST_LR"
echo "  num_blocks  = $BEST_NUM_BLOCKS"
echo "  dropout     = $BEST_DROPOUT"
echo "  hidden_size = $BEST_HIDDEN_SIZE"
echo "  ff_size     = $BEST_FF_SIZE"

# Append hyperparameters into config.json
python - <<PY
import json

cfg_path = "$MODEL_DIR/config.json"
with open(cfg_path) as f:
    cfg = json.load(f)

cfg["best_lr"] = "$BEST_LR"
cfg["best_num_blocks"] = "$BEST_NUM_BLOCKS"
cfg["best_dropout"] = "$BEST_DROPOUT"
cfg["best_hidden_size"] = "$BEST_HIDDEN_SIZE"
cfg["best_ff_size"] = "$BEST_FF_SIZE"

with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
PY

cp "$MODEL_DIR/config.json" "$OUTPUT_DIR/config.json"

###############################################################################
# 2. LAMBDA SWEEP：TSMixer 訓練（使用 Optuna 找到的最佳超參數）
###############################################################################

echo ""
echo "=========================================="
echo "STEP 2: 掃描所有 λ，使用最佳超參數訓練 TSMixer"
echo "=========================================="

run_train_tsmixer() {
    local lam=$1
    local lam_str=$(echo "$lam" | tr '.' '_')

    MODEL_LAM_DIR="$MODEL_DIR/lambda_${lam_str}"
    mkdir -p "$MODEL_LAM_DIR"

    LOG_LAM_FILE="$LOG_DIR/lambda_${lam_str}.log"

    out_path="$MODEL_LAM_DIR/model.pth"
    log_file="$LOG_LAM_FILE"

    echo "[Train] λ = $lam → $out_path"
    echo "       (lr=$BEST_LR, blocks=$BEST_NUM_BLOCKS, hidden=$BEST_HIDDEN_SIZE, ff=$BEST_FF_SIZE, dropout=$BEST_DROPOUT)"

    python src/TSMixer/model_train.py \
        --data "$DATA_PATH" \
        --lambda "$lam" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr "$BEST_LR" \
        --lr_scheduler $LR_SCHEDULER \
        --lr_gamma $LR_GAMMA \
        --grad_clip $GRAD_CLIP \
        --hidden_size "$BEST_HIDDEN_SIZE" \
        --ff_size "$BEST_FF_SIZE" \
        --num_blocks "$BEST_NUM_BLOCKS" \
        --dropout "$BEST_DROPOUT" \
        --model_path "$out_path" \
        --covariate_mode "$COVARIATE_MODE" \
        > "$log_file" 2>&1
}

export -f run_train_tsmixer
export DATA_PATH BEST_LR BEST_NUM_BLOCKS BEST_DROPOUT BEST_HIDDEN_SIZE BEST_FF_SIZE

for lam in "${LAMBDAS[@]}"; do
    run_train_tsmixer "$lam" &
    [[ $(jobs -r | wc -l) -ge "$PARALLEL_JOBS" ]] && wait
done
wait

echo "所有 λ 的 TSMixer 都訓練完成。"

###############################################################################
# 3. PREDICT + VISUALIZATION
###############################################################################

echo ""
echo "=========================================="
echo "STEP 3: TSMixer λ sweep — Prediction + Visualization"
echo "=========================================="

run_predict_vis() {
    local lam=$1
    local lam_str=$(echo "$lam" | tr '.' '_')

    MODEL_LAM_DIR="$MODEL_DIR/lambda_${lam_str}"
    OUTPUT_LAM_DIR="$OUTPUT_DIR/lambda_${lam_str}"
    mkdir -p "$OUTPUT_LAM_DIR"

    echo "[Predict] λ = $lam → $OUTPUT_LAM_DIR"

    CMD_PREDICT=(
        python src/TSMixer/model_predict_eval.py
        --data "$DATA_PATH"
        --model "$MODEL_LAM_DIR/model.pth"
        --output "$OUTPUT_LAM_DIR"
        --invert_train_scale "$INVERT_TRAIN_SCALE"
        --covariate_mode "$COVARIATE_MODE"
    )

    # 若使用者有輸入 --use_log_target 就加入
    if [[ "$USE_LOG_TARGET" -eq 1 ]]; then
        CMD_PREDICT+=(--use_log_target)
    fi

    "${CMD_PREDICT[@]}"

    python src/data_visualization.py \
        --input "$OUTPUT_LAM_DIR/metrics.csv" \
        --output "$OUTPUT_LAM_DIR/metrics.png"
}

export -f run_predict_vis

for lam in "${LAMBDAS[@]}"; do
    run_predict_vis "$lam" &
    [[ $(jobs -r | wc -l) -ge "$PARALLEL_JOBS" ]] && wait
done

wait
echo "所有 λ 的 Prediction + Visualization 完成。"

###############################################################################
# 4. MERGE ALL METRICS
###############################################################################

echo ""
echo "=========================================="
echo "STEP 4: 合併所有 λ 的 metrics.csv → all_metrics_combined.csv"
echo "=========================================="

python src/merge_lambda.py \
  --base_dir "$OUTPUT_DIR" \
  --output_csv $OUTPUT_DIR/all_metrics_combined.csv \
  --output_fig $OUTPUT_DIR/all_lambda_metrics.png

echo ""
echo "=========================================="
echo "    TSMixer Pipeline 完整執行完成！"
echo "=========================================="
echo "模型位置：$MODEL_DIR"
echo "輸出位置：$OUTPUT_DIR"
