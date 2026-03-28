#!/bin/bash
# run_ablations.sh
# ────────────────
# Priority ablation suite for 16h GPU budget at 8k examples/run.
#
# Runs (5 training + eval, ~3.2h each = ~16h total):
#   0. MoLoRA K=4, r=16, temp=10 — primary result re-run at 8k examples
#   1. Vanilla LoRA r=64  — capacity-matched baseline (K=4 × r=16 = 64 total rank)
#   2. Vanilla LoRA r=16  — parameter-matched baseline (same as one MoLoRA expert)
#   3. MoLoRA K=2, r=16   — fewer experts; tests whether expert count matters
#   4. MoLoRA K=4, r=8    — lower per-expert rank; tests whether capacity matters
#
# Free data points (no GPU needed):
#   • Zero-shot baseline  — inference only on 200 samples (~15min)
#   • MoLoRA K=4, temp=1 (collapsed) — already in logs
#
# All evaluations are capped at 200 test examples for consistency and speed.
#
# Usage:
#   bash run_ablations.sh
#   nohup bash run_ablations.sh > ablations.log 2>&1 &

set -e

# ── Shared paths ───────────────────────────────────────────────────────────
PROJECT="spiqa-molora-ablations"

TRAIN_JSON="./spiqa_plus/SPIQA_plus_train_1500.json"
IMG_DIR="./spiqa/train_val/SPIQA_train_val_Images"
VAL_JSON="./spiqa_plus/SPIQA_plus_val_200.json"
VAL_IMG_DIR="./spiqa/train_val/SPIQA_train_val_Images"
TEST_JSON="./spiqa_plus/SPIQA_plus_testA_118.json"
TEST_IMG_DIR="./spiqa/test-A/SPIQA_testA_Images"

SAMPLES=8000       # training examples
EVAL_SAMPLES=200   # test examples — consistent cap across all eval runs
EPOCHS=1

# ── Anti-collapse hyperparams (fixed for all MoLoRA runs) ─────────────────
TEMP_INIT=10.0
TEMP_FRAC=0.3
ENTROPY_COEF=0.02
NOISE=0.5

# ── Helper: evaluate a MoLoRA checkpoint ──────────────────────────────────
run_eval_molora() {
    local molora_path=$1
    local out_dir=$2
    local run_name=$3
    local K=${4:-4}
    echo "    >>> Evaluating $run_name (${EVAL_SAMPLES} samples) ..."
    python final_ablations/evaluate_molora.py \
        --model_id    Qwen/Qwen3-VL-8B-Instruct \
        --molora_path "$molora_path" \
        --test_json   $TEST_JSON \
        --image_dir   $TEST_IMG_DIR \
        --output_dir  "$out_dir" \
        --ablation    "$run_name" \
        --num_experts $K \
        --max_samples $EVAL_SAMPLES
    echo "    >>> Done. Results in $out_dir/${run_name}.json"
}

# ── Helper: evaluate a LoRA checkpoint ────────────────────────────────────
run_eval_lora() {
    local lora_path=$1
    local out_dir=$2
    local run_name=$3
    echo "    >>> Evaluating $run_name (${EVAL_SAMPLES} samples) ..."
    python final_ablations/evaluate.py \
        --model_id   Qwen/Qwen3-VL-8B-Instruct \
        --lora_path  "$lora_path" \
        --test_json  $TEST_JSON \
        --image_dir  $TEST_IMG_DIR \
        --output_dir "$out_dir" \
        --ablation   "$run_name" \
        --max_samples $EVAL_SAMPLES
    echo "    >>> Done. Results in $out_dir/${run_name}.json"
}

# ══════════════════════════════════════════════════════════════════════════
# RUN 0: MoLoRA K=4, r=16, temp=10 — primary result at 8k examples
# Re-run of the completed 10k run, capped at 8k for consistency with ablations.
# This is the anchor point all other runs are compared against.
# ══════════════════════════════════════════════════════════════════════════

# echo "=========================================="
# echo ">>> Run 0/4: MoLoRA K=4 r=16 (primary result, 8k examples)"
# echo "=========================================="
# python train_molora.py \
#     --num_experts             4 \
#     --molora_r                16 \
#     --molora_alpha            32 \
#     --aux_loss_coef           0.1 \
#     --router_entropy_coef     $ENTROPY_COEF \
#     --router_noise            $NOISE \
#     --router_temp_init        $TEMP_INIT \
#     --router_temp_anneal_frac $TEMP_FRAC \
#     --spiqa_train_json        $TRAIN_JSON \
#     --spiqa_image_dir         $IMG_DIR \
#     --spiqa_val_json          $VAL_JSON \
#     --spiqa_val_image_dir     $VAL_IMG_DIR \
#     --max_samples             $SAMPLES \
#     --num_train_epochs        $EPOCHS \
#     --no_use_qlora \
#     --output_dir              ./outputs/ablation-molora-K4-r16 \
#     --wandb_project           $PROJECT \
#     --wandb_run_name          "molora-K4-r16"
# 
# run_eval_molora \
#     "./outputs/ablation-molora-K4-r16/final_molora" \
#     "./eval_results" \
#     "molora-K4-r16" 4

# ══════════════════════════════════════════════════════════════════════════
# ZERO-SHOT BASELINE (no training — inference only, ~15min)
# ══════════════════════════════════════════════════════════════════════════

# echo "=========================================="
# echo ">>> Zero-shot baseline (no fine-tuning, ${EVAL_SAMPLES} samples)"
# echo "=========================================="
# python evaluate_baseline.py \
#     --test_json   $TEST_JSON \
#     --image_dir   $TEST_IMG_DIR \
#     --output_dir  ./eval_results \
#     --ablation    baseline-zeroshot \
#     --max_samples $EVAL_SAMPLES
# echo ">>> Zero-shot done."

# ══════════════════════════════════════════════════════════════════════════
# RUN 1: Vanilla LoRA r=64
# Capacity-matched baseline: K=4 experts × r=16 = 64 total rank
# ══════════════════════════════════════════════════════════════════════════

# echo "=========================================="
# echo ">>> Run 1/4: Vanilla LoRA r=64 (capacity-matched baseline)"
# echo "=========================================="
# python train.py \
#     --lora_r              64 \
#     --lora_alpha          128 \
#     --spiqa_train_json    $TRAIN_JSON \
#     --spiqa_image_dir     $IMG_DIR \
#     --spiqa_val_json      $VAL_JSON \
#     --spiqa_val_image_dir $VAL_IMG_DIR \
#     --max_samples         $SAMPLES \
#     --num_train_epochs    $EPOCHS \
#     --no_use_qlora \
#     --output_dir          ./outputs/ablation-lora-r64 \
#     --wandb_project       $PROJECT \
#     --wandb_run_name      "baseline-lora-r64"
# 
# run_eval_lora \
#     "./outputs/ablation-lora-r64/final_lora_adapter" \
#     "./eval_results" \
#     "baseline-lora-r64"

# ══════════════════════════════════════════════════════════════════════════
# RUN 2: Vanilla LoRA r=16
# Parameter-matched baseline: same params as one MoLoRA expert
# ══════════════════════════════════════════════════════════════════════════

# echo "=========================================="
# echo ">>> Run 2/4: Vanilla LoRA r=16 (parameter-matched baseline)"
# echo "=========================================="
# python train.py \
#     --lora_r              16 \
#     --lora_alpha          32 \
#     --spiqa_train_json    $TRAIN_JSON \
#     --spiqa_image_dir     $IMG_DIR \
#     --spiqa_val_json      $VAL_JSON \
#     --spiqa_val_image_dir $VAL_IMG_DIR \
#     --max_samples         $SAMPLES \
#     --num_train_epochs    $EPOCHS \
#     --no_use_qlora \
#     --output_dir          ./outputs/ablation-lora-r16 \
#     --wandb_project       $PROJECT \
#     --wandb_run_name      "baseline-lora-r16"
# 
# run_eval_lora \
#     "./outputs/ablation-lora-r16/final_lora_adapter" \
#     "./eval_results" \
#     "baseline-lora-r16"

# ══════════════════════════════════════════════════════════════════════════
# RUN 3: MoLoRA K=2, r=16
# Fewer experts — tests whether expert count matters.
# ══════════════════════════════════════════════════════════════════════════

# echo "=========================================="
# echo ">>> Run 3/4: MoLoRA K=2, r=16 (expert count ablation)"
# echo "=========================================="
# python train_molora.py \
#     --num_experts             2 \
#     --molora_r                16 \
#     --molora_alpha            32 \
#     --aux_loss_coef           0.1 \
#     --router_entropy_coef     $ENTROPY_COEF \
#     --router_noise            $NOISE \
#     --router_temp_init        $TEMP_INIT \
#     --router_temp_anneal_frac $TEMP_FRAC \
#     --spiqa_train_json        $TRAIN_JSON \
#     --spiqa_image_dir         $IMG_DIR \
#     --spiqa_val_json          $VAL_JSON \
#     --spiqa_val_image_dir     $VAL_IMG_DIR \
#     --max_samples             $SAMPLES \
#     --num_train_epochs        $EPOCHS \
#     --no_use_qlora \
#     --output_dir              ./outputs/ablation-molora-K2 \
#     --wandb_project           $PROJECT \
#     --wandb_run_name          "molora-K2-r16"
# 
# run_eval_molora \
#     "./outputs/ablation-molora-K2/final_molora" \
#     "./eval_results" \
#     "molora-K2-r16" 2

# ══════════════════════════════════════════════════════════════════════════
# RUN 4: MoLoRA K=4, r=8
# Lower per-expert rank — tests whether expert capacity matters.
# ══════════════════════════════════════════════════════════════════════════

# echo "=========================================="
# echo ">>> Run 4/4: MoLoRA K=4, r=8 (expert capacity ablation)"
# echo "=========================================="
# python train_molora.py \
#     --num_experts             4 \
#     --molora_r                8 \
#     --molora_alpha            16 \
#     --aux_loss_coef           0.1 \
#     --router_entropy_coef     $ENTROPY_COEF \
#     --router_noise            $NOISE \
#     --router_temp_init        $TEMP_INIT \
#     --router_temp_anneal_frac $TEMP_FRAC \
#     --spiqa_train_json        $TRAIN_JSON \
#     --spiqa_image_dir         $IMG_DIR \
#     --spiqa_val_json          $VAL_JSON \
#     --spiqa_val_image_dir     $VAL_IMG_DIR \
#     --max_samples             $SAMPLES \
#     --num_train_epochs        $EPOCHS \
#     --no_use_qlora \
#     --output_dir              ./outputs/ablation-molora-K4-r8 \
#     --wandb_project           $PROJECT \
#     --wandb_run_name          "molora-K4-r8"
# 
# run_eval_molora \
#     "./outputs/ablation-molora-K4-r8/final_molora" \
#     "./eval_results" \
#     "molora-K4-r8" 4
# 
# echo "=========================================="
# echo ">>> All ablation runs complete."
# echo ">>> Run 'python plot_ablations.py' to generate comparison figures."
# echo "=========================================="

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: Merger unfreezing ablation
# Run after Stage 1 results are in. Uses the best configuration from each
# architecture family (LoRA r=64, MoLoRA K=4 r=16) with merger unfrozen.
#
# Hypothesis: unfreezing the merger allows it to co-adapt with the adapters,
# producing sharper visual summaries. For MoLoRA this may improve routing
# discriminability. For LoRA it tests whether merger adaptation alone helps.
# ══════════════════════════════════════════════════════════════════════════

echo "=========================================="
echo ">>> Stage 2a: LoRA r=64 + unfrozen merger"
echo "=========================================="
python final_ablations/train.py \
    --lora_r              64 \
    --lora_alpha          128 \
    --spiqa_train_json    $TRAIN_JSON \
    --spiqa_image_dir     $IMG_DIR \
    --spiqa_val_json      $VAL_JSON \
    --spiqa_val_image_dir $VAL_IMG_DIR \
    --max_samples         $SAMPLES \
    --num_train_epochs    $EPOCHS \
    --no_use_qlora \
    --no_freeze_vision \
    --output_dir          ./outputs/ablation-lora-r64-merger \
    --wandb_project       $PROJECT \
    --wandb_run_name      "lora-r64-merger-tuned"

run_eval_lora \
    "./outputs/ablation-lora-r64-merger/final_lora_adapter" \
    "./eval_results" \
    "lora-r64-merger"

echo "=========================================="
echo ">>> Stage 2b: MoLoRA K=4 r=16 + unfrozen merger"
echo "=========================================="
python final_ablations/train_molora.py \
    --num_experts             4 \
    --molora_r                16 \
    --molora_alpha            32 \
    --aux_loss_coef           0.1 \
    --router_entropy_coef     $ENTROPY_COEF \
    --router_noise            $NOISE \
    --router_temp_init        $TEMP_INIT \
    --router_temp_anneal_frac $TEMP_FRAC \
    --spiqa_train_json        $TRAIN_JSON \
    --spiqa_image_dir         $IMG_DIR \
    --spiqa_val_json          $VAL_JSON \
    --spiqa_val_image_dir     $VAL_IMG_DIR \
    --max_samples             $SAMPLES \
    --num_train_epochs        $EPOCHS \
    --no_use_qlora \
    --no_freeze_vision \
    --output_dir              ./outputs/ablation-molora-K4-r16-merger \
    --wandb_project           $PROJECT \
    --wandb_run_name          "molora-K4-r16-merger-tuned"

run_eval_molora \
    "./outputs/ablation-molora-K4-r16-merger/final_molora" \
    "./eval_results" \
    "molora-K4-r16-merger" 4

echo "=========================================="
echo ">>> Stage 2 complete."
echo ">>> Run 'python plot_ablations.py' to generate comparison figures."
echo "=========================================="
