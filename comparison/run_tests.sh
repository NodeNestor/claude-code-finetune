#!/bin/bash
# Comparison test runner — runs each config sequentially, captures first 16 steps
# Run from: E:\AgentingStuff\ClaudeCodeFinetune\
set -e

REPO="E:/AgentingStuff/ClaudeCodeFinetune"
COMP="$REPO/comparison"
COMMON_ARGS="--seq-length 16384 --epochs 1 --save-steps 9999 --lora-rank 32 --lora-alpha 64 --profile-tensors"

run_test() {
    local name="$1"
    local trainpy="$2"
    local extra_args="$3"
    local logfile="$COMP/${name}.log"

    echo ""
    echo "============================================================"
    echo "  TEST: $name"
    echo "  train.py: $trainpy"
    echo "  extra args: $extra_args"
    echo "============================================================"

    # Run with mounted train.py from the right branch
    cd "$REPO"
    timeout 600 docker compose run --rm \
        --name "test-${name}" \
        -e WANDB_MODE=disabled \
        -v "${trainpy}:/app/train.py:ro" \
        train $COMMON_ARGS $extra_args \
        > "$logfile" 2>&1 || true

    # Extract key metrics
    echo "  -> Log saved to $logfile"
    echo "  -> Key metrics:"
    grep -E "(STEP [0-7] MEM|peak_delta|Per-token|grad_norm|loss.*=|SAVED TENSOR|MLP|LoRA|Total:)" "$logfile" | head -20
    echo ""
}

echo "Starting comparison tests at $(date)"
echo "Results will be in: $COMP/"

# 1. Baseline (master)
run_test "1-baseline" \
    "$REPO/train.py" \
    ""

# 2. fp8 activations only
run_test "2-fp8" \
    "$REPO/../ClaudeCodeFinetune-fp8-activations/train.py" \
    "--activation-dtype fp8"

# 3. Random projection only (k=256)
run_test "3-proj256" \
    "$REPO/../ClaudeCodeFinetune-fp8-activations/train.py" \
    "--proj-dim 256"

# 4. fp8 + projection combined (96x compression)
run_test "4-fp8-proj256" \
    "$REPO/../ClaudeCodeFinetune-fp8-activations/train.py" \
    "--activation-dtype fp8 --proj-dim 256"

# 5. Reduced LoRA targets (q_proj,v_proj only)
run_test "5-lora-qv" \
    "$REPO/../ClaudeCodeFinetune-lora-targets/train.py" \
    "--lora-targets q_proj,v_proj"

echo ""
echo "============================================================"
echo "  ALL TESTS COMPLETE"
echo "============================================================"
echo ""

# Summary comparison
echo "COMPARISON SUMMARY:"
echo "==================="
for log in "$COMP"/*.log; do
    name=$(basename "$log" .log)
    echo ""
    echo "--- $name ---"
    # Get the longest sequence step for fair comparison
    grep -E "STEP [1-7] MEM" "$log" | tail -1
    grep "peak_delta" "$log" | tail -1
    grep "Per-token peak" "$log" | tail -1
    grep "grad_norm" "$log" | tail -1
done
