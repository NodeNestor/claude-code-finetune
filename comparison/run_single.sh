#!/bin/bash
# Run a single comparison test properly — stops container after profiling completes
set -e

REPO="E:/AgentingStuff/ClaudeCodeFinetune"
COMP="$REPO/comparison"
COMMON_ARGS="--seq-length 16384 --epochs 1 --save-steps 9999 --lora-rank 32 --lora-alpha 64 --profile-tensors"

name="$1"
trainpy="$2"
shift 2
extra_args="$*"

logfile="$COMP/${name}.log"

echo "============================================================"
echo "  TEST: $name"
echo "  extra args: $extra_args"
echo "============================================================"

cd "$REPO"

# Ensure no other containers running
docker stop $(docker ps -q) 2>/dev/null || true
sleep 2

# Run container in background
docker compose run -d \
    --name "test-${name}" \
    -e WANDB_MODE=disabled \
    -v "${trainpy}:/app/train.py:ro" \
    train $COMMON_ARGS $extra_args

# Wait for step 7 profile or grad_norm output (training started), then wait 60s more for a full grad accum
echo "  Waiting for profiling to complete..."
for i in $(seq 1 120); do
    sleep 10
    if docker logs "test-${name}" 2>&1 | grep -q "grad_norm"; then
        echo "  Training started, collecting a few more steps..."
        sleep 60
        break
    fi
    if ! docker ps -q --filter "name=test-${name}" | grep -q .; then
        echo "  Container exited early!"
        break
    fi
done

# Capture logs and stop
docker logs "test-${name}" > "$logfile" 2>&1 || true
docker stop "test-${name}" 2>/dev/null || true
docker rm "test-${name}" 2>/dev/null || true
sleep 3

echo "  -> Log saved to $logfile ($(wc -c < "$logfile") bytes)"
echo "  -> Key metrics:"
grep -E "(STEP 7 MEM|peak_delta.*[5-9]|Per-token peak|Per-token fwd|grad_norm|MLP.*patched|Compress)" "$logfile" | head -8
echo ""
