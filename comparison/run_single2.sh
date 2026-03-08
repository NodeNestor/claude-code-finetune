#!/bin/bash
# Run a single comparison test — copies train.py from branch, runs, restores
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
echo "  train.py from: $trainpy"
echo "  extra args: $extra_args"
echo "============================================================"

cd "$REPO"

# Ensure no other containers running
docker stop $(docker ps -q) 2>/dev/null || true
sleep 2

# Backup original and copy branch version
cp train.py train.py.bak
cp "$trainpy" train.py

# Run container in detached mode
docker compose run -d \
    --name "test-${name}" \
    -e WANDB_MODE=disabled \
    train $COMMON_ARGS $extra_args

# Wait for training to start (grad_norm appears), then collect a bit more
echo "  Waiting for profiling to complete..."
for i in $(seq 1 120); do
    sleep 10
    if docker logs "test-${name}" 2>&1 | grep -q "grad_norm"; then
        echo "  Training started! Waiting 60s for more steps..."
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

# Restore original train.py
cp train.py.bak train.py

echo "  -> Log saved to $logfile ($(wc -c < "$logfile") bytes)"
grep -E "(STEP 7 MEM|peak_delta|Per-token peak|Per-token fwd|grad_norm|patched|Compress)" "$logfile" | head -8
echo ""
