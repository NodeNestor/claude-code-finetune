FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV TORCH_COMPILE_DISABLE=1
ENV MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl build-essential ninja-build && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. PyTorch cu128 for Blackwell sm_120
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. flash-linear-attention + causal-conv1d — used by the 24 DeltaNet layers in Qwen3.5
#    GQA layers use SDPA (built into PyTorch) — no flash-attn needed
#    Without these, DeltaNet falls back to O(n²) naive attention and OOMs on 16GB
RUN pip install --no-cache-dir --break-system-packages causal-conv1d
RUN pip install --no-cache-dir --break-system-packages flash-linear-attention

# 4. Unsloth + training stack
RUN pip install --no-cache-dir --break-system-packages \
    unsloth unsloth_zoo bitsandbytes \
    trl datasets peft accelerate \
    sentencepiece protobuf wandb

# 5. Triton for Blackwell kernels + latest transformers for Qwen3.5 support
RUN pip install --no-cache-dir --break-system-packages -U "triton>=3.3.1" transformers

COPY train.py .

ENTRYPOINT ["python", "train.py"]
