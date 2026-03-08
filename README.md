# Claude Code Fine-tune

Fine-tune open-source LLMs on your own [Claude Code](https://docs.anthropic.com/en/docs/claude-code) conversation traces. Train a local model that learns Claude Code's agentic coding style — tool use, thinking patterns, multi-step reasoning, and all.

Built for the **Qwen3.5** family (hybrid DeltaNet/GQA architecture) with QLoRA on consumer GPUs.

| Dense (tested) | MoE (experimental) |
|---|---|
| 0.8B, 2B, 4B, **9B** (default), 27B | 35B-A3B, 122B-A10B, 397B-A17B |

> **MoE note:** The MoE variants share the same base architecture so the training script *should* work, but they're untested. Potential issues: Unsloth MoE support, BNB 4-bit quantization on expert weights, multi-GPU offloading with MoE layer structure, and VRAM requirements (35B-A3B activates 3B per token but needs all 35B resident). If you try it, let us know!

## What this does

1. **Extracts** your Claude Code conversation history from local JSONL logs
2. **Converts** them into a training dataset with realistic system prompts, tool calls, thinking blocks, and tool results
3. **Fine-tunes** using QLoRA with aggressive VRAM optimizations
4. **Exports** to LoRA adapters or GGUF for local inference

## Quick start

### Prerequisites

- NVIDIA GPU (8GB+ for small models, 16GB+ for 9B)
- Docker with NVIDIA Container Toolkit (recommended), or Python 3.10+ with CUDA
- Your Claude Code conversation logs (usually at `~/.claude/projects/`)

### 1. Generate your dataset

```bash
python finetune.py dataset
```

That's it. Auto-detects your Claude Code logs, converts them, splits at compression boundaries. Output: `claude-traces-split.jsonl`.

### 2. Pick your mode

**Single LoRA** — one adapter trained on all your traces (simplest):

```bash
python finetune.py train
```

**Expert swarm** — multiple specialist LoRAs + a router (better for small models):

```bash
# See what expert categories your traces split into
python finetune.py experts --analyze-only

# Train everything (splits → trains each expert → trains router)
python finetune.py experts
```

The expert mode auto-classifies your conversations by what Claude Code *did* (tool patterns, file types, thinking depth) into categories like `code_edit`, `debugging`, `exploration`, `planning`, `frontend`, `testing`, etc. — then trains a narrow LoRA for each + a router that picks the right one.

### 3. Configure (optional)

```bash
cp config.example.yaml config.yaml    # training config
cp .env.example .env                   # API keys (wandb, HF)
```

Config layers: **defaults < config.yaml < CLI flags**. Skip the config file entirely and just use flags, or put everything in `config.yaml`.

### 4. Use the model

```bash
# Single LoRA → checkpoints/claude-code-agent/lora/
# Expert swarm → checkpoints/expert-*/lora/ + checkpoints/router/final/

# Load in Ollama (if you exported GGUF)
ollama create claude-code-local -f checkpoints/claude-code-agent/Modelfile

# Or load the LoRA adapter with any HuggingFace-compatible tool
```

### Docker

```bash
# With Docker (handles all dependencies)
docker compose run train                                          # single LoRA
docker compose run train --seq-length 8192 --epochs 1 --save-steps 50  # quick test

# Without Docker
pip install -r requirements.txt
```

### Advanced: individual scripts

The `finetune.py` wrapper calls these under the hood — use them directly for more control:

| Script | What it does |
|---|---|
| `claude-trace-converter.py` | Convert Claude Code JSONL logs → ShareGPT training format |
| `split_dataset.py` | Split long conversations at compression boundaries |
| `train.py` | QLoRA training with all VRAM optimizations |
| `train_router.py` | Analyze/split/train expert router (analyze, split, train, eval) |

## Configuration

All training parameters can be set via CLI flags, `config.yaml`, or both (CLI wins):

| Parameter | Default | Description |
|---|---|---|
| `--model` | `unsloth/Qwen3.5-9B` | Base model from HuggingFace |
| `--dataset` | `/data/claude-traces-dataset.jsonl` | Training JSONL path |
| `--seq-length` | `8192` | Max sequence length (see VRAM guide below) |
| `--lora-rank` | `64` | LoRA rank (higher = more capacity, more VRAM) |
| `--lora-alpha` | `128` | LoRA alpha (usually 2x rank) |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `1` | Per-device batch size |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--packing` | off | Enable sequence packing |
| `--export-gguf` | off | Export GGUF after training |
| `--export-merged` | off | Export full merged 16-bit model |
| `--offload-layers N` | `-1` | Offload N layers to GPU 1 (-1 = all) |
| `--profile-tensors` | off | Log saved tensor sizes on step 0 |
| `--profile-deep` | off | Full CUDA memory snapshot at peak |

### VRAM guide

| GPU VRAM | Model | Recommended `--seq-length` | Notes |
|---|---|---|---|
| 8 GB | 0.8B, 2B | 8192-16384 | Expert LoRAs, router training |
| 12 GB | 4B | 8192-16384 | Comfortable for short conversations |
| 16 GB | 9B | 16384-32768 | Sweet spot with all optimizations |
| 16 GB + 2nd GPU | 9B | 32768-45056 | Use `--offload-layers -1` |
| 24 GB | 9B, 27B | 32768-65536 | Room for higher LoRA rank |
| 48 GB+ | 27B, MoE | 65536+ | MoE models (35B-A3B, etc.) |

## The interesting bits

This project is less about the fine-tuning recipe and more about the VRAM engineering required to train a 9B parameter model with 32K context on a 16GB GPU. Here's what we had to do:

### Data pipeline

- **Trace converter** (`claude-trace-converter.py`) — parses Claude Code's local JSONL conversation logs, reconstructs dynamic system prompts per conversation (with actual tools used, working directory, git branch, platform), handles streamed assistant chunk merging, deduplication, and path sanitization
- **Session splitter** (`split_dataset.py`) — splits at Claude Code's auto-compression boundaries ("This session is being continued from a previous conversation...") so each segment becomes an independent training example with the original system prompt

### VRAM optimizations (2.8 GB peak savings on 16GB GPU)

| Optimization | Savings | How |
|---|---|---|
| **Apple CCE (Cut Cross-Entropy)** | ~2 GB | Eliminates materializing the full `seq_len x 248K vocab` logits tensor. Had to discover that Unsloth resets `UNSLOTH_ENABLE_CCE=0` during import. |
| **Native fp8 lm_head via Triton source patching** | 1.0 GB | Patches 3 source files in CCE/Unsloth *before import* so Triton JIT compiles with fp8 support. Zero-copy: loads fp8 from VRAM, casts to bf16 in GPU registers. |
| **int8 embed_tokens quantization** | 1.0 GB | Per-row scale factor with on-the-fly dequantization during forward. Embedding lookup doesn't need full precision. |
| **RMSNorm bf16 patch** | ~14% per-token | Monkey-patches all 81 RMSNorm instances to skip fp32 upcasts. Reduces fp32 in saved activations from 40% to 0.4%. |
| **Vision encoder removal** | 0.3 GB | Qwen3.5-9B loads as a VL model with an unused vision encoder. |
| **SDPA forced on GQA layers** | 84% per-layer | Without this, the 8 full-attention layers fall back to eager attention and materialize full `[B, H, N, N]` attention matrices. |
| **Gradient checkpointing CPU offload fix** | 44% peak | SFTTrainer silently replaces Unsloth's CPU-offloading checkpoint function. We patch `_set_gradient_checkpointing` to always use `unsloth_checkpoint`. |

### Multi-GPU weight offloading

Uses a second GPU (even an 8GB one) as pure weight storage. All 32 decoder layers offloaded to GPU 1, shuttled to GPU 0 on-demand via `pre_forward` hooks. All computation stays on GPU 0.

- GPU 0 base allocation: 6.15 GB -> **2.35 GB** (freed 3.8 GB)
- Handles BNB `Params4bit` quirks (`quant_state.to()` is in-place, returns None!)
- Uses a `_last_loaded` pattern instead of post-hooks (which break with gradient checkpointing)

### Memory scaling analysis

After all optimizations, memory scales **linearly** at ~211 KB/token (the earlier quadratic fit was measured before the CPU offloading fix):

```
peak_delta = 4068 + 0.211 * (n - 15000)  MB
```

Theoretical max context with offloading: **~49K tokens** on 16GB.

### What we tried that didn't work

- **int8 saved_tensors_hooks** — crashes with `torch.func.grad_and_value`, grad_norm explosion (1393 vs 4)
- **fp8 MLP activations** — Unsloth's TiledMLP already recomputes MLP forward during backward, so our fp8 patches added overhead on top of recomputation (+5% VRAM)
- **Random projection activation compression** — numerically unstable (SiLU derivative with reconstructed inputs -> grad_norm infinity)
- **Model parallelism** — backward workspace needs ~4GB per GPU independently, too much for 8GB
- **accelerate AlignDevicesHook** — `Params4bit.__new__()` rejects `_is_hf_initialized` kwarg

### Built-in profiling

```bash
python train.py --profile-tensors   # Shows what's eating VRAM during backward
python train.py --profile-deep      # Live CUDA tensor snapshot at peak
```

## Expert LoRA routing (optional)

Instead of one fat LoRA, train narrow expert LoRAs and a tiny router that picks the right one per request. Especially good for smaller models (0.8B-4B) where a single LoRA can't cover everything well.

The router classifies each conversation by what Claude Code **actually does** — tool usage patterns, file types touched, thinking depth — not keyword guessing. Categories are Claude Code-specific:

| Expert | What it handles | How it's detected |
|---|---|---|
| `code_edit` | Implementing features, writing code | Heavy Edit/Write on source files |
| `exploration` | Searching, understanding codebases | Heavy Grep/Glob/Read, minimal edits |
| `debugging` | Finding and fixing bugs | Error results + targeted fix edits |
| `planning` | Architecture, design, reasoning | Long thinking blocks, few tools |
| `commands` | Builds, git, installs, deploys | Heavy Bash usage |
| `frontend` | UI/component work | Edits on .tsx/.css/.html/.vue/.svelte |
| `config` | Docker, CI/CD, yaml, envs | Edits on config files |
| `testing` | Writing/running tests | Test file edits + test commands |
| `refactor` | Restructuring existing code | Many Edits, few new files |
| `compact` | Quick one-shot answers | Short conversations, minimal tools |

### 1. See what you've got

```bash
# Analyze your traces — see how they distribute across categories
python train_router.py analyze --dataset claude-traces-split.jsonl -v
```

### 2. Split into expert datasets + train each expert

```bash
# Split traces into per-expert JSONL files
python train_router.py split --dataset claude-traces-split.jsonl

# Train each expert LoRA
python train.py --model unsloth/Qwen3.5-0.8B \
    --dataset expert-datasets/code_edit.jsonl \
    --output checkpoints/expert-code_edit --lora-rank 32

python train.py --model unsloth/Qwen3.5-0.8B \
    --dataset expert-datasets/debugging.jsonl \
    --output checkpoints/expert-debugging --lora-rank 32

# ... repeat for each expert
```

### 3. Train the router

```bash
# One command — classifies traces + trains router LoRA
python train_router.py train --dataset claude-traces-split.jsonl
```

### 4. Deploy with vLLM multi-LoRA

Load the router + all experts into vLLM. The router classifies each incoming request in a single forward pass (<10ms), then the request gets routed to the right expert LoRA adapter.

## Project structure

```
finetune.py                  Main entry point — dataset, train, or experts mode
train.py                     QLoRA training with all VRAM optimizations
train_router.py              Expert classification, splitting, and router training
claude-trace-converter.py    Claude Code JSONL -> ShareGPT training format
split_dataset.py             Split conversations at compression boundaries
config.example.yaml          Annotated config template
Dockerfile                   CUDA 12.8 + PyTorch + Flash Attention + Unsloth
docker-compose.yml           One-command training with GPU passthrough
CHANGELOG.md                 Detailed engineering log of every optimization
```

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- [Unsloth](https://github.com/unslothai/unsloth) + [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) (for DeltaNet fast path)

The Dockerfile handles all dependencies. For bare metal, `pip install -r requirements.txt`.

## License

[MIT](LICENSE)
