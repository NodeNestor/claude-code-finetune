"""
Fine-tune Qwen3.5 on Claude Code traces using Unsloth QLoRA.

Supports all Qwen3.5 sizes — they share the same hybrid DeltaNet/GQA architecture:
  Dense:  0.8B, 2B, 4B, 9B, 27B
  MoE:    35B-A3B, 122B-A10B, 397B-A17B

Optimized for consumer GPUs (16GB+) with aggressive VRAM engineering.

Usage:
    python train.py                                    # defaults (9B, 8K context)
    python train.py --model unsloth/Qwen3.5-0.8B       # tiny (great for expert LoRAs)
    python train.py --model unsloth/Qwen3.5-4B          # mid-range
    python train.py --model unsloth/Qwen3.5-27B         # large (needs 24GB+)
    python train.py --seq-length 16384                  # longer context
    python train.py --epochs 2 --lr 1e-4                # custom hyperparams
"""

import argparse
import os
import sys
import subprocess

# Auto-install wandb if not present (avoids Docker rebuild)
try:
    import wandb
except ImportError:
    if os.environ.get("WANDB_API_KEY"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "wandb"])
            import wandb
        except Exception:
            print("  wandb install failed, continuing without logging")

# ─── Patch CCE + Unsloth for native fp8 lm_head BEFORE any imports ───
# Must happen before import unsloth/cut_cross_entropy so Triton JIT compiles patched source.
def _patch_cce_for_fp8():
    """Patch 4 lines in CCE/Unsloth so Triton kernels handle fp8 natively (zero copy)."""
    import importlib
    # Find package paths without importing
    import importlib.util
    _cce_spec = importlib.util.find_spec('cut_cross_entropy')
    _zoo_spec = importlib.util.find_spec('unsloth_zoo')
    if not _cce_spec or not _zoo_spec:
        return
    _cce_pkg = os.path.dirname(_cce_spec.origin)
    _zoo_pkg = os.path.dirname(_zoo_spec.origin)

    patched = []

    # 1. indexed_dot.py: add .to(e.dtype) on tl.load(c_ptrs)
    p = os.path.join(_cce_pkg, 'indexed_dot.py')
    with open(p) as f:
        src = f.read()
    if 'tl.load(c_ptrs)' in src and '.to(e.dtype)' not in src.split('def indexed_neg_dot')[0]:
        src = src.replace(
            'c = tl.load(c_ptrs)\n',
            'c = tl.load(c_ptrs).to(e.dtype)\n'
        ).replace(
            'c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0)\n',
            'c = tl.load(c_ptrs, mask=offs_d[None, :] < D, other=0.0).to(e.dtype)\n'
        )
        with open(p, 'w') as f:
            f.write(src)
        patched.append('indexed_dot')

    # 2. cce_backward.py: remove bf16/fp16/fp32 assert
    p = os.path.join(_cce_pkg, 'cce_backward.py')
    with open(p) as f:
        src = f.read()
    if '"Backwards requires classifier' in src:
        src = src.replace(
            'assert c.dtype in (\n        torch.float16,\n        torch.bfloat16,\n        torch.float32,\n    ), "Backwards requires classifier to be bf16 or fp16 or fp32"',
            'pass  # fp8 OK: Triton kernel casts via .to(e.dtype)'
        )
        with open(p, 'w') as f:
            f.write(src)
        patched.append('cce_backward')

    # 3. loss_utils.py: don't cast hidden_states to lm_weight.dtype
    p = os.path.join(_zoo_pkg, 'loss_utils.py')
    with open(p) as f:
        src = f.read()
    if 'hidden_states.to(lm_weight.dtype)' in src:
        src = src.replace(
            'hidden_states.to(lm_weight.dtype)',
            'hidden_states'
        )
        with open(p, 'w') as f:
            f.write(src)
        patched.append('loss_utils')

    # Clear __pycache__ for patched files
    for pkg_dir in [_cce_pkg, _zoo_pkg]:
        cache = os.path.join(pkg_dir, '__pycache__')
        if os.path.isdir(cache):
            import shutil
            shutil.rmtree(cache, ignore_errors=True)

    if patched:
        print(f"  fp8 patches applied: {', '.join(patched)}")

_patch_cce_for_fp8()

import unsloth  # Must be imported before transformers

# CRITICAL: Unsloth resets UNSLOTH_ENABLE_CCE=0 during import.
# Set it back to "1" AFTER import — the compiled module reads it lazily at first forward pass.
os.environ["UNSLOTH_ENABLE_CCE"] = "1"
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


def format_sharegpt_to_chatml(example, tokenizer):
    """Convert ShareGPT format (from/value) to ChatML text via tokenizer."""
    conversations = example.get("conversations", [])
    messages = []
    has_user = False
    for turn in conversations:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(turn["from"], turn["from"])
        if role == "user":
            has_user = True
        messages.append({"role": role, "content": turn["value"]})

    if not has_user or len(messages) < 2:
        return {"text": ""}

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return {"text": ""}
    return {"text": text}


def load_config_file(path="config.yaml"):
    """Load config from YAML file if it exists. Returns dict of overrides."""
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
    except ImportError:
        # Fall back to basic YAML parsing for simple key: value files
        config = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val.startswith("#"):
                    continue
                # Strip inline comments
                if " #" in val:
                    val = val[:val.index(" #")].strip()
                # Parse types
                if val.lower() in ("true", "yes"):
                    config[key] = True
                elif val.lower() in ("false", "no"):
                    config[key] = False
                elif val.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
                    config[key] = float(val) if "." in val or "e" in val.lower() else int(val)
                else:
                    config[key] = val
        return config
    else:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5 on Claude Code traces",
        epilog="Config file: put a config.yaml in the working directory (see config.example.yaml). CLI flags override config file values.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file (default: config.yaml)")
    parser.add_argument("--model", default=None, help="Base model (default: unsloth/Qwen3.5-9B)")
    parser.add_argument("--dataset", default=None, help="Dataset path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--seq-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--eval-split", type=float, default=None, help="Eval split ratio")
    parser.add_argument("--save-steps", type=int, default=None, help="Save checkpoint every N steps")
    parser.add_argument("--packing", action="store_true", default=None, help="Enable sequence packing (better GPU util)")
    parser.add_argument("--export-gguf", action="store_true", default=None, help="Export GGUF after training")
    parser.add_argument("--export-merged", action="store_true", default=None, help="Export merged model after training")
    parser.add_argument("--profile-tensors", action="store_true", default=None, help="Profile saved tensors on step 0 (shows per-token memory breakdown)")
    parser.add_argument("--profile-deep", action="store_true", default=None, help="Deep memory profile: snapshot all live tensors at peak, check CPU offload, per-layer backward")
    parser.add_argument("--offload-layers", type=int, default=None, help="Number of decoder layers to offload to GPU 1 (-1=all that fit)")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Warmup ratio")
    cli_args = parser.parse_args()

    # Layer: defaults < config.yaml < CLI flags
    defaults = {
        "model": "unsloth/Qwen3.5-9B",
        "dataset": "/data/claude-traces-dataset.jsonl",
        "output": "/checkpoints/claude-code-agent",
        "seq_length": 8192,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 2e-4,
        "lora_rank": 64,
        "lora_alpha": 128,
        "eval_split": 0.02,
        "save_steps": 200,
        "packing": False,
        "export_gguf": False,
        "export_merged": False,
        "profile_tensors": False,
        "profile_deep": False,
        "offload_layers": -1,
        "warmup_ratio": 0.05,
    }

    # Load config file
    config = load_config_file(cli_args.config)

    # Merge: defaults -> config.yaml -> CLI
    final = dict(defaults)
    for k, v in config.items():
        if v is not None:
            final[k] = v

    # CLI overrides (map hyphenated names to underscored)
    cli_dict = vars(cli_args)
    for cli_key, val in cli_dict.items():
        if cli_key == "config":
            continue
        cfg_key = cli_key.replace("-", "_")
        if val is not None:
            final[cfg_key] = val

    # Build args namespace from merged config
    args = argparse.Namespace(**final)

    print(f"{'='*60}")
    print(f"Claude Code Agent Fine-tuning")
    print(f"{'='*60}")
    print(f"  Model:        {args.model}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Seq length:   {args.seq_length}")
    print(f"  LoRA:         rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch:        {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  LR:           {args.lr}")
    print(f"  Packing:      {args.packing}")
    print(f"  Output:       {args.output}")
    print(f"  CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ─── Load model ───
    print("Loading model...")
    # Single-threaded loading prevents VRAM spikes on 16GB GPUs
    import transformers.core_model_loading as _cml
    _cml.GLOBAL_WORKERS = 1

    _n_gpus = torch.cuda.device_count()
    _load_kwargs = dict(
        model_name=args.model,
        max_seq_length=args.seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        unsloth_tiled_mlp=True,  # ~40% less activation memory for long context
    )
    if _n_gpus > 1:
        print(f"  Multi-GPU: {_n_gpus} GPUs detected (weight offloading mode)")
        for i in range(_n_gpus):
            _name = torch.cuda.get_device_name(i)
            _mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    GPU {i}: {_name} ({_mem:.1f} GB)")
        print(f"    Strategy: load on GPU 0, offload layer weights to GPU 1 after setup")
    model, tokenizer = FastLanguageModel.from_pretrained(**_load_kwargs)

    # ─── Force SDPA on GQA attention layers ───
    # Without this, Qwen3.5's 8 full_attention layers fall back to eager attention
    # which materializes the full [B, H, N, N] attention matrix — O(n²) memory!
    # SDPA uses Flash Attention under the hood — O(n) memory, no n×n matrix.
    # Profiled: eager=923 KB/tok/layer at 8K, SDPA=151 KB/tok/layer (84% reduction)
    _text_config = getattr(model.config, "text_config", model.config)
    _old_attn = getattr(_text_config, "_attn_implementation", None)
    _text_config._attn_implementation = "sdpa"
    if hasattr(_text_config, "attn_implementation"):
        _text_config.attn_implementation = "sdpa"
    # Also set on nested configs that attention layers reference
    for _m in model.modules():
        if hasattr(_m, "config") and hasattr(_m.config, "_attn_implementation"):
            _m.config._attn_implementation = "sdpa"
    print(f"  Attention: forced SDPA (was {_old_attn!r}) — eliminates O(n²) attention matrix")

    # Check if DeltaNet fast path is available
    try:
        import fla  # flash-linear-attention
        print("  DeltaNet fast path: ENABLED (flash-linear-attention)")
    except ImportError:
        print("  DeltaNet fast path: DISABLED (will use O(n^2) fallback!)")

    try:
        import flash_attn
        print(f"  Flash Attention 2:  ENABLED (v{flash_attn.__version__})")
    except ImportError:
        print("  Flash Attention 2:  DISABLED (using xformers fallback)")

    # ─── Apply LoRA ───
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,  # Must be 0 for Unsloth fast patching
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM than standard
        random_state=42,
    )

    # ─── Remove vision encoder (unused for text-only training) ───
    import gc
    for vision_attr in ["visual", "vision_model", "vision_tower", "multi_modal_projector"]:
        # Check through PEFT wrapper chain
        for obj_path in [model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)]:
            if obj_path is not None and hasattr(obj_path, vision_attr):
                vis = getattr(obj_path, vision_attr)
                if vis is not None:
                    vis_size = sum(p.nelement() * p.element_size() for p in vis.parameters())
                    setattr(obj_path, vision_attr, None)
                    print(f"  Removed {vision_attr}: freed {vis_size/1e9:.2f} GB")
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Optimize large unquantized tensors ───
    target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Find the base model through PEFT wrapper
    # Qwen3.5 VL: model.base_model.model.model.language_model
    base = None
    for path in [
        lambda m: m.model.model.model.language_model,
        lambda m: m.base_model.model.model.language_model,
        lambda m: m.model.model.language_model,
        lambda m: m.model.language_model,
        lambda m: m.language_model,
        lambda m: m.model.model.model,
        lambda m: m.model.model,
        lambda m: m.model,
        lambda m: m,
    ]:
        try:
            candidate = path(model)
            if hasattr(candidate, "lm_head") and hasattr(candidate, "model"):
                base = candidate
                break
        except (AttributeError, TypeError):
            continue

    if base is not None:
        # Ensure lm_head is bf16 (not float32) so CCE activates
        if base.lm_head.weight.dtype == torch.float32:
            base.lm_head.weight.data = base.lm_head.weight.data.to(target_dtype)
            print(f"  lm_head cast to {target_dtype} (enables Apple CCE)")
        print(f"  lm_head: {base.lm_head.weight.dtype}, requires_grad={base.lm_head.weight.requires_grad}")

        # lm_head stays bf16 (2.03 GB) — required for CCE, fp8 tested and reverted

        # Debug: test CCE condition exactly as compiled module does
        requires_grad_ = base.lm_head.weight.requires_grad
        requires_grad_ = requires_grad_ or base.lm_head.weight.dtype == torch.float32
        UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
        NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
        loss_fn_name = getattr(base, 'loss_function', None)
        if loss_fn_name is not None:
            loss_fn_name = loss_fn_name.__name__
        print(f"  CCE conditions: enable={UNSLOTH_ENABLE_CCE}, not_return={NOT_RETURN_LOGITS}, loss_fn={loss_fn_name}, req_grad={requires_grad_}")
        print(f"  CCE WILL ACTIVATE: {UNSLOTH_ENABLE_CCE and NOT_RETURN_LOGITS and (loss_fn_name or '').endswith('ForCausalLMLoss') and not requires_grad_}")

        # Report embed_tokens size — search through model hierarchy
        embed = None
        for obj in [base, getattr(base, "model", None)]:
            if obj is None:
                continue
            if hasattr(obj, "embed_tokens"):
                embed = obj.embed_tokens
                break
        # Also check through the full PEFT chain
        if embed is None:
            for name, module in model.named_modules():
                if name.endswith("embed_tokens"):
                    embed = module
                    print(f"  embed_tokens found at: {name}")
                    break
        if embed is not None:
            emb_size = embed.weight.nelement() * embed.weight.element_size()
            print(f"  embed_tokens: {embed.weight.dtype}, {emb_size/1e9:.2f} GB")

            # Quantize embed_tokens to int8 — saves ~1 GB
            # Embedding lookup doesn't need full precision, we dequant on the fly
            if embed.weight.dtype in (torch.bfloat16, torch.float16):
                w = embed.weight.data.clone()  # clone so we can free original
                scale = w.abs().amax(dim=1, keepdim=True) / 127.0
                scale = scale.clamp(min=1e-8)
                int8_w = (w / scale).clamp(-127, 127).to(torch.int8)
                embed._embed_scale = scale.squeeze(1).to("cuda")  # [vocab_size]
                embed._embed_dtype = w.dtype
                # Replace weight with int8 and free bf16
                embed.weight = torch.nn.Parameter(int8_w, requires_grad=False)
                del w, int8_w
                import gc; gc.collect()
                torch.cuda.empty_cache()

                def _quantized_embed_forward(input_ids, _embed=embed):
                    int8_out = torch.nn.functional.embedding(input_ids, _embed.weight.data.to(torch.int16).to(_embed._embed_dtype))
                    scales = _embed._embed_scale[input_ids]
                    return int8_out * scales.unsqueeze(-1)

                embed.forward = _quantized_embed_forward
                new_size = embed.weight.nelement() * embed.weight.element_size() + embed._embed_scale.nelement() * embed._embed_scale.element_size()
                saved = emb_size - new_size
                alloc_after = torch.cuda.memory_allocated() / 1e9
                print(f"  embed_tokens quantized to int8: {new_size/1e9:.2f} GB (saved {saved/1e9:.2f} GB)")
                print(f"  CUDA allocated after embed quant: {alloc_after:.2f} GB")
    else:
        print("  WARNING: base model not found for optimization")

    # ─── Patch RMSNorm to stay in bf16 (eliminates 40% of saved activation memory) ───
    # Qwen3_5RMSNorm upcasts to fp32 for numerical stability in _norm() and forward().
    # This causes 40% of all saved-for-backward tensors to be fp32 instead of bf16.
    # Patching to bf16 halves those tensors — no copies, just preventing the upcast.
    # bf16 has enough precision for RMSNorm during training (verified by stable grad_norm).
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    _original_rmsnorm_forward = Qwen3_5RMSNorm.forward

    def _bf16_rmsnorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.bfloat16).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * (1.0 + self.weight.to(hidden_states.dtype))).to(input_dtype)

    Qwen3_5RMSNorm.forward = _bf16_rmsnorm_forward
    # Count how many RMSNorm instances exist
    _norm_count = sum(1 for m in model.modules() if isinstance(m, Qwen3_5RMSNorm))
    print(f"  RMSNorm patched to bf16: {_norm_count} instances (saves ~40% activation memory)")

    # ─── Detailed VRAM breakdown ───
    if torch.cuda.is_available():
        print("\n  VRAM breakdown by module:")
        module_sizes = {}
        for name, param in model.named_parameters():
            size = param.nelement() * param.element_size()
            # Group by top-level module
            parts = name.split(".")
            # Find a meaningful group name
            if "embed_tokens" in name:
                group = "embed_tokens"
            elif "lm_head" in name:
                group = "lm_head"
            elif "lora" in name.lower():
                group = "lora_adapters"
            elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                group = "attention_weights"
            elif any(x in name for x in ["gate_proj", "up_proj", "down_proj"]):
                group = "mlp_weights"
            elif "norm" in name.lower():
                group = "layer_norms"
            else:
                group = "other"
            module_sizes[group] = module_sizes.get(group, 0) + size

        for group, size in sorted(module_sizes.items(), key=lambda x: -x[1]):
            print(f"    {group:<25} {size/1e9:.3f} GB")
        total_params = sum(module_sizes.values())
        print(f"    {'TOTAL':<25} {total_params/1e9:.3f} GB")
        print(f"    CUDA allocated:          {torch.cuda.memory_allocated()/1e9:.3f} GB")
        print(f"    CUDA reserved:           {torch.cuda.memory_reserved()/1e9:.3f} GB")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ─── Load dataset ───
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"  Raw examples: {len(dataset)}")

    # Convert ShareGPT -> ChatML text
    print("Formatting with chat template...")
    dataset = dataset.map(
        lambda ex: format_sharegpt_to_chatml(ex, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Formatting",
    )

    # Filter out empty/invalid examples
    before = len(dataset)
    dataset = dataset.filter(lambda ex: len(ex["text"]) > 0)
    print(f"  After validity filter: {len(dataset)} (dropped {before - len(dataset)} invalid)")

    # Filter by length (rough char estimate: ~4 chars per token)
    max_chars = args.seq_length * 4
    before = len(dataset)
    dataset = dataset.filter(lambda ex: len(ex["text"]) <= max_chars)
    print(f"  After length filter: {len(dataset)} (dropped {before - len(dataset)} too-long)")

    # Split
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train: {len(train_dataset)}, Eval: none")

    # ─── Training ───
    os.makedirs(args.output, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=args.output,
            max_seq_length=args.seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            logging_steps=1,
            save_steps=args.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=args.save_steps if eval_dataset else None,
            save_total_limit=3,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_num_proc=4,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name=f"{args.model.split('/')[-1]}-lora-r{args.lora_rank}-ctx{args.seq_length}",
            packing=args.packing,
        ),
    )

    # ─── Quantize lm_head to fp8 (after SFTTrainer init) ───
    # Must be after init: Unsloth's fix_untrained_tokens needs bf16 for torch.amax.
    # CCE Triton kernels handle fp8 natively; source patched at top of file.
    if base is not None and base.lm_head.weight.dtype in (torch.bfloat16, torch.float16) and not base.lm_head.weight.requires_grad:
        lm_size_before = base.lm_head.weight.nelement() * base.lm_head.weight.element_size()
        _old_w = base.lm_head.weight.data
        _old_ptr = _old_w.data_ptr()
        _old_size = _old_w.nelement() * _old_w.element_size()
        _fp8_w = _old_w.to(torch.float8_e4m3fn)

        # Replace the parameter
        base.lm_head.weight = torch.nn.Parameter(_fp8_w, requires_grad=False)
        del _fp8_w, _old_w
        gc.collect()
        torch.cuda.empty_cache()

        alloc_after = torch.cuda.memory_allocated() / 1e9
        lm_size_after = base.lm_head.weight.nelement() * base.lm_head.weight.element_size()
        print(f"  lm_head quantized to fp8: {lm_size_after/1e9:.2f} GB (saved {(lm_size_before-lm_size_after)/1e9:.2f} GB)")
        print(f"  CUDA allocated after fp8 quant: {alloc_after:.2f} GB")

        # Check if old tensor was freed by looking at allocation delta
        expected = 7.17 - 1.02 + 1.02  # embed_quant_alloc - bf16_lm + fp8_lm = 7.17
        if alloc_after > expected + 0.5:
            ghost_gb = alloc_after - expected
            print(f"  WARNING: {ghost_gb:.2f} GB ghost! Scanning named params for bf16 {_old_ptr}...")
            for name, param in model.named_parameters():
                if param.data.data_ptr() == _old_ptr:
                    print(f"    Ghost at: {name} ({param.dtype})")
            for name, buf in model.named_buffers():
                if buf.data_ptr() == _old_ptr:
                    print(f"    Ghost buffer at: {name} ({buf.dtype})")
            # Brute force: list all large CUDA tensors
            print(f"  Large CUDA tensors (>500MB):")
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda and obj.nelement() * obj.element_size() > 500_000_000:
                    print(f"    {obj.shape} {obj.dtype} {obj.nelement()*obj.element_size()/1e9:.2f}GB ptr={obj.data_ptr()}")

    # Report memory before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n  VRAM before training:")
        print(f"    Allocated: {alloc:.2f} GB")
        print(f"    Reserved:  {reserved:.2f} GB")

    # ─── Fix gradient checkpointing: enable Unsloth CPU offloading ───
    # SFTTrainer.__init__ calls gradient_checkpointing_enable() which sets
    # _gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    # This uses non-reentrant checkpointing WITHOUT CPU offloading.
    # Fix: replace with unsloth_checkpoint which does async CPU offloading.
    from unsloth_zoo.gradient_checkpointing import unsloth_checkpoint as _unsloth_ckpt
    # The Trainer calls gradient_checkpointing_enable() during train(), which resets
    # _gradient_checkpointing_func to partial(checkpoint, use_reentrant=False).
    # Fix: monkey-patch _set_gradient_checkpointing to always use unsloth_checkpoint.
    from transformers import PreTrainedModel
    _orig_set_gc = PreTrainedModel._set_gradient_checkpointing
    def _patched_set_gc(self, enable=True, gradient_checkpointing_func=None):
        _orig_set_gc(self, enable=enable, gradient_checkpointing_func=gradient_checkpointing_func)
        if enable:
            for module in self.modules():
                if hasattr(module, "_gradient_checkpointing_func"):
                    module._gradient_checkpointing_func = _unsloth_ckpt
    PreTrainedModel._set_gradient_checkpointing = _patched_set_gc
    print(f"  Gradient checkpointing: patched _set_gradient_checkpointing → Unsloth CPU offloading")

    # ─── Multi-GPU weight offloading ───
    # Store decoder layer weights on GPU 1, shuttle to GPU 0 for computation.
    # All activations/backward stay on GPU 0. GPU 1 is pure weight storage.
    # Uses module.to() which handles BNB 4-bit Params4bit correctly.
    #
    # Hook strategy:
    #   forward_pre_hook:  layer.to('cuda:0')  — load weights for forward
    #   forward_hook:      layer.to('cuda:1')  — offload after forward (skipped during backward)
    #   backward_hook:     layer.to('cuda:1')  — offload after backward completes
    # During backward recomputation (gradient checkpointing):
    #   forward_pre_hook fires again → weights loaded to cuda:0
    #   forward_hook is SKIPPED (flag set) → weights stay on cuda:0 for gradient computation
    #   backward_hook fires → weights offloaded back to cuda:1
    _lm_head_offloaded = False  # set True if lm_head moved to GPU 1
    _embed_offloaded = False    # set True if embed_tokens moved to GPU 1
    if _n_gpus > 1:
        # Verify GPU 0 (compute) is the bigger GPU — model loads there by default
        _gpu0_mem = torch.cuda.get_device_properties(0).total_memory
        _gpu1_mem = torch.cuda.get_device_properties(1).total_memory
        if _gpu1_mem > _gpu0_mem:
            print(f"  WARNING: GPU 1 ({torch.cuda.get_device_name(1)}, {_gpu1_mem/1e9:.0f}GB) is larger than GPU 0 ({torch.cuda.get_device_name(0)}, {_gpu0_mem/1e9:.0f}GB)!")
            print(f"  Swap CUDA_VISIBLE_DEVICES order so the bigger GPU is first (compute device).")
            print(f"  Continuing without weight offloading...")
            _n_gpus = 1  # disable offloading
        _offload_dev = torch.device("cuda:1")
        _exec_dev = torch.device("cuda:0")
        _gpu1_capacity = torch.cuda.get_device_properties(1).total_memory
        _gpu1_used = 0
        _gpu1_limit = int(_gpu1_capacity * 0.90)  # leave 10% margin (GPU 1 is pure frozen weight storage)
        _offloaded_bytes = 0
        _offloaded_count = 0

        # Find decoder layers through PEFT wrapper
        _layers_obj = model
        for _attr in ["model", "base_model", "model", "model", "language_model", "model"]:
            if hasattr(_layers_obj, _attr):
                _layers_obj = getattr(_layers_obj, _attr)

        def _move_layer(layer, device):
            """Move only frozen weights to device — LoRA params stay on compute GPU to avoid
            leaking gradients/optimizer state to the storage GPU and OOMing it."""
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    continue  # skip LoRA adapters — keep on compute GPU
                param.data = param.data.to(device, non_blocking=True)
                qs = getattr(param, "quant_state", None)
                if qs is not None:
                    qs.to(device)

        # Track the last loaded layer so we can offload it when the next one loads.
        # This avoids post_hook timing issues with gradient checkpointing backward.
        _last_loaded = [None]
        _hook_transfer_log = []  # Memory before/after each transfer for debugging
        _hook_log_enabled = [True]  # Only log first few steps

        def _setup_offload_hooks(layer, storage_dev, compute_dev):
            """Add hooks to shuttle layer weights between storage_dev and compute_dev."""

            def _pre_fwd(module, inputs):
                if _hook_log_enabled[0]:
                    _mem_before = torch.cuda.memory_allocated(0) / 1e6
                # Offload the previously loaded layer (if different)
                if _last_loaded[0] is not None and _last_loaded[0] is not module:
                    _move_layer(_last_loaded[0], storage_dev)
                if _hook_log_enabled[0]:
                    _mem_mid = torch.cuda.memory_allocated(0) / 1e6
                # Load this layer
                _move_layer(module, compute_dev)
                _last_loaded[0] = module
                if _hook_log_enabled[0]:
                    _mem_after = torch.cuda.memory_allocated(0) / 1e6
                    _hook_transfer_log.append((_mem_before, _mem_mid, _mem_after))

            layer.register_forward_pre_hook(_pre_fwd)

        if hasattr(_layers_obj, "layers"):
            _decoder_layers = _layers_obj.layers
            n_layers = len(_decoder_layers)
            _max_offload = args.offload_layers if args.offload_layers >= 0 else n_layers
            print(f"\n  Weight offloading to GPU 1:")
            print(f"    GPU 1 capacity: {_gpu1_capacity/1e9:.1f} GB (limit: {_gpu1_limit/1e9:.1f} GB)")
            print(f"    Max layers to offload: {_max_offload} {'(all)' if _max_offload >= n_layers else '(partial)'}")

            for li in range(n_layers):
                if _offloaded_count >= _max_offload:
                    print(f"    Reached offload limit ({_max_offload} layers)")
                    break
                _layer_bytes = sum(
                    p.nelement() * p.element_size()
                    for p in _decoder_layers[li].parameters()
                    if p.device.type != "meta"
                )
                if _gpu1_used + _layer_bytes > _gpu1_limit:
                    print(f"    GPU 1 full after {_offloaded_count} layers ({_gpu1_used/1e9:.2f} GB)")
                    break

                # Move layer to GPU 1 and add hooks
                _move_layer(_decoder_layers[li], _offload_dev)
                _setup_offload_hooks(_decoder_layers[li], _offload_dev, _exec_dev)
                _gpu1_used += _layer_bytes
                _offloaded_bytes += _layer_bytes
                _offloaded_count += 1

            torch.cuda.empty_cache()
            _alloc_after = torch.cuda.memory_allocated(0) / 1e9
            _alloc_gpu1 = torch.cuda.memory_allocated(1) / 1e9
            print(f"    Offloaded {_offloaded_count}/{n_layers} layers ({_offloaded_bytes/1e9:.2f} GB)")
            print(f"    GPU 0 allocated after offload: {_alloc_after:.2f} GB")
            print(f"    GPU 1 allocated: {_alloc_gpu1:.2f} GB")

        # ─── Offload embed_tokens to GPU 1 ───
        # embed_tokens is only used once at the start of forward.
        # Move weight+scale to GPU 1, compute there, transfer output to GPU 0.
        # Strategy: keep weights on GPU 1 permanently. Use hooks to:
        # 1. Pre-hook: move input_ids to GPU 1 (tiny — just token indices)
        # 2. Forward runs on GPU 1 with weights already there (no copy)
        # 3. Post-hook: move output embedding back to GPU 0 for decoder layers
        if embed is not None:
            _embed_before = torch.cuda.memory_allocated(0) / 1e9
            embed.weight.data = embed.weight.data.to(_offload_dev, non_blocking=True)
            if hasattr(embed, '_embed_scale'):
                embed._embed_scale = embed._embed_scale.to(_offload_dev, non_blocking=True)
            _embed_offloaded = True

            def _embed_pre_hook(module, args, _dev=_offload_dev, _exec=_exec_dev):
                """Ensure embed weights on GPU 1, move input_ids there."""
                # Accelerate may move weights back to GPU 0 — re-offload if needed
                if module.weight.device != _dev:
                    module.weight.data = module.weight.data.to(_dev)
                    if hasattr(module, '_embed_scale'):
                        module._embed_scale = module._embed_scale.to(_dev)
                    torch.cuda.empty_cache()
                input_ids = args[0]
                if input_ids.device != _dev:
                    return (input_ids.to(_dev),) + args[1:]
                return args

            def _embed_post_hook(module, args, output, _dev=_exec_dev):
                """Move embedding output back to GPU 0 for decoder layers."""
                if output.device != _dev:
                    return output.to(_dev)
                return output

            embed.register_forward_pre_hook(_embed_pre_hook)
            embed.register_forward_hook(_embed_post_hook)

            torch.cuda.empty_cache()
            _embed_saved = _embed_before - torch.cuda.memory_allocated(0) / 1e9
            _alloc_gpu1 = torch.cuda.memory_allocated(1) / 1e9
            print(f"    embed_tokens → GPU 1 (saved {_embed_saved:.2f} GB on GPU 0)")
            print(f"    GPU 1 allocated: {_alloc_gpu1:.2f} GB")

        # NOTE: lm_head offloading disabled — shuttle approach moves weight to GPU 0
        # for the entire training step (CCE needs it), so it doesn't reduce peak VRAM.
        # Only embed_tokens benefits from offloading (via hooks, weights on GPU 0 briefly).

        _alloc_gpu0_final = torch.cuda.memory_allocated(0) / 1e9
        _alloc_gpu1_final = torch.cuda.memory_allocated(1) / 1e9
        print(f"    Final: GPU 0 = {_alloc_gpu0_final:.2f} GB, GPU 1 = {_alloc_gpu1_final:.2f} GB")

    print(f"\nStarting training...")
    print(f"  Total steps: ~{len(train_dataset) * args.epochs // (args.batch_size * args.grad_accum)}")
    # ─── Per-operation VRAM profiling ───
    # Hooks into the model to measure memory at each phase of training
    from transformers import TrainerCallback
    import subprocess

    def gpu_nvidia_smi_mb():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            return int(out.strip().split("\n")[0])
        except Exception:
            return -1

    def mem_mb():
        return torch.cuda.memory_allocated() / 1e6

    def peak_mb():
        return torch.cuda.max_memory_allocated() / 1e6

    # Add forward hooks to key modules to track per-layer memory
    _hook_log = []
    _profiling_active = False

    def make_hook(name):
        def hook(module, input, output):
            if _profiling_active:
                _hook_log.append((name, mem_mb()))
        return hook

    # Find the actual model layers through PEFT wrapper
    hooks = []
    inner = model
    for attr in ["model", "base_model", "model", "model", "language_model", "model"]:
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    if hasattr(inner, "embed_tokens"):
        hooks.append(inner.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
    if hasattr(inner, "layers"):
        # Hook first, middle, last transformer layer
        n = len(inner.layers)
        for i in [0, n//4, n//2, 3*n//4, n-1]:
            hooks.append(inner.layers[i].register_forward_hook(make_hook(f"layer_{i}")))
    if hasattr(inner, "norm"):
        hooks.append(inner.norm.register_forward_hook(make_hook("final_norm")))
    # Hook lm_head via the parent
    if base is not None and hasattr(base, "lm_head"):
        hooks.append(base.lm_head.register_forward_hook(make_hook("lm_head")))

    # ─── Deep profiler: backward hooks + peak snapshot ───
    _bwd_hook_log = []
    _peak_snapshot_done = [False]

    if args.profile_deep and hasattr(inner, "layers"):
        # Register BACKWARD hooks on every layer to track memory during backward
        def make_bwd_hook(name):
            def hook(module, grad_input, grad_output):
                if _profiling_active:
                    torch.cuda.synchronize()
                    _bwd_hook_log.append((name, torch.cuda.memory_allocated() / 1e6))
            return hook
        for i in range(len(inner.layers)):
            inner.layers[i].register_full_backward_hook(make_bwd_hook(f"bwd_layer_{i}"))
        print(f"  Deep profiler: backward hooks on {len(inner.layers)} layers")

        # Check CPU offloading state
        _offload_active = getattr(model, '_offloaded_gradient_checkpointing', False)
        print(f"  CPU offloading: {'ACTIVE' if _offload_active else 'NOT DETECTED'}")

    def _snapshot_all_cuda_tensors():
        """Snapshot every live CUDA tensor — the definitive view of what's in VRAM."""
        import gc
        gc.collect()
        torch.cuda.synchronize()

        tensors = []
        seen_ptrs = set()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    ptr = obj.data_ptr()
                    if ptr in seen_ptrs:
                        continue
                    seen_ptrs.add(ptr)
                    nbytes = obj.nelement() * obj.element_size()
                    # Try to identify the tensor
                    name = "unknown"
                    if nbytes > 500_000:  # Only track >500KB tensors
                        tensors.append({
                            'shape': tuple(obj.shape),
                            'dtype': str(obj.dtype),
                            'bytes': nbytes,
                            'requires_grad': obj.requires_grad,
                            'is_leaf': obj.is_leaf,
                            'grad_fn': type(obj.grad_fn).__name__ if obj.grad_fn else 'None',
                        })
            except Exception:
                continue

        tensors.sort(key=lambda x: -x['bytes'])
        total = sum(t['bytes'] for t in tensors)
        total_alloc = torch.cuda.memory_allocated()

        print(f"\n  === LIVE CUDA TENSOR SNAPSHOT ===")
        print(f"  Tracked: {len(tensors)} tensors >500KB, {total/1024**2:.0f} MB")
        print(f"  Torch allocated: {total_alloc/1024**2:.0f} MB")
        print(f"  Untracked (small/internal): {(total_alloc - total)/1024**2:.0f} MB")

        # Group by category
        from collections import defaultdict
        cats = defaultdict(lambda: {'bytes': 0, 'count': 0})
        for t in tensors:
            shape = t['shape']
            dtype = t['dtype']
            grad = t['requires_grad']
            gfn = t['grad_fn']

            # Categorize by shape/dtype patterns
            if len(shape) == 2 and shape[0] == 248320:
                cat = 'lm_head/embed (vocab)'
            elif len(shape) == 2 and shape[1] in (32, 64, 128) and not grad:
                cat = 'LoRA weights (small)'
            elif len(shape) == 2 and not grad and t['bytes'] > 10_000_000:
                cat = 'Model weights (frozen)'
            elif 'bfloat16' in dtype and grad and len(shape) >= 2 and shape[-1] == 4096:
                cat = f'Activations (hidden=4096) grad_fn={gfn}'
            elif 'bfloat16' in dtype and grad and len(shape) >= 2 and shape[-1] == 12288:
                cat = f'Activations (intermediate=12288) grad_fn={gfn}'
            elif 'bfloat16' in dtype and len(shape) >= 2 and max(shape) > 1000:
                cat = f'Activations (other) grad_fn={gfn}'
            elif grad:
                cat = f'Gradient tensors grad_fn={gfn}'
            elif 'int8' in dtype or 'uint8' in dtype:
                cat = 'Quantized weights'
            elif 'float8' in dtype:
                cat = 'fp8 weights'
            else:
                cat = f'Other ({dtype}, {"grad" if grad else "no_grad"})'

            cats[cat]['bytes'] += t['bytes']
            cats[cat]['count'] += 1

        print(f"\n  {'Category':<55s} {'MB':>8s} {'%':>6s} {'#':>5s}")
        print(f"  {'-'*78}")
        for cat, data in sorted(cats.items(), key=lambda x: -x[1]['bytes']):
            mb = data['bytes'] / 1024**2
            pct = 100 * data['bytes'] / max(total, 1)
            print(f"  {cat:<55s} {mb:8.1f} {pct:5.1f}% {data['count']:5d}")

        # Top 30 individual tensors
        print(f"\n  Top 30 tensors:")
        print(f"  {'Shape':<30s} {'dtype':<15s} {'MB':>8s} {'grad':>5s} {'grad_fn':<25s}")
        print(f"  {'-'*87}")
        for t in tensors[:30]:
            mb = t['bytes'] / 1024**2
            print(f"  {str(t['shape']):<30s} {t['dtype']:<15s} {mb:8.1f} {'yes' if t['requires_grad'] else 'no':>5s} {t['grad_fn']:<25s}")

    # Override training_step to profile forward vs backward vs loss
    original_training_step = trainer.training_step
    _step_count = [0]
    _saved_tensor_info = []  # For --profile-tensors

    def profiled_training_step(model, inputs, num_items_in_batch=None):
        nonlocal _profiling_active
        step = _step_count[0]
        _step_count[0] += 1

        # Profile first 3 steps + every 20th
        # 3. Reclaim allocator cache between steps to prevent bloat
        if step > 0:
            torch.cuda.empty_cache()

        # Log seq_len and peak VRAM every step (for wandb)
        seq_len = 0
        if hasattr(inputs, 'keys'):
            for key in ["input_ids", "labels", "attention_mask"]:
                if key in inputs and hasattr(inputs[key], 'shape'):
                    seq_len = inputs[key].shape[-1] if inputs[key].dim() > 1 else inputs[key].shape[0]
                    break

        if step < 8:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = mem_mb()
            nvsmi_before = gpu_nvidia_smi_mb()
            _hook_log.clear()
            _profiling_active = True

            # On step 0 with --profile-tensors, hook saved tensors
            _tensor_ctx = None
            if step == 0 and args.profile_tensors:
                import traceback as _tb
                _saved_tensor_info.clear()
                def _prof_pack(tensor):
                    frames = _tb.extract_stack()
                    src = 'unknown'
                    for f in reversed(frames):
                        fn = f.filename.replace('\\', '/')
                        if any(k in fn for k in ['modeling_qwen3_5', 'linear_attention', 'causal_conv1d',
                                                   'lora', 'delta', 'fla/', 'chunk', 'unsloth']):
                            src = f'{os.path.basename(f.filename)}:{f.lineno} {f.name}'
                            break
                        if 'torch/nn' in fn or 'functional' in fn:
                            src = f'{os.path.basename(f.filename)}:{f.lineno} {f.name}'
                    _saved_tensor_info.append({
                        'shape': tuple(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'bytes': tensor.nelement() * tensor.element_size(),
                        'source': src,
                    })
                    return tensor
                _tensor_ctx = torch.autograd.graph.saved_tensors_hooks(_prof_pack, lambda t: t)
                _tensor_ctx.__enter__()

            # Run the actual training step
            loss = original_training_step(model, inputs, num_items_in_batch)

            if _tensor_ctx is not None:
                _tensor_ctx.__exit__(None, None, None)

            _profiling_active = False
            torch.cuda.synchronize()
            after = mem_mb()
            step_peak = peak_mb()
            nvsmi_after = gpu_nvidia_smi_mb()

            # Detailed nvidia-smi stats
            try:
                nv_out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
                     "--format=csv,noheader,nounits"], text=True
                ).strip().split("\n")[0].split(", ")
                nv_used, nv_free, nv_total, nv_util, nv_temp = nv_out
            except Exception:
                nv_used = nv_free = nv_total = nv_util = nv_temp = "?"

            print(f"\n  === STEP {step} MEMORY PROFILE (seq_len={seq_len}) ===")
            print(f"  GPU:   used={nv_used}MB / {nv_total}MB  free={nv_free}MB  util={nv_util}%  temp={nv_temp}C")
            print(f"  Torch: alloc={before:.0f}->{after:.0f}MB  peak={step_peak:.0f}MB  peak_delta={step_peak-before:.0f}MB ({(step_peak-before)/1024:.2f}GB)")
            print(f"  NvSMI: {nvsmi_before}->{nvsmi_after}MB  delta={nvsmi_after-nvsmi_before:+}MB")
            gap = int(nv_used) - after if nv_used != "?" else 0
            print(f"  Non-torch GPU overhead: {gap:.0f}MB (nvidia-smi - torch_alloc)")

            if _hook_log:
                print(f"  Forward pass breakdown:")
                prev = before
                for name, current in _hook_log:
                    delta = current - prev
                    print(f"    {name:<20} {current:>8.0f}MB  ({delta:+.0f}MB)")
                    prev = current
                fwd_end = _hook_log[-1][1]
                fwd_total = fwd_end - before
                bwd_total = step_peak - fwd_end
                print(f"  -----------------------------------------")
                print(f"  Forward total:  {fwd_total:.0f}MB ({fwd_total/1024:.2f}GB)")
                print(f"  Backward peak:  {bwd_total:.0f}MB ({bwd_total/1024:.2f}GB)")
                print(f"  Per-token fwd:  {fwd_total*1024/max(seq_len,1):.1f}KB/tok")
                print(f"  Per-token peak: {(step_peak-before)*1024/max(seq_len,1):.1f}KB/tok")

            # Print saved tensor breakdown if profiling was active
            if _saved_tensor_info and seq_len > 0:
                from collections import defaultdict as _ddict
                cats = _ddict(lambda: {'bytes': 0, 'count': 0, 'tensors': []})
                for info in _saved_tensor_info:
                    src = info['source'].lower()
                    if 'conv1d' in src or 'causal_conv' in src:
                        cat = 'DeltaNet causal_conv1d'
                    elif 'delta' in src or 'linear_attention' in src or 'chunk' in src or 'fla/' in src:
                        cat = 'DeltaNet linear attn'
                    elif 'sdpa' in src or 'scaled_dot' in src:
                        cat = 'GQA SDPA'
                    elif 'modeling_qwen3_5' in src and ('attention' in src or 'attn' in src):
                        cat = 'Attention (projections)'
                    elif 'mlp' in src or 'gate' in src or 'up_proj' in src or 'down_proj' in src:
                        cat = 'MLP'
                    elif 'norm' in src or 'rms' in src:
                        cat = 'RMSNorm'
                    elif 'lora' in src:
                        cat = 'LoRA'
                    elif 'embed' in src:
                        cat = 'Embedding'
                    elif 'unsloth' in src:
                        cat = 'Unsloth patches'
                    else:
                        cat = 'Other'
                    cats[cat]['bytes'] += info['bytes']
                    cats[cat]['count'] += 1
                    cats[cat]['tensors'].append(info)

                total_saved = sum(c['bytes'] for c in cats.values())
                total_fp32 = sum(i['bytes'] for i in _saved_tensor_info if 'float32' in i['dtype'])
                total_bf16 = sum(i['bytes'] for i in _saved_tensor_info if 'bfloat16' in i['dtype'])
                total_other = total_saved - total_fp32 - total_bf16

                print(f"\n  === SAVED TENSOR PROFILE (step 0, seq_len={seq_len}) ===")
                print(f"  Total: {len(_saved_tensor_info)} tensors, {total_saved/1024**2:.0f} MB ({total_saved/1024/seq_len:.1f} KB/tok)")
                print(f"  Dtypes: fp32={total_fp32/1024**2:.0f}MB ({100*total_fp32/max(total_saved,1):.0f}%), "
                      f"bf16={total_bf16/1024**2:.0f}MB ({100*total_bf16/max(total_saved,1):.0f}%), "
                      f"other={total_other/1024**2:.0f}MB ({100*total_other/max(total_saved,1):.0f}%)")
                print(f"\n  {'Category':<30s} {'MB':>8s} {'KB/tok':>8s} {'%':>6s} {'#':>5s}")
                print(f"  {'-'*61}")
                for cat, data in sorted(cats.items(), key=lambda x: -x[1]['bytes']):
                    mb = data['bytes'] / 1024**2
                    kbtok = data['bytes'] / 1024 / seq_len
                    pct = 100 * data['bytes'] / max(total_saved, 1)
                    print(f"  {cat:<30s} {mb:8.1f} {kbtok:8.1f} {pct:5.1f}% {data['count']:5d}")

                print(f"\n  Top 20 individual tensors:")
                print(f"  {'Shape':<35s} {'dtype':<15s} {'MB':>8s} {'Source':<50s}")
                print(f"  {'-'*112}")
                for info in sorted(_saved_tensor_info, key=lambda x: -x['bytes'])[:20]:
                    mb = info['bytes'] / 1024**2
                    print(f"  {str(info['shape']):<35s} {info['dtype']:<15s} {mb:8.1f} {info['source'][:50]:<50s}")

            # Report weight offloading transfer costs
            if _n_gpus > 1 and _hook_transfer_log:
                # Find max spike during transfers
                max_spike = 0
                for before, mid, after in _hook_transfer_log:
                    spike = max(mid, after) - before
                    max_spike = max(max_spike, spike)
                print(f"  Weight transfers: {len(_hook_transfer_log)} moves, max spike: {max_spike:.0f}MB")
                # Show first 5 and last 5
                for i, (before, mid, after) in enumerate(_hook_transfer_log[:3]):
                    print(f"    xfer[{i}]: {before:.0f} → {mid:.0f} (offload prev) → {after:.0f}MB (load next)")
                if len(_hook_transfer_log) > 6:
                    print(f"    ... ({len(_hook_transfer_log)-6} more) ...")
                for i, (before, mid, after) in enumerate(_hook_transfer_log[-3:]):
                    idx = len(_hook_transfer_log) - 3 + i
                    print(f"    xfer[{idx}]: {before:.0f} → {mid:.0f} → {after:.0f}MB")
                _hook_transfer_log.clear()
                if step >= 3:
                    _hook_log_enabled[0] = False  # Stop logging after first few steps

            # Deep profiling: backward memory per layer + live tensor snapshot
            if args.profile_deep and step == 0:
                # 1. Backward hooks log
                if _bwd_hook_log:
                    print(f"\n  Backward pass per-layer memory:")
                    prev_bwd = None
                    for name, mem in _bwd_hook_log:
                        if prev_bwd is not None:
                            delta = mem - prev_bwd
                            print(f"    {name:<20s} {mem:>8.0f}MB  ({delta:+.0f}MB)")
                        else:
                            print(f"    {name:<20s} {mem:>8.0f}MB")
                        prev_bwd = mem
                    _bwd_hook_log.clear()

                # 2. Snapshot all live CUDA tensors RIGHT NOW
                # (post-step, still has gradients + optimizer state + any leaked activations)
                _snapshot_all_cuda_tensors()

                # 3. Check Unsloth offloading internals
                try:
                    from unsloth_zoo.gradient_checkpointing import CPU_BUFFERS, GPU_BUFFERS
                    cpu_total = sum(b.nelement() * b.element_size() for b in CPU_BUFFERS if hasattr(b, 'nelement')) if CPU_BUFFERS else 0
                    gpu_total = sum(b.nelement() * b.element_size() for b in GPU_BUFFERS if hasattr(b, 'nelement')) if GPU_BUFFERS else 0
                    print(f"\n  Unsloth offload buffers:")
                    print(f"    CPU buffers: {len(CPU_BUFFERS) if CPU_BUFFERS else 0} ({cpu_total/1024**2:.1f} MB)")
                    print(f"    GPU buffers: {len(GPU_BUFFERS) if GPU_BUFFERS else 0} ({gpu_total/1024**2:.1f} MB)")
                except Exception as e:
                    print(f"\n  Unsloth offload buffers: could not inspect ({e})")

                # 4. Memory allocator stats
                stats = torch.cuda.memory_stats()
                print(f"\n  CUDA allocator stats:")
                print(f"    Active allocations: {stats.get('active.all.current', '?')}")
                print(f"    Active bytes:       {stats.get('active_bytes.all.current', 0)/1024**2:.0f} MB")
                print(f"    Reserved bytes:     {stats.get('reserved_bytes.all.current', 0)/1024**2:.0f} MB")
                print(f"    Peak active bytes:  {stats.get('active_bytes.all.peak', 0)/1024**2:.0f} MB")
                print(f"    Num alloc retries:  {stats.get('num_alloc_retries', '?')}")
                print(f"    Num OOM:            {stats.get('num_ooms', '?')}")

            try:
                import wandb as _wb
                if _wb.run is not None:
                    _wb.log({"seq_len": seq_len, "peak_vram_mb": step_peak}, commit=False)
            except Exception:
                pass
            print()
            return loss
        else:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loss = original_training_step(model, inputs, num_items_in_batch)
            step_peak = peak_mb()
            try:
                import wandb as _wb
                if _wb.run is not None:
                    _wb.log({"seq_len": seq_len, "peak_vram_mb": step_peak}, commit=False)
            except Exception:
                pass
            return loss

    trainer.training_step = profiled_training_step

    result = trainer.train()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Loss:    {result.training_loss:.4f}")
    print(f"  Runtime: {result.metrics['train_runtime']:.0f}s")
    print(f"  Samples/sec: {result.metrics['train_samples_per_second']:.2f}")

    # ─── Save ───
    lora_dir = os.path.join(args.output, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"  LoRA adapter saved to: {lora_dir}")

    if args.export_gguf:
        print("Exporting GGUF (Q4_K_M)...")
        gguf_dir = os.path.join(args.output, "gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(f"  GGUF saved to: {gguf_dir}")

    if args.export_merged:
        print("Exporting merged 16-bit model...")
        merged_dir = os.path.join(args.output, "merged")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"  Merged model saved to: {merged_dir}")

    print(f"\nDone! Your Claude Code agent LoRA is at: {lora_dir}")


if __name__ == "__main__":
    main()
