# Changelog

## 2026-03-06 — Multi-GPU Weight Offloading

### Concept
Use GPU 1 (RTX 4060 8GB) as pure weight storage. All 32 decoder layers offloaded to GPU 1,
shuttled to GPU 0 (RTX 5060 Ti 16GB) on-demand via pre_forward hooks. All computation
(forward, backward, optimizer) stays on GPU 0.

### Results
- GPU 0 base allocation: 6.15 GB → **2.35 GB** (freed 3.8 GB!)
- GPU 1 usage: 3.81 GB (32 layers × ~115 MB packed 4-bit)
- 4K training verified: loss decreasing, grad_norm stable

### Key Technical Details

**BNB Params4bit device movement:**
- `module.to(device)` works but misses `param.quant_state`
- Must call `qs.to(device)` separately — it's IN-PLACE (returns None!)
- `param.quant_state = qs.to(device)` → sets quant_state=None → CUBLAS crash

**Hook design — `_last_loaded` pattern:**
- Only `pre_forward_hook`, NO post_hook
- Post_hook fires after recomputed forward but BEFORE backward needs weights = crash
- `full_backward_pre_hook` doesn't fire with reentrant gradient checkpointing
- Solution: when loading layer N, first offload layer N-1

**What failed:**
1. Model parallelism (device_map split): backward workspace ~4 GB needed per GPU independently
   → GPU 1 (8 GB) too small for long-sequence backward
2. accelerate AlignDevicesHook: `Params4bit.__new__()` rejects `_is_hf_initialized` kwarg
3. Raw `param.data` swap: BNB packed format + quant_state mismatch → shape errors

### Memory Scaling — CORRECTION: LINEAR not quadratic!
Previous analysis (pre-CPU-offload-fix) showed quadratic scaling. **This was wrong for the
fixed version.** Post-fix data points show LINEAR scaling:

| Seq Len | peak_delta | Marginal from 15K |
|---------|-----------|-------------------|
| 15K | 4068 MB | — (backward floor) |
| 26K | 6288 MB | 202 KB/tok |
| 32K | 7658 MB | 211 KB/tok |

Marginal cost is constant (~211 KB/tok) → **linear**, not quadratic.
Formula: `peak_delta = 4068 + 0.211 * (n - 15000)` MB

The old quadratic fit (`peak_MB = 10500 + c*n²`) was measured BEFORE the CPU offloading fix,
when all 32 layer checkpoint inputs accumulated on GPU (128 MB each). After the fix, Unsloth's
CPU offloading prevents this accumulation, making growth linear.

### Theoretical Max Context with Offloading
Available VRAM on GPU 0: 16.0 - 2.35 (base) - 2.2 (CUDA overhead) - 0.115 (1 layer) = 11.3 GB
Max: `4068 + 0.211*(n-15000) = 11335` → **n ≈ 49,400 tokens**

| Seq Len | Peak Delta | Total | Headroom |
|---------|-----------|-------|----------|
| 32K | 7.66 GB | 12.3 GB | 3.7 GB |
| 40K | 9.34 GB | 14.0 GB | 2.0 GB |
| 48K | 11.0 GB | 15.7 GB | 0.3 GB |

### Current Issue: OOM at 36K
Despite ~3.6 GB theoretical headroom at 36K, training OOMs. Suspected cause:
`module.to()` GPU-to-GPU transfer may create transient bf16 copies during dequant/requant.
Added memory instrumentation to hooks (logs GPU 0 memory before/mid/after each transfer).
Also added `--offload-layers N` flag for partial offloading to reduce transfer frequency.

### Next Steps
1. Restart Docker Desktop (crashed during 32K test)
2. Test 32K baseline with new instrumentation
3. If transfer spikes confirmed, try partial offloading or raw tensor copy approach
4. Push to 48K target

---

## 2026-03-05 — VRAM Optimization Sprint

### Working Optimizations (in train.py)
1. **Apple CCE (Cut Cross-Entropy) enabled**
   - `UNSLOTH_ENABLE_CCE=1` must be set AFTER `import unsloth` (it resets to 0 during import)
   - Eliminates materialization of full logits tensor (seq_len x 248K vocab x 2 bytes)
   - Uses `fused_linear_cross_entropy` instead of `unsloth_fused_ce_loss`

2. **embed_tokens quantized to int8**
   - Per-row scale factor, dequant on the fly during forward
   - Saves ~1.02 GB (2.03 GB -> 1.02 GB)
   - Must clone+del original weight to avoid ghost allocation

3. **Vision encoder removal**
   - Qwen3.5-9B loads as VL model (ConditionalGeneration) with unused vision encoder
   - Set visual/multi_modal_projector to None after loading, saves ~0.3 GB
   - Also reduces "other" param category

4. **empty_cache() between training steps**
   - Prevents allocator cache bloat (was growing to 5.7 GB and never releasing)

5. **Single-threaded model loading**
   - Prevents VRAM spikes during weight loading on small GPUs

### Failed / Reverted Optimizations
1. **Int8 saved_tensors_hooks** — REMOVED
   - Quantizes autograd saved tensors to int8 during backward pass
   - Crashes with Unsloth's chunked CE loss (uses torch.func.grad_and_value)
   - Even with CCE (no grad_and_value), causes grad_norm explosion (1393 vs 4)
   - Per-tensor int8 is too lossy for gradient computation

2. **fp8 lm_head via Python-level cast** — REVERTED (replaced by native Triton patches)
   - Python-level c.to(bf16) creates 2 GB temporary copy during loss — net negative
   - Replaced by source-patching CCE Triton kernels (see below)

### fp8 lm_head — Native Triton Patch (WORKING)
The breakthrough: patch CCE source files BEFORE import so Triton JIT compiles with fp8 support.
Three files patched at startup via `_patch_cce_for_fp8()`:
1. `indexed_dot.py`: add `.to(e.dtype)` on `tl.load(c_ptrs)` (2 lines)
2. `cce_backward.py`: remove bf16/fp16/fp32 dtype assert (Triton kernel already casts)
3. `loss_utils.py`: remove `hidden_states.to(lm_weight.dtype)` (would cast to fp8)

Zero-copy: Triton loads fp8 from VRAM and casts to bf16 in GPU registers. No temporary tensors.

Note: lm_head fp8 conversion must happen AFTER SFTTrainer init because Unsloth's
`fix_untrained_tokens` needs bf16 for `torch.amax`. Ghost allocation clears before training starts.

### Memory Progress
| Run | Optimizations | Base Alloc | Peak (12K) | grad_norm |
|-----|--------------|------------|------------|-----------|
| Baseline | none | 8.19 GB | 13.0 GB | ~16 |
| +CCE +embed_int8 (leak) | CCE, embed | 9.20 GB | 13.3 GB | 4.2 |
| +ghost fix | CCE, embed, fixed | 7.17 GB | 11.2 GB | 4.1 |
| +fp8 lm_head (native) | all optimizations | **6.15 GB** | **10.2 GB** | 4.2 |

**Total savings: 2.04 GB base, 2.8 GB peak**

### Key Discoveries
- `UNSLOTH_ENABLE_CCE` is overridden to "0" by unsloth during import — set AFTER import
- Unsloth reports 5.8B params (language model scope) vs 9.5B (full VL model)
- Memory scales ~linearly (n^0.96) with context length, ~587 KB/token
- CUDA allocator cache grows permanently after first long sequence
- The "other" 1.1 GB category includes the unused vision encoder
- CCE's `fused_linear_cross_entropy` does `hidden_states.to(lm_weight.dtype)` — breaks fp8
- CCE's `indexed_neg_dot` Triton kernel missing `.to(e.dtype)` cast (unlike main kernel)
- CCE backward kernel has bf16 assert but Triton already handles fp8 via `.to(e.dtype)`
- Ghost allocations from weight replacement: old tensor freed from Python but CUDA allocator
  holds the block. Frees eventually (before training starts) but not immediately after empty_cache().

### Context Length Scaling (post-optimization)
- Scaling is **QUADRATIC** due to 8 GQA attention layers (O(n²) attention matrices)
- Fit: `peak_MB = 10500 + c*n²` where c ≈ 0.0000197 MB/tok²
- RMSE: 102 MB (quadratic) vs 474 MB (linear) — quadratic fits 4.6x better
- Marginal cost per token doubles from 8K (214 KB) to 16K (544 KB)
- Max theoretical fit: ~20,309 tokens on 16 GB

| Seq Len | Peak VRAM | Marginal/tok | Status |
|---------|-----------|-------------|--------|
| 4K | 10.1 GB | 49 KB | Safe |
| 8K | 10.6 GB | 214 KB | Safe |
| 12K | 11.8 GB | 379 KB | Safe |
| 16K | 13.6 GB | 544 KB | Running |
| 20K | 16.0 GB | 709 KB | Borderline |
| 24K | 19.1 GB | 875 KB | OOM |

- 16K training run active with wandb logging (1004 examples, 45% of dataset)

### Dataset Splitting (2026-03-05)
- Added `split_dataset.py`: splits at Claude Code auto-compression boundaries
- Marker: "This session is being continued from a previous conversation that ran out of context"
- 2318 → 2740 conversations (+422 from 130 multi-session traces)
- Each segment gets original system prompt prepended (realistic to how Claude Code works)
- docker-compose now mounts `claude-traces-split.jsonl`
- Segments are still large (median ~28K tokens) — splitting helps at 32K+ context

### Research: KV Cache Quantization (2026-03-05)
**Finding: There is NO KV cache during training.** `use_cache=False` by default.
The ~250 KB/token cost is **saved activations** for backpropagation, not KV cache.

Architecture breakdown (Qwen3.5-9B):
- 8 GQA layers: KV projections + O(n²) attention matrices saved for backward
- 24 DeltaNet layers: O(n) with fixed ~12 MB state (not the bottleneck)
- KV cache would only be 32 KB/token if it existed — activations are ~8x more

**Activation compression approaches (GACT, COAT, CompAct):**
- `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` can intercept saved tensors
- Previous global attempt crashed (torch.func.grad_and_value incompatibility)
- Surgical approach: wrap only 8 GQA attention forwards in int8 hooks
- Expected savings: ~1.7 GB (int8) to ~2.6 GB (int4), enabling 20-28K context

### RMSNorm bf16 Patch (2026-03-05)
- Monkey-patched `Qwen3_5RMSNorm.forward` to stay in bf16 (no fp32 upcasts)
- 81 instances patched across the model
- Per-token forward: 350 KB → 302 KB (-14%)
- Per-token peak at 15K: 544 KB → 474 KB (-13%)
- grad_norm stable: 1.25–1.69 (no explosion from bf16 precision loss)
- fp32 in saved tensors: 40% → 0.4% (27 MB out of 7259 MB)

### Saved Tensor Deep Profile (2026-03-05)
**Added `--profile-tensors` flag** to train.py. Uses `saved_tensors_hooks` on step 0 to trace every tensor saved for backward pass.

Profile at seq_len=1635, LoRA rank=32 (556 tensors, 7259 MB total):

| Category | MB | KB/tok | % of total | # tensors | What |
|----------|-----|--------|-----------|-----------|------|
| **MLP intermediates** | 3679 | 2304 | **50.7%** | 96 | `(1, seq, 12288)` bf16, gate/up outputs, 3/layer × 32 layers |
| **LoRA inputs** | 2149 | 1346 | **29.6%** | 384 | `(seq, dim)` bf16, saved for LoRA backward, 12/layer × 32 layers |
| **CCE lm_head weight** | 970 | — | **13.4%** | 1 | `(248320, 4096)` fp8, fixed cost (not per-token) |
| **Misc/small** | 461 | — | **6.3%** | 75 | Scalars, small buffers, etc |

Key findings:
- **80% of all saved memory = MLP + LoRA**, both scale linearly with seq_len
- MLP saves 3 tensors per layer: gate_proj output, up_proj output, SiLU activation (all `seq × 12288` bf16)
- LoRA saves input to each adapter for backward (q/k/v/o/gate/up/down projections)
- At 16K tokens, MLP alone would be ~6 GB, LoRA ~3.5 GB
- GQA attention / DeltaNet / RMSNorm are NOT the bottleneck (invisible in profile)
- Source: `unsloth_compiled_module_qwen3_5.py:873 Qwen3_5MLP_` and `Linear4bit_peft_forward.py:47 lora_forward`
- Dtypes: fp32=27MB (0.4%), bf16=6262MB (86%), fp8=970MB (13%)

### Comparison Test Results (2026-03-05)
Tested 5 configs at seq_len=15101, LoRA rank=32, alpha=64:

| Config | peak_delta | KB/tok | grad_norm | Verdict |
|--------|-----------|--------|-----------|---------|
| baseline | 6992 MB | 474.2 | 1.69 | Reference |
| fp8 MLP activations | 7345 MB | 498.0 | 1.68 | **+5% WORSE** |
| proj-256 | 7200 MB | 488.2 | inf | **BROKEN** |
| fp8 + proj-256 | 7197 MB | 488.0 | inf | **BROKEN** |
| **LoRA q_proj,v_proj** | **6863 MB** | **465.4** | **1.23** | **-1.8% WINNER** |

**Key discovery: Unsloth's TiledMLP already handles MLP activation memory!**
- TiledMLP chunks the sequence and RECOMPUTES MLP forward during backward
- MLP intermediates are never saved long-term — only one chunk at a time
- Our fp8/projection patches added overhead on top of recomputation = net negative
- The profiler showed 7259 MB "total saved" but TiledMLP means only 1 chunk is live
- Random projection backward (SiLU derivative with reconstructed inputs) is numerically unstable

**Only winner: reducing LoRA target modules** from 7 to 2 (q_proj, v_proj)
- 430 MB less base allocation (fewer LoRA parameters)
- 129 MB less peak memory
- Stable grad_norm (1.23 vs 1.69)

### Next Steps
- Merge --lora-targets feature into master
- Investigate sub-layer gradient checkpointing (separate attn + MLP checkpoints)
- Profile what's actually live during backward (not total saved across all recomputations)
- The gradient checkpoint floor (4068 MB) may be the real remaining target
