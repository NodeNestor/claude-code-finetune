#!/usr/bin/env python3
"""Claude Code Fine-tune — unified entry point.

Two modes:
  1. Single LoRA  — one adapter trained on all your Claude Code traces
  2. Expert swarm  — multiple specialist LoRAs + a router that picks the right one

Both start from the same place: your Claude Code conversation history.

Usage:
    # ─── Generate dataset (both modes start here) ───
    python finetune.py dataset                     # auto-detect logs, convert + split
    python finetune.py dataset --input ~/other/logs

    # ─── Single LoRA mode (default) ───
    python finetune.py train                       # train one LoRA on everything
    python finetune.py train --seq-length 16384    # custom context length

    # ─── Expert swarm mode ───
    python finetune.py experts                     # analyze → split → train all + router

    # ─── Just analyze (see what experts you'd get) ───
    python finetune.py experts --analyze-only
"""

import argparse
import os
import subprocess
import sys


def run(cmd, check=True):
    """Run a command, streaming output."""
    print(f"\n{'─'*60}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}\n")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0


def cmd_dataset(args):
    """Generate training dataset from Claude Code logs."""
    python = sys.executable

    # Step 1: Convert traces
    convert_cmd = [python, "claude-trace-converter.py"]
    if args.input:
        convert_cmd += ["--input", args.input]
    convert_cmd += ["--output", args.raw_output]
    if args.no_thinking:
        convert_cmd.append("--no-thinking")
    if args.include_subagents:
        convert_cmd.append("--include-subagents")

    print("Step 1/2: Converting Claude Code traces...")
    if not run(convert_cmd, check=False):
        print("\nERROR: Trace conversion failed.")
        return

    # Step 2: Split at compression boundaries
    print("\nStep 2/2: Splitting at compression boundaries...")
    split_cmd = [python, "split_dataset.py",
                 "--input", args.raw_output,
                 "--output", args.output]
    if not run(split_cmd, check=False):
        print("\nWARNING: Split failed, using unsplit dataset.")
        # Copy raw as output
        import shutil
        shutil.copy2(args.raw_output, args.output)

    print(f"\n{'='*60}")
    print(f"Dataset ready: {args.output}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  Single LoRA:   python finetune.py train")
    print(f"  Expert swarm:  python finetune.py experts")
    print(f"  Just analyze:  python finetune.py experts --analyze-only")


def cmd_train(args):
    """Train a single LoRA on all traces."""
    python = sys.executable

    if not os.path.isfile(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        print(f"Run 'python finetune.py dataset' first.")
        return

    cmd = [python, "train.py", "--dataset", args.dataset]

    # Pass through all training args
    if args.model:
        cmd += ["--model", args.model]
    if args.seq_length:
        cmd += ["--seq-length", str(args.seq_length)]
    if args.lora_rank:
        cmd += ["--lora-rank", str(args.lora_rank)]
    if args.lora_alpha:
        cmd += ["--lora-alpha", str(args.lora_alpha)]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.lr:
        cmd += ["--lr", str(args.lr)]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.grad_accum:
        cmd += ["--grad-accum", str(args.grad_accum)]
    if args.output:
        cmd += ["--output", args.output]
    if args.export_gguf:
        cmd.append("--export-gguf")
    if args.offload_layers is not None:
        cmd += ["--offload-layers", str(args.offload_layers)]

    run(cmd)


def cmd_experts(args):
    """Train expert LoRA swarm: split → train each expert → train router."""
    python = sys.executable

    if not os.path.isfile(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        print(f"Run 'python finetune.py dataset' first.")
        return

    # Step 1: Analyze
    print("Step 1: Analyzing dataset by expert category...")
    run([python, "train_router.py", "analyze", "--dataset", args.dataset, "-v"])

    if args.analyze_only:
        print(f"\n  Done. To proceed with training:")
        print(f"    python finetune.py experts --dataset {args.dataset}")
        return

    # Step 2: Split into per-expert datasets
    print("\nStep 2: Splitting into per-expert datasets...")
    expert_dir = args.expert_dir
    run([python, "train_router.py", "split",
         "--dataset", args.dataset,
         "--output-dir", expert_dir])

    # Step 3: Train each expert LoRA
    expert_files = sorted(f for f in os.listdir(expert_dir) if f.endswith(".jsonl"))
    if not expert_files:
        print("ERROR: No expert datasets generated.")
        return

    # Filter by minimum size
    expert_files_filtered = []
    for f in expert_files:
        path = os.path.join(expert_dir, f)
        n_lines = sum(1 for _ in open(path, encoding="utf-8"))
        if n_lines >= args.min_examples:
            expert_files_filtered.append(f)
        else:
            name = f.replace(".jsonl", "")
            print(f"  Skipping {name} ({n_lines} examples < {args.min_examples} minimum)")

    if not expert_files_filtered:
        print("ERROR: No expert datasets have enough examples.")
        print(f"  Lower --min-examples (currently {args.min_examples}) or add more traces.")
        return

    print(f"\nStep 3: Training {len(expert_files_filtered)} expert LoRAs...")
    checkpoints_dir = args.output
    os.makedirs(checkpoints_dir, exist_ok=True)

    for i, fname in enumerate(expert_files_filtered):
        name = fname.replace(".jsonl", "")
        dataset_path = os.path.join(expert_dir, fname)
        output_path = os.path.join(checkpoints_dir, f"expert-{name}")

        if os.path.isdir(os.path.join(output_path, "lora")):
            print(f"\n  [{i+1}/{len(expert_files_filtered)}] Skipping {name} (already trained)")
            continue

        print(f"\n  [{i+1}/{len(expert_files_filtered)}] Training expert: {name}")
        cmd = [python, "train.py",
               "--dataset", dataset_path,
               "--output", output_path,
               "--model", args.model,
               "--seq-length", str(args.seq_length),
               "--lora-rank", str(args.lora_rank),
               "--epochs", str(args.expert_epochs),
               "--lr", str(args.lr)]
        if args.offload_layers is not None:
            cmd += ["--offload-layers", str(args.offload_layers)]

        success = run(cmd, check=False)
        if not success:
            print(f"  WARNING: Training failed for {name}, continuing...")

    # Step 4: Train router
    print(f"\nStep 4: Training router LoRA...")
    router_cmd = [python, "train_router.py", "train",
                  "--dataset", args.dataset,
                  "--model", args.model,
                  "--output", os.path.join(checkpoints_dir, "router"),
                  "--lora-rank", str(args.router_rank),
                  "--min-count", str(args.min_examples)]
    run(router_cmd, check=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"Expert swarm training complete!")
    print(f"{'='*60}")
    print(f"\n  Checkpoints: {checkpoints_dir}/")
    trained = [f.replace(".jsonl", "") for f in expert_files_filtered]
    for name in trained:
        lora_path = os.path.join(checkpoints_dir, f"expert-{name}", "lora")
        status = "OK" if os.path.isdir(lora_path) else "FAILED"
        print(f"    expert-{name:<15s} [{status}]")
    router_path = os.path.join(checkpoints_dir, "router", "final")
    status = "OK" if os.path.isdir(router_path) else "FAILED"
    print(f"    router{'':<16s} [{status}]")
    print(f"\n  To serve with vLLM multi-LoRA, load all adapters from {checkpoints_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Fine-tune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. python finetune.py dataset                  # generate training data
  2. python finetune.py train                    # single LoRA (simple)
     OR
  2. python finetune.py experts                  # expert swarm (multi-LoRA)

The 'dataset' step converts your Claude Code conversation logs into training
format. Run it once, then train as many times as you want.
""",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ─── dataset ───
    ds = sub.add_parser("dataset", help="Generate training dataset from Claude Code logs")
    ds.add_argument("--input", default=None,
                    help="Claude Code logs directory (default: auto-detect ~/.claude/projects)")
    ds.add_argument("--output", default="claude-traces-split.jsonl",
                    help="Final output dataset (default: claude-traces-split.jsonl)")
    ds.add_argument("--raw-output", default="claude-traces-dataset.jsonl",
                    help="Intermediate unsplit output")
    ds.add_argument("--no-thinking", action="store_true",
                    help="Exclude thinking/reasoning blocks")
    ds.add_argument("--include-subagents", action="store_true",
                    help="Include sub-agent conversations")

    # ─── train ───
    tr = sub.add_parser("train", help="Train a single LoRA on all traces")
    tr.add_argument("--dataset", default="claude-traces-split.jsonl",
                    help="Training dataset (default: claude-traces-split.jsonl)")
    tr.add_argument("--model", default=None, help="Base model")
    tr.add_argument("--output", default=None, help="Output directory")
    tr.add_argument("--seq-length", type=int, default=None, dest="seq_length")
    tr.add_argument("--lora-rank", type=int, default=None, dest="lora_rank")
    tr.add_argument("--lora-alpha", type=int, default=None, dest="lora_alpha")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--lr", type=float, default=None)
    tr.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    tr.add_argument("--grad-accum", type=int, default=None, dest="grad_accum")
    tr.add_argument("--export-gguf", action="store_true", dest="export_gguf")
    tr.add_argument("--offload-layers", type=int, default=None, dest="offload_layers")

    # ─── experts ───
    ex = sub.add_parser("experts", help="Train expert LoRA swarm + router")
    ex.add_argument("--dataset", default="claude-traces-split.jsonl",
                    help="Training dataset (default: claude-traces-split.jsonl)")
    ex.add_argument("--model", default="unsloth/Qwen3.5-0.8B",
                    help="Base model (default: Qwen3.5-0.8B — small is better for experts)")
    ex.add_argument("--output", default="checkpoints",
                    help="Output directory for all expert + router checkpoints")
    ex.add_argument("--expert-dir", default="expert-datasets", dest="expert_dir",
                    help="Directory for per-expert split datasets")
    ex.add_argument("--seq-length", type=int, default=8192, dest="seq_length",
                    help="Seq length for expert training (default: 8192)")
    ex.add_argument("--lora-rank", type=int, default=32, dest="lora_rank",
                    help="LoRA rank for experts (default: 32)")
    ex.add_argument("--router-rank", type=int, default=32, dest="router_rank",
                    help="LoRA rank for router (default: 32)")
    ex.add_argument("--expert-epochs", type=int, default=1, dest="expert_epochs",
                    help="Epochs per expert (default: 1)")
    ex.add_argument("--lr", type=float, default=2e-4)
    ex.add_argument("--min-examples", type=int, default=10, dest="min_examples",
                    help="Skip experts with fewer examples (default: 10)")
    ex.add_argument("--analyze-only", action="store_true", dest="analyze_only",
                    help="Only analyze — show distribution, don't train")
    ex.add_argument("--offload-layers", type=int, default=None, dest="offload_layers")

    args = parser.parse_args()

    if args.command == "dataset":
        cmd_dataset(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "experts":
        cmd_experts(args)


if __name__ == "__main__":
    main()
