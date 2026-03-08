"""Train a router LoRA that selects expert LoRAs per Claude Code task.

Classifies Claude Code conversations by what actually happened — tool usage
patterns, thinking complexity, output type — not keyword guessing. Each
conversation gets labeled automatically, then we train a 1-token sequence
classifier that can dispatch to the right expert LoRA at inference time.

Works directly with the output of claude-trace-converter.py. No separate
data generation step needed.

Expert categories (Claude Code specific):
  - code_edit:    Heavy Edit/Write usage — implementing features, writing code
  - exploration:  Heavy Grep/Glob/Read — searching, understanding codebases
  - debugging:    Bash with test/error output + Edit fixes — finding and fixing bugs
  - planning:     Long thinking blocks, minimal tools — architecture, design decisions
  - commands:     Heavy Bash — running builds, git, installs, shell tasks
  - frontend:     Edit/Write on .tsx/.css/.html — UI/component work
  - config:       Edit on config files — Docker, CI/CD, yaml, toml, package.json
  - testing:      Write/Edit on test files + Bash running tests
  - refactor:     Many Edits on existing files, few Writes — restructuring code
  - compact:      Short conversations, quick answers — one-shot questions

Architecture:
  - Input:  first user message of a conversation (truncated to --max-length tokens)
  - Output: single classification token → expert LoRA name
  - Model:  same Qwen3.5 base as your experts (e.g. 0.8B for swarms, 9B for full)
  - Latency: <10ms per classification (single forward pass)

Usage:
    # One-shot: classify dataset + train router (just give it your traces)
    python train_router.py train --dataset claude-traces-split.jsonl

    # Inspect what the classifier sees before training
    python train_router.py analyze --dataset claude-traces-split.jsonl

    # Evaluate a trained router
    python train_router.py eval --dataset claude-traces-split.jsonl \\
        --model unsloth/Qwen3.5-0.8B --adapter checkpoints/router-lora

    # Split traces into per-expert datasets (for training expert LoRAs)
    python train_router.py split --dataset claude-traces-split.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ─── Claude Code expert categories ───
# Classified by tool usage patterns + file types + thinking patterns.
# These are what Claude Code *actually does*, not what the user asks for.

EXPERT_CATEGORIES = {
    "code_edit": "Implementing features, writing new code (heavy Edit/Write on source files)",
    "exploration": "Searching and reading codebases (heavy Grep/Glob/Read, minimal edits)",
    "debugging": "Finding and fixing bugs (error patterns + targeted edits)",
    "planning": "Architecture, design, reasoning (long thinking, minimal tool use)",
    "commands": "Shell tasks — builds, git, installs, deploys (heavy Bash)",
    "frontend": "UI/component work (Edit/Write on .tsx/.jsx/.css/.html/.vue/.svelte)",
    "config": "Configuration — Docker, CI/CD, yaml, toml, package.json, env files",
    "testing": "Writing/running tests (test file edits + Bash test runners)",
    "refactor": "Restructuring existing code (many Edits, few new files)",
    "compact": "Quick one-shot answers, short conversations (minimal tool use)",
}

# File extension patterns for category detection
FRONTEND_EXTS = {".tsx", ".jsx", ".css", ".scss", ".less", ".html", ".vue", ".svelte",
                 ".astro", ".ejs", ".hbs", ".pug"}
CONFIG_EXTS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".json", ".xml"}
CONFIG_NAMES = {"dockerfile", "docker-compose", "makefile", "cmakelists", ".gitignore",
                ".eslintrc", ".prettierrc", "tsconfig", "package.json", "pyproject.toml",
                "setup.py", "setup.cfg", "cargo.toml", "go.mod", "pom.xml", "build.gradle"}
TEST_PATTERNS = [r"test[_/]", r"_test\.", r"\.test\.", r"\.spec\.", r"tests/", r"__tests__/",
                 r"spec/", r"pytest", r"jest", r"mocha", r"vitest"]


def extract_tool_calls(conversation):
    """Extract tool usage patterns from a conversation's gpt turns."""
    tools = Counter()       # tool_name -> call count
    file_paths = []         # files touched by Edit/Write/Read
    bash_commands = []      # Bash command strings
    thinking_chars = 0      # total chars in <think> blocks
    total_gpt_chars = 0     # total chars in gpt turns
    edit_count = 0
    write_count = 0

    for turn in conversation:
        role = turn.get("from", "")
        text = turn.get("value", "")

        if role == "gpt":
            total_gpt_chars += len(text)

            # Count thinking blocks
            for match in re.finditer(r'<think>(.*?)</think>', text, re.DOTALL):
                thinking_chars += len(match.group(1))

            # Parse tool calls
            for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
                try:
                    call = json.loads(match.group(1))
                    name = call.get("name", "")
                    inp = call.get("input", {})
                    tools[name] += 1

                    if name in ("Edit", "Write"):
                        fp = inp.get("file_path", "")
                        if fp:
                            file_paths.append(fp)
                        if name == "Edit":
                            edit_count += 1
                        else:
                            write_count += 1
                    elif name == "Read":
                        fp = inp.get("file_path", "")
                        if fp:
                            file_paths.append(fp)
                    elif name == "Bash":
                        cmd = inp.get("command", "")
                        if cmd:
                            bash_commands.append(cmd)
                    elif name in ("Grep", "Glob"):
                        pass  # counted in tools dict
                except (json.JSONDecodeError, KeyError):
                    continue

        elif role == "human":
            # Also check tool results for error patterns
            if "<tool_response>" in text and ("error" in text.lower() or "traceback" in text.lower()
                                               or "failed" in text.lower() or "exception" in text.lower()):
                tools["_error_result"] += 1

    return {
        "tools": tools,
        "file_paths": file_paths,
        "bash_commands": bash_commands,
        "thinking_chars": thinking_chars,
        "total_gpt_chars": total_gpt_chars,
        "edit_count": edit_count,
        "write_count": write_count,
    }


def classify_conversation(conversation):
    """Classify a Claude Code conversation into an expert category.

    Uses tool usage patterns, file types, and conversation structure.
    Returns (category, confidence_reason).
    """
    convs = conversation.get("conversations", [])
    if len(convs) < 3:
        return "compact", "too short"

    info = extract_tool_calls(convs)
    tools = info["tools"]
    file_paths = info["file_paths"]
    bash_cmds = info["bash_commands"]
    total_tool_calls = sum(tools.values())
    n_turns = len(convs)

    # ─── Detect file types touched ───
    frontend_files = sum(1 for f in file_paths
                         if any(f.lower().endswith(ext) for ext in FRONTEND_EXTS))
    config_files = sum(1 for f in file_paths
                       if any(f.lower().endswith(ext) for ext in CONFIG_EXTS)
                       or any(n in f.lower() for n in CONFIG_NAMES))
    test_files = sum(1 for f in file_paths
                     if any(re.search(p, f.lower()) for p in TEST_PATTERNS))

    # ─── Detect bash patterns ───
    test_commands = sum(1 for c in bash_cmds
                        if any(kw in c.lower() for kw in
                               ["pytest", "jest", "npm test", "cargo test", "go test",
                                "vitest", "mocha", "rspec", "unittest", "make test"]))
    git_commands = sum(1 for c in bash_cmds if c.strip().startswith("git "))
    build_commands = sum(1 for c in bash_cmds
                         if any(kw in c.lower() for kw in
                                ["npm ", "yarn ", "pip ", "cargo ", "make", "docker ",
                                 "go build", "mvn ", "gradle"]))

    # ─── Classify by strongest signal ───

    # Compact: very short conversations with minimal tool use
    if n_turns <= 5 and total_tool_calls <= 2:
        return "compact", f"{n_turns} turns, {total_tool_calls} tool calls"

    # Planning: lots of thinking, few tools
    thinking_ratio = info["thinking_chars"] / max(info["total_gpt_chars"], 1)
    if thinking_ratio > 0.5 and total_tool_calls <= 3:
        return "planning", f"{thinking_ratio:.0%} thinking, {total_tool_calls} tools"

    # Testing: test files + test bash commands
    if test_files >= 2 or (test_files >= 1 and test_commands >= 1):
        return "testing", f"{test_files} test files, {test_commands} test commands"
    if test_commands >= 2:
        return "testing", f"{test_commands} test commands"

    # Frontend: touching frontend files
    if frontend_files >= 2 or (frontend_files >= 1 and frontend_files / max(len(file_paths), 1) > 0.4):
        return "frontend", f"{frontend_files}/{len(file_paths)} frontend files"

    # Config: touching config files
    if config_files >= 2 or (config_files >= 1 and config_files / max(len(file_paths), 1) > 0.5):
        return "config", f"{config_files}/{len(file_paths)} config files"

    # Debugging: error results + edits (fix cycle)
    error_count = tools.get("_error_result", 0)
    if error_count >= 2 and info["edit_count"] >= 1:
        return "debugging", f"{error_count} errors + {info['edit_count']} edits"

    # Commands: heavy bash, few edits
    bash_count = tools.get("Bash", 0)
    if bash_count >= 3 and info["edit_count"] <= 1:
        return "commands", f"{bash_count} bash calls, {info['edit_count']} edits"

    # Exploration: heavy read/search, few edits
    search_count = tools.get("Grep", 0) + tools.get("Glob", 0) + tools.get("Read", 0)
    if search_count >= 4 and info["edit_count"] + info["write_count"] <= 1:
        return "exploration", f"{search_count} search/read, {info['edit_count']} edits"

    # Refactor: many edits on existing files, few new files
    if info["edit_count"] >= 4 and info["write_count"] <= 1:
        return "refactor", f"{info['edit_count']} edits, {info['write_count']} writes"

    # Code editing: default for conversations with edits/writes
    if info["edit_count"] + info["write_count"] >= 2:
        return "code_edit", f"{info['edit_count']} edits + {info['write_count']} writes"

    # Fallback: exploration if any tools, compact if not
    if total_tool_calls >= 2:
        return "exploration", f"fallback ({total_tool_calls} tool calls)"
    return "compact", f"fallback ({n_turns} turns, {total_tool_calls} tools)"


def analyze_dataset(args):
    """Analyze a Claude Code dataset and show classification distribution."""
    print(f"Analyzing {args.dataset}...")

    examples = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} conversations")

    labels = Counter()
    reasons = defaultdict(list)
    per_label_examples = defaultdict(list)

    for ex in examples:
        label, reason = classify_conversation(ex)
        labels[label] += 1
        reasons[label].append(reason)

        # Store first user message for preview
        convs = ex.get("conversations", [])
        for turn in convs:
            if turn.get("from") == "human":
                preview = turn["value"][:120].replace("\n", " ")
                per_label_examples[label].append(preview)
                break

    print(f"\n  {'Category':<15s} {'Count':>6s} {'%':>6s}  Description")
    print(f"  {'-'*80}")
    for label, count in labels.most_common():
        pct = 100 * count / len(examples)
        desc = EXPERT_CATEGORIES.get(label, "")
        print(f"  {label:<15s} {count:>6d} {pct:>5.1f}%  {desc}")

    print(f"\n  Total: {len(examples)} conversations → {len(labels)} categories")

    # Show example reasons + previews for each category
    if args.verbose:
        for label in sorted(labels.keys()):
            print(f"\n  ── {label} ({labels[label]} examples) ──")
            for reason, preview in zip(reasons[label][:3], per_label_examples[label][:3]):
                print(f"    [{reason}] {preview}...")

    # Show recommended min-count for balanced training
    min_count = min(labels.values())
    print(f"\n  Smallest category: {min_count} examples")
    print(f"  For balanced training, each expert gets ≤{min_count} examples")
    print(f"  Consider merging small categories or using --min-count to filter")


def train_router(args):
    """Train the router LoRA directly from Claude Code traces."""
    import torch
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading and classifying {args.dataset}...")
    examples = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    # Classify every conversation
    router_data = []
    label_counts = Counter()
    skipped = 0

    for ex in examples:
        label, reason = classify_conversation(ex)

        # Find first user message as router input
        convs = ex.get("conversations", [])
        user_msg = None
        for turn in convs:
            if turn.get("from") == "human":
                text = turn.get("value", "").strip()
                # Skip tool results as the "first" message
                if text.startswith("<tool_response>"):
                    continue
                if len(text) >= 10:
                    user_msg = text
                    break

        if user_msg is None:
            skipped += 1
            continue

        # Truncate
        user_msg = user_msg[:args.max_chars]
        router_data.append({"input": user_msg, "label": label})
        label_counts[label] += 1

    print(f"  Classified {len(router_data)} conversations ({skipped} skipped)")

    # Filter small categories
    if args.min_count > 0:
        keep_labels = {l for l, c in label_counts.items() if c >= args.min_count}
        before = len(router_data)
        router_data = [ex for ex in router_data if ex["label"] in keep_labels]
        removed_labels = set(label_counts.keys()) - keep_labels
        if removed_labels:
            print(f"  Removed categories with <{args.min_count} examples: {', '.join(removed_labels)}")
            print(f"  {before} → {len(router_data)} examples")
        label_counts = Counter(ex["label"] for ex in router_data)

    # Balance if requested
    if args.balance:
        min_count = min(label_counts.values())
        balanced = []
        per_label = Counter()
        for ex in router_data:
            if per_label[ex["label"]] < min_count:
                balanced.append(ex)
                per_label[ex["label"]] += 1
        print(f"  Balanced to {len(balanced)} examples ({min_count} per label)")
        router_data = balanced
        label_counts = Counter(ex["label"] for ex in router_data)

    # Build label mapping
    expert_names = sorted(label_counts.keys())
    label2id = {name: i for i, name in enumerate(expert_names)}
    id2label = {i: name for name, i in label2id.items()}
    num_labels = len(expert_names)

    print(f"\n{'='*60}")
    print(f"Router LoRA Training")
    print(f"{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Experts:  {', '.join(expert_names)} ({num_labels} classes)")
    print(f"  LoRA:     rank={args.lora_rank}")
    print(f"  Output:   {args.output}")
    print(f"\n  Label distribution:")
    for label, count in label_counts.most_common():
        print(f"    {label:<15s} {count:>5d} ({100*count/len(router_data):.1f}%)")

    # Save classified data for reproducibility
    classified_path = os.path.join(args.output, "router-training-data.jsonl")
    os.makedirs(args.output, exist_ok=True)
    with open(classified_path, "w", encoding="utf-8") as f:
        for ex in router_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n  Saved classified data to: {classified_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Build HF dataset
    dataset = Dataset.from_list(router_data)
    dataset = dataset.map(lambda ex: {"label": label2id[ex["label"]]})

    def tokenize(examples):
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["input"])
    split = dataset.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        # Per-class accuracy
        per_class = {}
        for i, name in id2label.items():
            mask = labels == i
            if mask.sum() > 0:
                per_class[f"acc_{name}"] = float((predictions[mask] == labels[mask]).mean())
        return {"accuracy": accuracy, **per_class}

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"router-{args.model.split('/')[-1]}-{num_labels}experts",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining router...")
    result = trainer.train()

    eval_results = trainer.evaluate()
    print(f"\n{'='*60}")
    print(f"Router training complete!")
    print(f"  Train loss:    {result.training_loss:.4f}")
    print(f"  Eval accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
    for k, v in eval_results.items():
        if k.startswith("eval_acc_"):
            print(f"  {k.replace('eval_', ''):>20s}: {v:.2%}")

    # Save
    final_dir = os.path.join(args.output, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    mapping = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "experts": EXPERT_CATEGORIES,
    }
    with open(os.path.join(final_dir, "label_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"  Saved to: {final_dir}")
    print(f"\nTo use with vLLM multi-LoRA serving, load this adapter as 'router'")
    print(f"alongside your expert LoRAs ({', '.join(expert_names)}).")


def split_dataset(args):
    """Split a Claude Code dataset into per-expert subsets for training expert LoRAs."""
    print(f"Splitting {args.dataset} by expert category...")

    examples = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    os.makedirs(args.output_dir, exist_ok=True)
    writers = {}
    counts = Counter()

    for ex in examples:
        label, reason = classify_conversation(ex)
        counts[label] += 1

        if label not in writers:
            path = os.path.join(args.output_dir, f"{label}.jsonl")
            writers[label] = open(path, "w", encoding="utf-8")

        writers[label].write(json.dumps(ex, ensure_ascii=False) + "\n")

    for w in writers.values():
        w.close()

    print(f"\n  Split {len(examples)} conversations into {len(writers)} expert datasets:")
    for label, count in counts.most_common():
        path = os.path.join(args.output_dir, f"{label}.jsonl")
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"    {label:<15s} {count:>5d} conversations  ({size_mb:.1f} MB)")
        desc = EXPERT_CATEGORIES.get(label, "")
        print(f"    {'':15s} {desc}")

    print(f"\n  Output directory: {args.output_dir}/")
    print(f"\n  Train each expert with:")
    print(f"    python train.py --dataset {args.output_dir}/<expert>.jsonl \\")
    print(f"        --output checkpoints/expert-<name> --lora-rank 32")


def eval_router(args):
    """Evaluate a trained router on Claude Code traces."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    mapping_path = os.path.join(args.adapter, "final", "label_mapping.json")
    if not os.path.exists(mapping_path):
        mapping_path = os.path.join(args.adapter, "label_mapping.json")
    with open(mapping_path) as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    num_labels = len(label2id)

    print(f"Evaluating router ({num_labels} experts: {', '.join(label2id.keys())})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    adapter_path = args.adapter
    if os.path.isdir(os.path.join(args.adapter, "final")):
        adapter_path = os.path.join(args.adapter, "final")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Classify ground truth from traces, then compare to router predictions
    print(f"Loading and classifying {args.dataset}...")
    examples = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    correct = 0
    total = 0
    confusion = Counter()
    per_class_correct = Counter()
    per_class_total = Counter()

    for ex in examples:
        true_label, _ = classify_conversation(ex)
        if true_label not in label2id:
            continue

        # Get first user message
        convs = ex.get("conversations", [])
        user_msg = None
        for turn in convs:
            if turn.get("from") == "human":
                text = turn.get("value", "").strip()
                if not text.startswith("<tool_response>") and len(text) >= 10:
                    user_msg = text[:2000]
                    break
        if user_msg is None:
            continue

        inputs = tokenizer(user_msg, truncation=True, max_length=512,
                           padding="max_length", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = logits.argmax(-1).item()
            pred_label = id2label[pred_id]

        per_class_total[true_label] += 1
        if pred_label == true_label:
            correct += 1
            per_class_correct[true_label] += 1
        else:
            confusion[(true_label, pred_label)] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n  Overall accuracy: {correct}/{total} ({100*accuracy:.1f}%)")

    print(f"\n  Per-class accuracy:")
    for label in sorted(per_class_total.keys()):
        c = per_class_correct[label]
        t = per_class_total[label]
        print(f"    {label:<15s} {c:>4d}/{t:<4d} ({100*c/max(t,1):.1f}%)")

    if confusion:
        print(f"\n  Top misclassifications:")
        for (true, pred), count in confusion.most_common(15):
            print(f"    {true:<15s} → {pred:<15s} {count:>3d}x")


def main():
    parser = argparse.ArgumentParser(
        description="Train a router LoRA for Claude Code expert selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expert categories (auto-detected from tool usage patterns):
  code_edit    — Implementing features (heavy Edit/Write on source files)
  exploration  — Searching codebases (heavy Grep/Glob/Read)
  debugging    — Finding/fixing bugs (error patterns + targeted edits)
  planning     — Architecture/design (long thinking, minimal tools)
  commands     — Shell tasks (heavy Bash — builds, git, deploys)
  frontend     — UI work (edits on .tsx/.css/.html/.vue/.svelte)
  config       — Config files (Docker, CI/CD, yaml, package.json)
  testing      — Test writing/running (test files + test commands)
  refactor     — Restructuring code (many edits, few new files)
  compact      — Quick answers (short conversations, minimal tools)

Examples:
  python train_router.py analyze --dataset claude-traces-split.jsonl
  python train_router.py train --dataset claude-traces-split.jsonl
  python train_router.py split --dataset claude-traces-split.jsonl
""",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ─── analyze ───
    ana = sub.add_parser("analyze", help="Analyze dataset classification distribution")
    ana.add_argument("--dataset", required=True, help="Claude Code traces JSONL")
    ana.add_argument("--verbose", "-v", action="store_true", help="Show example conversations per category")

    # ─── train ───
    trn = sub.add_parser("train", help="Classify traces + train router LoRA (one command)")
    trn.add_argument("--dataset", required=True, help="Claude Code traces JSONL")
    trn.add_argument("--model", default="unsloth/Qwen3.5-0.8B", help="Base model")
    trn.add_argument("--output", default="checkpoints/router-lora", help="Output directory")
    trn.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (default: 32)")
    trn.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    trn.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    trn.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    trn.add_argument("--max-length", type=int, default=512, help="Max input tokens (default: 512)")
    trn.add_argument("--max-chars", type=int, default=2000, help="Max chars per input message")
    trn.add_argument("--min-count", type=int, default=5, help="Min examples per category (default: 5)")
    trn.add_argument("--balance", action="store_true", help="Balance dataset (downsample to smallest class)")

    # ─── split ───
    spl = sub.add_parser("split", help="Split traces into per-expert datasets")
    spl.add_argument("--dataset", required=True, help="Claude Code traces JSONL")
    spl.add_argument("--output-dir", default="expert-datasets", help="Output directory for split files")

    # ─── eval ───
    evl = sub.add_parser("eval", help="Evaluate a trained router against ground-truth classification")
    evl.add_argument("--dataset", required=True, help="Claude Code traces JSONL")
    evl.add_argument("--model", required=True, help="Base model")
    evl.add_argument("--adapter", required=True, help="Path to trained router adapter")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_dataset(args)
    elif args.command == "train":
        train_router(args)
    elif args.command == "split":
        split_dataset(args)
    elif args.command == "eval":
        eval_router(args)


if __name__ == "__main__":
    main()
