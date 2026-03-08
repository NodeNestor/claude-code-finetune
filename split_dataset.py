#!/usr/bin/env python3
"""Split Claude Code trace conversations at compression/continuation boundaries.

Claude Code auto-compresses at ~200K context. When it does, it inserts a
"This session is being continued from a previous conversation that ran out
of context" message with a summary. This script splits at those boundaries
so each segment becomes an independent training example with the system
prompt prepended.

Usage:
    python split_dataset.py [--input FILE] [--output FILE]
"""

import argparse
import json
import sys

MARKER = "This session is being continued from a previous conversation that ran out of context"


def split_conversation(convs):
    """Split a conversation at compression boundaries.

    Returns a list of conversation segments. Each segment after the first
    gets the original system prompt prepended (if present).
    """
    # Find system prompt (first message if role=system)
    system_msg = None
    first_role = convs[0].get("from", convs[0].get("role", ""))
    if first_role == "system":
        system_msg = convs[0]
        start = 1
    else:
        start = 0

    # Find all split points (message indices with the continuation marker)
    split_indices = []
    for j in range(start, len(convs)):
        val = convs[j].get("value", convs[j].get("content", ""))
        if isinstance(val, str) and MARKER in val:
            split_indices.append(j)

    if not split_indices:
        return [convs]  # No splits needed

    # Build segments
    segments = []
    boundaries = [start] + split_indices + [len(convs)]

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]

        if seg_start >= seg_end:
            continue

        segment = list(convs[seg_start:seg_end])

        # Prepend system prompt to all segments
        if system_msg:
            segment.insert(0, system_msg)

        segments.append(segment)

    return segments


def main():
    parser = argparse.ArgumentParser(description="Split conversations at compression boundaries")
    parser.add_argument("--input", "-i", default="claude-traces-dataset.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", "-o", default="claude-traces-split.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()

    total_in = 0
    total_out = 0
    total_splits = 0

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total_in += 1
            obj = json.loads(line)
            convs = obj.get("conversations", [])

            segments = split_conversation(convs)

            if len(segments) > 1:
                total_splits += len(segments) - 1

            for segment in segments:
                out = {"conversations": segment}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"Input:  {total_in} conversations")
    print(f"Output: {total_out} conversations (+{total_splits} from splits)")

    # Length stats
    lengths = []
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            convs = obj["conversations"]
            total_chars = sum(
                len(m.get("value", m.get("content", "")))
                for m in convs
                if isinstance(m.get("value", m.get("content", "")), str)
            )
            lengths.append(total_chars)

    lengths.sort()
    n = len(lengths)
    print(f"\nChar length distribution (~4 chars/token):")
    for label, idx in [("Min", 0), ("P10", n//10), ("P25", n//4),
                        ("Median", n//2), ("P75", 3*n//4), ("P90", 9*n//10),
                        ("P95", int(n*0.95)), ("Max", n-1)]:
        print(f"  {label:>6}: {lengths[idx]:>10,} chars (~{lengths[idx]//4:>7,} tok)")

    print(f"\nFit under seq-length limits:")
    for limit_tok in [4096, 8192, 16384, 32768, 65536, 131072]:
        limit_chars = limit_tok * 4
        count = sum(1 for l in lengths if l <= limit_chars)
        print(f"  <={limit_tok:>6} tok: {count:>4}/{n} ({100*count/n:.0f}%)")


if __name__ == "__main__":
    main()
