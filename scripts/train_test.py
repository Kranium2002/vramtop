"""Finetune a small LLM (GPT-2) on WikiText-2 with vramtop deep mode.

Demonstrates real ML memory patterns for vramtop monitoring:
- Phase 1: Model loading (weights → GPU)
- Phase 2: Dataset tokenization (CPU, minimal GPU)
- Phase 3: Training with AdamW (weights + gradients + optimizer states + activations)
- Phase 4: Cleanup

Usage:
    pip install transformers datasets accelerate
    python scripts/train_test.py
    # In another terminal: python -m vramtop
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch


def print_gpu_mem(label: str) -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"  [{label}] Allocated: {alloc:.0f} MB, Reserved: {reserved:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune GPT-2 on WikiText-2 for vramtop testing")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default: 256)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max training steps (0 = full epoch)")
    parser.add_argument("--no-deep-mode", action="store_true",
                        help="Disable vramtop deep mode reporter")
    args = parser.parse_args()

    # --- Check dependencies ---
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: Missing dependencies. Install with:")
        print("  pip install transformers datasets accelerate")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires an NVIDIA GPU.")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Model: {args.model}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"Seq len: {args.seq_len}  |  LR: {args.lr}")
    print()
    print("Tip: Run 'python -m vramtop' in another terminal to watch memory usage.")
    print("     Press 'd' to open the detail panel and see PyTorch internals.")
    print()

    # --- Deep mode: expose PyTorch internals to vramtop ---
    if not args.no_deep_mode:
        try:
            from vramtop.reporter.pytorch import report
            report()
            print("Deep mode reporter started.\n")
        except Exception as exc:
            print(f"Deep mode unavailable: {exc}\n")

    # === Phase 1: Load model ===
    print("=" * 60)
    print("Phase 1: Loading model")
    print("=" * 60)
    print_gpu_mem("Before model load")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {trainable:,} trainable")
    print_gpu_mem("Model on GPU")
    time.sleep(2)

    # === Phase 2: Load and tokenize WikiText-2 ===
    print()
    print("=" * 60)
    print("Phase 2: Loading WikiText-2 dataset")
    print("=" * 60)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Filter empty lines and tokenize
    texts = [row["text"] for row in dataset if row["text"].strip()]
    print(f"  {len(texts):,} non-empty text samples")

    # Concatenate all text and chunk into fixed-length sequences
    full_text = "\n".join(texts)
    tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    all_ids = tokens["input_ids"].squeeze(0)
    print(f"  Total tokens: {len(all_ids):,}")

    # Chunk into sequences of seq_len
    n_chunks = len(all_ids) // args.seq_len
    all_ids = all_ids[: n_chunks * args.seq_len]
    input_ids = all_ids.view(n_chunks, args.seq_len)
    print(f"  Chunks of {args.seq_len} tokens: {n_chunks:,}")
    print_gpu_mem("After tokenization")
    time.sleep(1)

    # === Phase 3: Training ===
    print()
    print("=" * 60)
    print(f"Phase 3: Finetuning ({args.epochs} epochs)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print_gpu_mem("Optimizer created")

    n_batches = n_chunks // args.batch_size
    max_steps = args.max_steps if args.max_steps > 0 else n_batches * args.epochs
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        # Shuffle indices each epoch
        perm = torch.randperm(n_chunks)

        for i in range(0, n_chunks - args.batch_size + 1, args.batch_size):
            if global_step >= max_steps:
                break

            batch_idx = perm[i : i + args.batch_size]
            batch = input_ids[batch_idx].to(device)

            # Causal LM: input = tokens, labels = same tokens shifted
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log every 10 steps
            if global_step % 10 == 0:
                avg = epoch_loss / epoch_steps
                ppl = math.exp(min(avg, 20))  # cap to avoid overflow
                elapsed = time.time() - t_start
                steps_per_sec = epoch_steps / elapsed if elapsed > 0 else 0
                print(
                    f"  Step {global_step:>5d} | "
                    f"Loss: {avg:.4f} | "
                    f"PPL: {ppl:.1f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )
                print_gpu_mem(f"Step {global_step}")

        if global_step >= max_steps:
            print(f"\n  Reached max_steps={max_steps}, stopping.")
            break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t_start
        print(
            f"\n  Epoch {epoch + 1}/{args.epochs} complete | "
            f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
            f"Time: {elapsed:.1f}s"
        )
        print_gpu_mem(f"Epoch {epoch + 1} end")
        time.sleep(2)  # Pause for vramtop to observe steady state

    # === Phase 4: Cleanup ===
    print()
    print("=" * 60)
    print("Phase 4: Cleanup")
    print("=" * 60)
    print_gpu_mem("Before cleanup")

    del model, optimizer, input_ids, all_ids
    torch.cuda.empty_cache()
    time.sleep(2)

    print_gpu_mem("After cleanup")
    print()
    print("Done! Check vramtop for the memory usage timeline.")
    print("  - Memory bar should have shown green → yellow gradient")
    print("  - Timeline sparkline captures the full training curve")
    print("  - Phase detector should have caught GROWING → STABLE → SHRINKING")
    print("  - Detail panel (d) shows PyTorch allocated vs reserved vs active")


if __name__ == "__main__":
    main()
