"""Small PyTorch training script to test vramtop real-time monitoring.

Trains a simple CNN on synthetic data to generate GPU memory usage patterns:
- Phase 1: Model loading (~200-400 MB)
- Phase 2: Training with growing batch sizes (~1-4 GB)
- Phase 3: Steady-state training (~stable memory)
- Phase 4: Cleanup (memory freed)

Usage:
    python scripts/train_test.py
    # In another terminal: python -m vramtop --export-csv /tmp/train_log.csv
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleCNN(nn.Module):
    """A moderately sized CNN to consume visible GPU memory."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(width * 4 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1000),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def print_gpu_mem(label: str) -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"  [{label}] Allocated: {alloc:.0f} MB, Reserved: {reserved:.0f} MB")


def phase_grow(device: torch.device, target_mb: int = 2000) -> list[torch.Tensor]:
    """Phase: Gradually allocate GPU memory to simulate growing usage."""
    print(f"\n--- Phase: GROWING (target ~{target_mb} MB) ---")
    tensors = []
    allocated = 0
    chunk_mb = 100
    while allocated < target_mb:
        # ~100 MB per chunk (25M float32 values)
        t = torch.randn(25_000_000, device=device)
        tensors.append(t)
        allocated += chunk_mb
        print_gpu_mem(f"Grow +{chunk_mb}MB")
        time.sleep(1)  # Slow enough for vramtop to catch each step
    return tensors


def phase_train(
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 32,
    width: int = 256,
    img_size: int = 64,
) -> None:
    """Phase: Train a CNN on synthetic data (steady-state memory)."""
    print(f"\n--- Phase: TRAINING (epochs={epochs}, batch={batch_size}) ---")

    model = SimpleCNN(width=width).to(device)
    print_gpu_mem("Model loaded")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data
    data = torch.randn(batch_size * 10, 3, img_size, img_size, device=device)
    labels = torch.randint(0, 1000, (batch_size * 10,), device=device)
    print_gpu_mem("Data loaded")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = len(data) // batch_size

        for i in range(n_batches):
            batch_x = data[i * batch_size : (i + 1) * batch_size]
            batch_y = labels[i * batch_size : (i + 1) * batch_size]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")
        print_gpu_mem(f"Epoch {epoch + 1}")
        time.sleep(0.5)

    del model, optimizer, data, labels
    torch.cuda.empty_cache()
    print_gpu_mem("After cleanup")


def phase_shrink(tensors: list[torch.Tensor]) -> None:
    """Phase: Gradually free memory to simulate shrinking usage."""
    print(f"\n--- Phase: SHRINKING (freeing {len(tensors)} chunks) ---")
    while tensors:
        tensors.pop()
        torch.cuda.empty_cache()
        print_gpu_mem(f"Freed chunk, {len(tensors)} remaining")
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU training test for vramtop")
    parser.add_argument("--grow-mb", type=int, default=2000,
                        help="Target MB for growth phase (default: 2000)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--width", type=int, default=256,
                        help="CNN channel width (default: 256)")
    parser.add_argument("--skip-grow", action="store_true",
                        help="Skip the growth phase")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip the training phase")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires an NVIDIA GPU.")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print("Tip: Run 'python -m vramtop' in another terminal to watch memory usage.\n")

    print_gpu_mem("Initial")

    # Phase 1: Grow
    tensors: list[torch.Tensor] = []
    if not args.skip_grow:
        tensors = phase_grow(device, target_mb=args.grow_mb)
        time.sleep(3)  # Hold for vramtop to observe

    # Phase 2: Train
    if not args.skip_train:
        phase_train(device, epochs=args.epochs, batch_size=args.batch_size,
                    width=args.width)
        time.sleep(3)

    # Phase 3: Shrink
    if tensors:
        phase_shrink(tensors)
        time.sleep(3)

    # Final cleanup
    torch.cuda.empty_cache()
    print_gpu_mem("Final")
    print("\nDone! Check vramtop for the memory usage timeline.")


if __name__ == "__main__":
    main()
