"""Configurable GPU VRAM allocation workload for container testing.

Usage:
    python gpu_workload.py --mb 512 --duration 60

Allocates the specified amount of VRAM and holds it for the given duration.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU VRAM allocation workload")
    parser.add_argument("--mb", type=int, default=256, help="VRAM to allocate in MiB")
    parser.add_argument("--duration", type=int, default=60, help="Hold duration in seconds")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    num_floats = (args.mb * 1024 * 1024) // 4  # 4 bytes per float32

    print(f"Allocating {args.mb} MiB on device {args.device}...")
    tensor = torch.zeros(num_floats, dtype=torch.float32, device=device)

    print(f"Allocated. Holding for {args.duration}s (PID={__import__('os').getpid()})...")

    # Graceful shutdown on SIGTERM
    stop = False

    def _handler(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handler)

    elapsed = 0
    while elapsed < args.duration and not stop:
        time.sleep(1)
        elapsed += 1

    del tensor
    torch.cuda.empty_cache()
    print("Released VRAM. Exiting.")


if __name__ == "__main__":
    main()
