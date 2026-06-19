#!/usr/bin/env python3
"""PyTorch CPU oracle timing for the SmoothL1 grad gauntlet row."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import torch
import torch.nn.functional as F


def run_one(n: int) -> float:
    x = torch.randn(n, dtype=torch.float64, requires_grad=True)
    target = torch.randn(n, dtype=torch.float64)
    loss = F.smooth_l1_loss(x, target, reduction="mean", beta=1.0)
    loss.backward()
    checksum = float(loss.detach()) + float(x.grad[0])
    return checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4096 * 2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    checksum = 0.0
    for _ in range(args.warmups):
        checksum += run_one(args.n)
    gc.collect()

    times_ms: list[float] = []
    for _ in range(args.iters):
        start = time.perf_counter_ns()
        checksum += run_one(args.n)
        end = time.perf_counter_ns()
        times_ms.append((end - start) / 1_000_000.0)

    ordered = sorted(times_ms)
    p95_idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    result = {
        "benchmark": "torch_smooth_l1_grad_8m",
        "n": args.n,
        "warmups": args.warmups,
        "iters": args.iters,
        "dtype": "float64",
        "reduction": "mean",
        "beta": 1.0,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p95_ms": ordered[p95_idx],
        "times_ms": times_ms,
        "checksum": checksum,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
