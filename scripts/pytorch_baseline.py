#!/usr/bin/env python3
"""PyTorch baseline benchmarks for apples-to-apples comparison with FrankenTorch."""

import torch
import time
import statistics
import json
import sys
import platform

def benchmark(name, setup_fn, bench_fn, warmup=10, runs=30):
    """Run benchmark with warmup and collect timing statistics."""
    # Setup once
    args = setup_fn()

    # Warmup - run many times to ensure JIT is warmed
    for _ in range(warmup):
        bench_fn(*args)

    # Synchronize before timed runs (CPU is synchronous but good practice)
    torch.cpu.synchronize()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        bench_fn(*args)
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # nanoseconds to microseconds

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]
    mean = statistics.mean(times)
    stddev = statistics.stdev(times) if len(times) > 1 else 0

    return {
        "name": name,
        "p50_us": round(p50, 2),
        "p95_us": round(p95, 2),
        "p99_us": round(p99, 2),
        "mean_us": round(mean, 2),
        "stddev_us": round(stddev, 2),
        "runs": runs,
    }

def main():
    results = []

    # Set to single thread for fair comparison with FrankenTorch
    torch.set_num_threads(1)

    # Environment fingerprint
    env = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "num_threads": torch.get_num_threads(),
    }

    print(f"PyTorch {torch.__version__} on {platform.platform()}", file=sys.stderr)
    print(f"Using {torch.get_num_threads()} threads (single-threaded for fair comparison)", file=sys.stderr)
    print("Running benchmarks...", file=sys.stderr)

    # Matmul benchmarks
    for size in [64, 128, 256, 512, 1024]:
        results.append(benchmark(
            f"matmul_{size}x{size}",
            lambda s=size: (torch.randn(s, s), torch.randn(s, s)),
            lambda a, b: torch.matmul(a, b)
        ))

    # BMM benchmarks
    for batch in [8, 16, 32]:
        n = 128
        results.append(benchmark(
            f"bmm_b{batch}_128x128",
            lambda b=batch, n=n: (torch.randn(b, n, n), torch.randn(b, n, n)),
            lambda a, b: torch.bmm(a, b)
        ))

    # Conv2d benchmarks
    for hw in [32, 64, 128]:
        batch, in_ch, out_ch, kh, kw = 4, 64, 64, 3, 3
        results.append(benchmark(
            f"conv2d_{hw}x{hw}",
            lambda h=hw: (torch.randn(batch, in_ch, h, h), torch.randn(out_ch, in_ch, kh, kw)),
            lambda x, w: torch.nn.functional.conv2d(x, w, padding=1)
        ))

    # Sum benchmarks
    for size in [1000, 10000, 100000, 1000000]:
        results.append(benchmark(
            f"sum_{size}",
            lambda s=size: (torch.randn(s),),
            lambda x: x.sum()
        ))

    # Softmax benchmarks
    for vocab in [128, 512, 2048, 8192]:
        batch = 32
        results.append(benchmark(
            f"softmax_b32_{vocab}",
            lambda v=vocab: (torch.randn(batch, v),),
            lambda x: torch.nn.functional.softmax(x, dim=-1)
        ))

    # ReLU benchmarks
    for size in [10000, 100000, 1000000]:
        results.append(benchmark(
            f"relu_{size}",
            lambda s=size: (torch.randn(s),),
            lambda x: torch.relu(x)
        ))

    # Exp benchmarks
    for size in [10000, 100000, 1000000]:
        results.append(benchmark(
            f"exp_{size}",
            lambda s=size: (torch.randn(s),),
            lambda x: torch.exp(x)
        ))

    # Add benchmarks
    for size in [10000, 100000, 1000000]:
        results.append(benchmark(
            f"add_{size}",
            lambda s=size: (torch.randn(s), torch.randn(s)),
            lambda x, y: x + y
        ))

    # Backward matmul benchmarks - need fresh tensors each time
    for size in [64, 128, 256]:
        def backward_matmul(s=size):
            a = torch.randn(s, s, requires_grad=True)
            b = torch.randn(s, s, requires_grad=True)
            c = torch.matmul(a, b)
            loss = c.sum()
            loss.backward()
            return a.grad, b.grad

        times = []
        for _ in range(10):
            backward_matmul()  # warmup

        for _ in range(30):
            start = time.perf_counter_ns()
            backward_matmul()
            end = time.perf_counter_ns()
            times.append((end - start) / 1000.0)

        times.sort()
        results.append({
            "name": f"backward_matmul_{size}x{size}",
            "p50_us": round(times[len(times) // 2], 2),
            "p95_us": round(times[int(len(times) * 0.95)], 2),
            "p99_us": round(times[int(len(times) * 0.99)], 2),
            "mean_us": round(statistics.mean(times), 2),
            "stddev_us": round(statistics.stdev(times), 2),
            "runs": 30,
        })

    # Linear forward benchmarks
    for hidden in [256, 512, 1024, 2048]:
        batch, in_features = 32, 512
        results.append(benchmark(
            f"linear_b32_512_{hidden}",
            lambda h=hidden: (torch.randn(batch, in_features), torch.randn(h, in_features), torch.randn(h)),
            lambda x, w, bias: torch.nn.functional.linear(x, w, bias)
        ))

    output = {
        "environment": env,
        "benchmarks": results,
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
