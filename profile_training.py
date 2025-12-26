"""
Profile the training pipeline to identify bottlenecks.

Tests:
1. Data loading speed
2. Forward pass timing
3. Backward pass timing
4. Optimizer step timing
"""

import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import statistics

from src.models import GPT2, GrassmannGPT, GrassmannGPTv2
from src.data import load_wikitext, get_dataloader, get_tokenizer


@dataclass
class TimingStats:
    name: str
    times: list

    @property
    def mean(self):
        return statistics.mean(self.times) * 1000  # ms

    @property
    def std(self):
        return statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0

    @property
    def min(self):
        return min(self.times) * 1000

    @property
    def max(self):
        return max(self.times) * 1000

    def __str__(self):
        return f"{self.name}: {self.mean:.2f}ms +/- {self.std:.2f}ms (min={self.min:.2f}, max={self.max:.2f})"


class Profiler:
    def __init__(self):
        self.timings = {}

    @contextmanager
    def track(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    def get_stats(self):
        return {name: TimingStats(name, times) for name, times in self.timings.items()}

    def report(self):
        print("\n" + "="*60)
        print("PROFILING RESULTS")
        print("="*60)

        stats = self.get_stats()
        total_time = 0

        for name in ["data_loading", "forward", "backward", "optimizer_step", "total_step"]:
            if name in stats:
                print(f"  {stats[name]}")
                if name != "total_step":
                    total_time += stats[name].mean

        if "total_step" in stats:
            overhead = stats["total_step"].mean - total_time
            print(f"\n  Overhead (misc): {overhead:.2f}ms")

            # Breakdown percentages
            print("\n  Breakdown:")
            for name in ["data_loading", "forward", "backward", "optimizer_step"]:
                if name in stats:
                    pct = (stats[name].mean / stats["total_step"].mean) * 100
                    print(f"    {name}: {pct:.1f}%")

        print("="*60)


def profile_model(
    model_type: str,
    device: torch.device,
    batch_size: int = 16,
    seq_len: int = 256,
    num_warmup: int = 5,
    num_steps: int = 20,
    use_amp: bool = True,
):
    """Profile a single model."""
    print(f"\n{'='*60}")
    print(f"Profiling {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  AMP: {use_amp}")

    # Load tokenizer and data
    print("\nLoading tokenizer and data...")
    tokenizer = get_tokenizer("gpt2")
    vocab_size = len(tokenizer)

    dataset = load_wikitext(
        tokenizer,
        seq_len=seq_len,
        version="wikitext-2-raw-v1",
        split="train",
    )
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    data_iter = iter(dataloader)

    # Create model
    print("Creating model...")
    if model_type == "gpt2":
        model = GPT2(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            model_dim=512,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
        )
    elif model_type == "grassmann":
        model = GrassmannGPT(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            model_dim=512,
            num_layers=6,
            reduced_dim=64,
            dropout=0.1,
        )
    elif model_type == "grassmann_v2":
        model = GrassmannGPTv2(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            model_dim=512,
            num_layers=6,
            reduced_dim=64,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    num_params = model.get_num_params()
    print(f"  Parameters: {num_params:,}")

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    profiler = Profiler()

    # Warmup
    print(f"\nWarming up ({num_warmup} steps)...")
    model.train()
    for _ in range(num_warmup):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if scaler:
            with autocast():
                _, loss = model(input_ids, labels=labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(input_ids, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Profile
    print(f"Profiling ({num_steps} steps)...")
    for step in range(num_steps):
        with profiler.track("total_step"):
            # Data loading
            with profiler.track("data_loading"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

            # Forward
            with profiler.track("forward"):
                if scaler:
                    with autocast():
                        _, loss = model(input_ids, labels=labels)
                else:
                    _, loss = model(input_ids, labels=labels)

            # Backward
            optimizer.zero_grad()
            with profiler.track("backward"):
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Optimizer step
            with profiler.track("optimizer_step"):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

    # Report
    profiler.report()

    # Calculate throughput
    stats = profiler.get_stats()
    step_time_s = stats["total_step"].mean / 1000
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / step_time_s
    samples_per_sec = batch_size / step_time_s

    print(f"\nThroughput:")
    print(f"  {tokens_per_sec:,.0f} tokens/sec")
    print(f"  {samples_per_sec:.1f} samples/sec")
    print(f"  {step_time_s*1000:.1f} ms/step")

    # Memory usage
    if device.type == "cuda":
        mem_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved: {mem_reserved:.2f} GB")
        torch.cuda.reset_peak_memory_stats(device)

    return profiler.get_stats()


def main():
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Running on CPU (no GPU available)")

    # Profile all models
    gpt2_stats = profile_model("gpt2", device)
    grassmann_v2_stats = profile_model("grassmann_v2", device)

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON: GPT-2 vs GrassmannGPT v2 (Optimized)")
    print("="*60)

    for name in ["forward", "backward", "total_step"]:
        if name in gpt2_stats and name in grassmann_v2_stats:
            gpt2_time = gpt2_stats[name].mean
            grass_time = grassmann_v2_stats[name].mean
            ratio = grass_time / gpt2_time
            diff = grass_time - gpt2_time
            print(f"\n{name}:")
            print(f"  GPT-2:        {gpt2_time:.2f} ms")
            print(f"  Grassmann v2: {grass_time:.2f} ms")
            print(f"  Ratio:        {ratio:.2f}x")
            print(f"  Difference:   {diff:+.2f} ms")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
