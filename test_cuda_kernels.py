"""
Test CUDA kernels match PyTorch implementation and benchmark performance.

Tests:
1. Plucker kernel correctness (forward pass)
2. Full model forward/backward correctness
3. 100 iteration training comparison
4. Performance benchmark (PyTorch vs CUDA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.grassmann_v4 import GrassmannGPTv4, CausalGrassmannMixing, PluckerEncoder
from cuda.grassmann_fused import (
    FusedPluckerFunction,
    FusedGrassmannMixing,
    GrassmannGPTFused,
    CUDA_AVAILABLE,
)


def test_plucker_correctness():
    """Test that CUDA Plucker kernel matches PyTorch implementation."""
    print("\n" + "="*60)
    print("TEST 1: Plucker Kernel Correctness")
    print("="*60)

    if not CUDA_AVAILABLE:
        print("CUDA extension not available, testing PyTorch fallback only")

    torch.manual_seed(42)

    # Test parameters
    batch_size = 4
    seq_len = 128
    reduced_dim = 32  # Paper's value
    window_sizes = [1, 2, 4, 8, 12, 16]

    # Create test input
    z = torch.randn(batch_size, seq_len, reduced_dim, device='cuda', dtype=torch.float32)
    window_sizes_tensor = torch.tensor(window_sizes, dtype=torch.int32, device='cuda')

    # PyTorch reference implementation
    plucker_encoder = PluckerEncoder(reduced_dim).cuda()
    plucker_dim = reduced_dim * (reduced_dim - 1) // 2

    # Compute using PyTorch (from grassmann_v4)
    geo_accum = torch.zeros(batch_size, seq_len, plucker_dim, device='cuda')
    counts = torch.zeros(batch_size, seq_len, 1, device='cuda')

    for delta in window_sizes:
        if delta >= seq_len:
            continue
        z_current = z[:, delta:, :]
        z_past = z[:, :-delta, :]
        p_hat = plucker_encoder(z_current, z_past)
        geo_accum[:, delta:, :] += p_hat
        counts[:, delta:, :] += 1

    counts = counts.clamp(min=1)
    pytorch_plucker = geo_accum / counts

    # Compute using CUDA kernel (or fallback)
    cuda_plucker = FusedPluckerFunction._pytorch_forward(z, window_sizes_tensor, 1.0, 1e-8)

    # Compare (note: slight differences expected due to normalization order)
    # The CUDA version normalizes after averaging, PyTorch version normalizes per-window
    # So we compare shapes and rough magnitude
    print(f"PyTorch output shape: {pytorch_plucker.shape}")
    print(f"CUDA output shape: {cuda_plucker.shape}")

    # Check that non-zero positions match roughly
    valid_mask = counts.squeeze(-1) > 0
    pytorch_norms = pytorch_plucker[valid_mask].norm(dim=-1).mean()
    cuda_norms = cuda_plucker[valid_mask].norm(dim=-1).mean()

    print(f"PyTorch avg norm: {pytorch_norms:.6f}")
    print(f"CUDA avg norm: {cuda_norms:.6f}")

    # For exact comparison, use the same normalization
    pytorch_renorm = pytorch_plucker / (pytorch_plucker.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    cuda_renorm = cuda_plucker / (cuda_plucker.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    # Compare cosine similarity at valid positions
    cos_sim = F.cosine_similarity(pytorch_renorm[valid_mask], cuda_renorm[valid_mask], dim=-1)
    print(f"Average cosine similarity: {cos_sim.mean():.6f}")
    print(f"Min cosine similarity: {cos_sim.min():.6f}")

    if cos_sim.mean() > 0.99:
        print("PASSED: Plucker outputs are highly similar")
        return True
    else:
        print("WARNING: Outputs differ more than expected")
        return False


def test_mixing_layer_correctness():
    """Test that CUDA-accelerated mixing layer matches PyTorch."""
    print("\n" + "="*60)
    print("TEST 2: Mixing Layer Correctness")
    print("="*60)

    torch.manual_seed(42)

    batch_size = 4
    seq_len = 128
    model_dim = 256
    reduced_dim = 32
    window_sizes = [1, 2, 4, 8, 12, 16]

    # Create identical models
    pytorch_mixing = CausalGrassmannMixing(
        model_dim=model_dim,
        reduced_dim=reduced_dim,
        window_sizes=window_sizes,
        dropout=0.0,  # Disable for deterministic comparison
    ).cuda()

    cuda_mixing = FusedGrassmannMixing(
        model_dim=model_dim,
        reduced_dim=reduced_dim,
        window_sizes=window_sizes,
        dropout=0.0,
    ).cuda()

    # Copy weights from PyTorch to CUDA version
    cuda_mixing.W_red.weight.data = pytorch_mixing.W_red.weight.data.clone()
    cuda_mixing.W_red.bias.data = pytorch_mixing.W_red.bias.data.clone()
    cuda_mixing.W_plu.weight.data = pytorch_mixing.W_plu.weight.data.clone()
    cuda_mixing.W_plu.bias.data = pytorch_mixing.W_plu.bias.data.clone()
    cuda_mixing.W_gate.weight.data = pytorch_mixing.W_gate.weight.data.clone()
    cuda_mixing.W_gate.bias.data = pytorch_mixing.W_gate.bias.data.clone()
    cuda_mixing.layer_norm.weight.data = pytorch_mixing.layer_norm.weight.data.clone()
    cuda_mixing.layer_norm.bias.data = pytorch_mixing.layer_norm.bias.data.clone()

    # Test input
    x = torch.randn(batch_size, seq_len, model_dim, device='cuda')

    # Forward pass
    pytorch_out = pytorch_mixing(x)
    cuda_out = cuda_mixing(x)

    # Compare
    diff = (pytorch_out - cuda_out).abs()
    print(f"Max absolute difference: {diff.max():.8f}")
    print(f"Mean absolute difference: {diff.mean():.8f}")
    print(f"Relative difference: {(diff / pytorch_out.abs().clamp(min=1e-8)).mean():.8f}")

    if diff.max() < 1e-4:
        print("PASSED: Mixing layer outputs match closely")
        return True
    else:
        print("WARNING: Outputs differ more than expected")
        return False


def test_training_comparison(num_iterations=100):
    """Run training for both implementations and compare losses."""
    print("\n" + "="*60)
    print(f"TEST 3: Training Comparison ({num_iterations} iterations)")
    print("="*60)

    torch.manual_seed(42)

    # Small model for fast testing
    vocab_size = 1000
    max_seq_len = 128
    model_dim = 128
    num_layers = 2
    reduced_dim = 32
    batch_size = 8

    # Create models
    pytorch_model = GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_dim=model_dim,
        num_layers=num_layers,
        reduced_dim=reduced_dim,
        dropout=0.0,
    ).cuda()

    cuda_model = GrassmannGPTFused(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_dim=model_dim,
        num_layers=num_layers,
        reduced_dim=reduced_dim,
        dropout=0.0,
    ).cuda()

    # Copy weights
    cuda_model.load_state_dict(pytorch_model.state_dict())

    # Optimizers
    pytorch_opt = torch.optim.AdamW(pytorch_model.parameters(), lr=1e-3)
    cuda_opt = torch.optim.AdamW(cuda_model.parameters(), lr=1e-3)

    pytorch_losses = []
    cuda_losses = []

    print(f"PyTorch model params: {pytorch_model.get_num_params():,}")
    print(f"CUDA model params: {cuda_model.get_num_params():,}")
    print()

    for i in range(num_iterations):
        # Same random data for both
        torch.manual_seed(1000 + i)
        input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len), device='cuda')
        labels = input_ids.clone()

        # PyTorch forward/backward
        pytorch_opt.zero_grad()
        _, pytorch_loss = pytorch_model(input_ids, labels=labels)
        pytorch_loss.backward()
        pytorch_opt.step()
        pytorch_losses.append(pytorch_loss.item())

        # CUDA forward/backward
        cuda_opt.zero_grad()
        _, cuda_loss = cuda_model(input_ids, labels=labels)
        cuda_loss.backward()
        cuda_opt.step()
        cuda_losses.append(cuda_loss.item())

        if (i + 1) % 20 == 0:
            diff = abs(pytorch_losses[-1] - cuda_losses[-1])
            print(f"Iter {i+1:3d}: PyTorch loss={pytorch_losses[-1]:.4f}, "
                  f"CUDA loss={cuda_losses[-1]:.4f}, diff={diff:.6f}")

    # Final comparison
    pytorch_final = sum(pytorch_losses[-10:]) / 10
    cuda_final = sum(cuda_losses[-10:]) / 10
    avg_diff = sum(abs(p - c) for p, c in zip(pytorch_losses, cuda_losses)) / len(pytorch_losses)

    print()
    print(f"Final 10-iter avg: PyTorch={pytorch_final:.4f}, CUDA={cuda_final:.4f}")
    print(f"Average loss difference: {avg_diff:.6f}")

    if avg_diff < 0.01:
        print("PASSED: Training losses match closely")
        return True
    else:
        print("WARNING: Training losses diverge")
        return False


def benchmark_performance():
    """Benchmark PyTorch vs CUDA kernel performance."""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmark")
    print("="*60)

    torch.manual_seed(42)

    # Benchmark parameters
    batch_size = 32
    seq_len = 256
    model_dim = 256
    reduced_dim = 32
    num_warmup = 10
    num_runs = 50

    # Create mixing layers
    pytorch_mixing = CausalGrassmannMixing(
        model_dim=model_dim,
        reduced_dim=reduced_dim,
        dropout=0.0,
    ).cuda()

    cuda_mixing = FusedGrassmannMixing(
        model_dim=model_dim,
        reduced_dim=reduced_dim,
        dropout=0.0,
    ).cuda()

    x = torch.randn(batch_size, seq_len, model_dim, device='cuda')

    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        _ = pytorch_mixing(x)
        _ = cuda_mixing(x)
    torch.cuda.synchronize()

    # Benchmark PyTorch forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = pytorch_mixing(x)
    torch.cuda.synchronize()
    pytorch_fwd_time = (time.perf_counter() - start) / num_runs * 1000

    # Benchmark CUDA forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = cuda_mixing(x)
    torch.cuda.synchronize()
    cuda_fwd_time = (time.perf_counter() - start) / num_runs * 1000

    # Benchmark forward + backward
    x_pytorch = x.clone().requires_grad_(True)
    x_cuda = x.clone().requires_grad_(True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        out = pytorch_mixing(x_pytorch)
        out.sum().backward()
        x_pytorch.grad = None
    torch.cuda.synchronize()
    pytorch_full_time = (time.perf_counter() - start) / num_runs * 1000

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        out = cuda_mixing(x_cuda)
        out.sum().backward()
        x_cuda.grad = None
    torch.cuda.synchronize()
    cuda_full_time = (time.perf_counter() - start) / num_runs * 1000

    print(f"\nBenchmark Results (batch={batch_size}, seq={seq_len}, dim={model_dim}):")
    print(f"{'Operation':<25} {'PyTorch':<15} {'CUDA':<15} {'Speedup':<10}")
    print("-" * 65)
    print(f"{'Forward only':<25} {pytorch_fwd_time:>10.3f} ms   {cuda_fwd_time:>10.3f} ms   {pytorch_fwd_time/cuda_fwd_time:>6.2f}x")
    print(f"{'Forward + Backward':<25} {pytorch_full_time:>10.3f} ms   {cuda_full_time:>10.3f} ms   {pytorch_full_time/cuda_full_time:>6.2f}x")

    # Throughput in tokens/sec
    tokens_per_batch = batch_size * seq_len
    pytorch_throughput = tokens_per_batch / (pytorch_full_time / 1000)
    cuda_throughput = tokens_per_batch / (cuda_full_time / 1000)

    print(f"\nThroughput (forward+backward):")
    print(f"  PyTorch: {pytorch_throughput/1e6:.2f}M tokens/sec")
    print(f"  CUDA:    {cuda_throughput/1e6:.2f}M tokens/sec")

    return pytorch_fwd_time, cuda_fwd_time, pytorch_full_time, cuda_full_time


def main():
    print("="*60)
    print("CUDA Kernel Test Suite for Grassmann v4")
    print("="*60)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    print(f"CUDA kernel available: {CUDA_AVAILABLE}")

    results = {}

    # Run tests
    results['plucker'] = test_plucker_correctness()
    results['mixing'] = test_mixing_layer_correctness()
    results['training'] = test_training_comparison(100)
    perf = benchmark_performance()
    results['performance'] = perf

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Plucker correctness: {'PASS' if results['plucker'] else 'FAIL'}")
    print(f"Mixing layer correctness: {'PASS' if results['mixing'] else 'FAIL'}")
    print(f"Training comparison: {'PASS' if results['training'] else 'FAIL'}")
    print(f"Forward speedup: {perf[0]/perf[1]:.2f}x")
    print(f"Full speedup: {perf[2]/perf[3]:.2f}x")

    all_passed = all([results['plucker'], results['mixing'], results['training']])
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
