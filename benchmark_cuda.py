"""
Benchmark CUDA vs PyTorch Grassmann layer implementations.
"""

import torch
import time
import sys

# Add cuda module to path
sys.path.insert(0, 'src/cuda')

def benchmark_plucker(batch_size=32, seq_len=256, reduced_dim=64, num_iters=100, warmup=10):
    """Benchmark Plucker coordinate computation."""
    print(f"\n{'='*60}")
    print(f"Plucker Benchmark: batch={batch_size}, seq={seq_len}, r={reduced_dim}")
    print(f"{'='*60}")

    device = torch.device('cuda')
    plucker_dim = reduced_dim * (reduced_dim - 1) // 2

    # Create test data
    z = torch.randn(batch_size, seq_len, reduced_dim, device=device, requires_grad=True)
    window_sizes = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.int32, device=device)
    scale = 1.0 / (plucker_dim ** 0.5)
    eps = 1e-6

    # Precompute index pairs for PyTorch version
    idx_i, idx_j = [], []
    for i in range(reduced_dim):
        for j in range(i + 1, reduced_dim):
            idx_i.append(i)
            idx_j.append(j)
    idx_i = torch.tensor(idx_i, device=device)
    idx_j = torch.tensor(idx_j, device=device)

    def pytorch_plucker():
        """PyTorch implementation."""
        plucker_out = torch.zeros(batch_size, seq_len, plucker_dim, device=device, dtype=z.dtype)
        counts = torch.zeros(batch_size, seq_len, 1, device=device, dtype=z.dtype)

        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue
            z_current = z[:, delta:, :]
            z_past = z[:, :-delta, :]
            p = z_current[..., idx_i] * z_past[..., idx_j] - z_current[..., idx_j] * z_past[..., idx_i]
            p = p * scale
            plucker_out[:, delta:, :] = plucker_out[:, delta:, :] + p
            counts[:, delta:, :] = counts[:, delta:, :] + 1

        counts = counts.clamp(min=1)
        plucker_out = plucker_out / counts
        norm = torch.sqrt((plucker_out ** 2).sum(dim=-1, keepdim=True) + eps)
        return plucker_out / norm

    # Try CUDA version
    try:
        import grassmann_cuda
        has_cuda = True
        print("CUDA extension loaded successfully!")

        def cuda_plucker():
            return grassmann_cuda.fused_plucker_forward(z, window_sizes, scale, eps)

    except ImportError as e:
        has_cuda = False
        print(f"CUDA extension not available: {e}")
        cuda_plucker = None

    # Warmup
    print("\nWarming up...")
    for _ in range(warmup):
        _ = pytorch_plucker()
        if has_cuda:
            _ = cuda_plucker()
    torch.cuda.synchronize()

    # Benchmark PyTorch
    print("Benchmarking PyTorch...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        out_pytorch = pytorch_plucker()
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iters * 1000

    # Benchmark CUDA
    if has_cuda:
        print("Benchmarking CUDA...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            out_cuda = cuda_plucker()
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / num_iters * 1000

        # Check correctness
        diff = (out_pytorch - out_cuda).abs().max().item()
        print(f"\nMax difference: {diff:.2e}")

        print(f"\nResults:")
        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  CUDA:    {cuda_time:.3f} ms")
        print(f"  Speedup: {pytorch_time/cuda_time:.2f}x")
    else:
        print(f"\nPyTorch only: {pytorch_time:.3f} ms")


def benchmark_grassmann_layer(batch_size=32, seq_len=256, model_dim=512, reduced_dim=64,
                               num_iters=50, warmup=10):
    """Benchmark full Grassmann mixing layer."""
    print(f"\n{'='*60}")
    print(f"Grassmann Layer Benchmark: batch={batch_size}, seq={seq_len}, d={model_dim}")
    print(f"{'='*60}")

    device = torch.device('cuda')

    # Create layer
    from src.models.grassmann_v3 import StableGrassmannMixing

    layer_pytorch = StableGrassmannMixing(
        model_dim=model_dim,
        reduced_dim=reduced_dim,
        window_sizes=[1, 2, 4, 8, 16, 32],
        dropout=0.0,
    ).to(device).eval()

    # Try fused version
    try:
        from src.cuda.grassmann_fused import FusedGrassmannMixing
        layer_fused = FusedGrassmannMixing(
            model_dim=model_dim,
            reduced_dim=reduced_dim,
            window_sizes=[1, 2, 4, 8, 16, 32],
            dropout=0.0,
        ).to(device).eval()
        has_fused = True
        print("Fused layer created successfully!")
    except Exception as e:
        has_fused = False
        print(f"Fused layer not available: {e}")

    # Create test data
    x = torch.randn(batch_size, seq_len, model_dim, device=device)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer_pytorch(x)
            if has_fused:
                _ = layer_fused(x)
    torch.cuda.synchronize()

    # Benchmark PyTorch forward
    print("Benchmarking PyTorch forward...")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            out_pytorch = layer_pytorch(x)
        torch.cuda.synchronize()
        pytorch_fwd_time = (time.perf_counter() - start) / num_iters * 1000

    # Benchmark PyTorch forward+backward
    print("Benchmarking PyTorch forward+backward...")
    x_grad = x.clone().requires_grad_(True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        out = layer_pytorch(x_grad)
        loss = out.sum()
        loss.backward()
        x_grad.grad = None
    torch.cuda.synchronize()
    pytorch_total_time = (time.perf_counter() - start) / num_iters * 1000

    if has_fused:
        # Benchmark fused forward
        print("Benchmarking Fused forward...")
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iters):
                out_fused = layer_fused(x)
            torch.cuda.synchronize()
            fused_fwd_time = (time.perf_counter() - start) / num_iters * 1000

        # Benchmark fused forward+backward
        print("Benchmarking Fused forward+backward...")
        x_grad = x.clone().requires_grad_(True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            out = layer_fused(x_grad)
            loss = out.sum()
            loss.backward()
            x_grad.grad = None
        torch.cuda.synchronize()
        fused_total_time = (time.perf_counter() - start) / num_iters * 1000

        print(f"\nResults:")
        print(f"  Forward only:")
        print(f"    PyTorch: {pytorch_fwd_time:.3f} ms")
        print(f"    Fused:   {fused_fwd_time:.3f} ms")
        print(f"    Speedup: {pytorch_fwd_time/fused_fwd_time:.2f}x")
        print(f"  Forward + Backward:")
        print(f"    PyTorch: {pytorch_total_time:.3f} ms")
        print(f"    Fused:   {fused_total_time:.3f} ms")
        print(f"    Speedup: {pytorch_total_time/fused_total_time:.2f}x")
    else:
        print(f"\nPyTorch only:")
        print(f"  Forward:  {pytorch_fwd_time:.3f} ms")
        print(f"  Fwd+Bwd:  {pytorch_total_time:.3f} ms")


if __name__ == "__main__":
    print("Grassmann CUDA Kernel Benchmark")
    print("=" * 60)

    # Run benchmarks
    benchmark_plucker()
    benchmark_grassmann_layer()
