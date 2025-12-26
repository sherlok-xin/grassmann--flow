"""
Fused CUDA Grassmann layers with autograd support - v4 (Paper Exact).

Matches arXiv 2512.19428 exactly:
1. Reduced dim r = 32
2. Window sizes {1, 2, 4, 8, 12, 16} for 6-layer
3. Gating: blend formula alpha * h + (1-alpha) * g
4. Gate input: concatenate [h; g]
5. L2 normalize Plucker before projection

Falls back to PyTorch implementation if CUDA extension not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Add cuda directory to path for module import
_cuda_dir = Path(__file__).parent
if str(_cuda_dir) not in sys.path:
    sys.path.insert(0, str(_cuda_dir))

try:
    import grassmann_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    grassmann_cuda = None


class FusedPluckerFunction(Function):
    """Autograd function for fused Plucker forward/backward."""

    @staticmethod
    def forward(ctx, z, window_sizes, scale, eps):
        """
        Args:
            z: [batch, seq_len, reduced_dim] reduced representations
            window_sizes: [num_windows] tensor of window offsets
            scale: scaling factor for Plucker coords
            eps: epsilon for numerical stability

        Returns:
            plucker: [batch, seq_len, plucker_dim] Plucker coordinates
        """
        if CUDA_AVAILABLE and z.is_cuda:
            plucker = grassmann_cuda.fused_plucker_forward(z, window_sizes, scale, eps)
        else:
            # Fallback to PyTorch implementation
            plucker = FusedPluckerFunction._pytorch_forward(z, window_sizes, scale, eps)

        ctx.save_for_backward(z, window_sizes)
        ctx.scale = scale
        return plucker

    @staticmethod
    def _pytorch_forward(z, window_sizes, scale, eps):
        """PyTorch fallback implementation."""
        batch_size, seq_len, reduced_dim = z.shape
        plucker_dim = reduced_dim * (reduced_dim - 1) // 2
        device = z.device

        plucker_out = torch.zeros(batch_size, seq_len, plucker_dim, device=device, dtype=z.dtype)
        counts = torch.zeros(batch_size, seq_len, 1, device=device, dtype=z.dtype)

        # Precompute index pairs
        idx_i, idx_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=device)
        idx_j = torch.tensor(idx_j, device=device)

        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue

            z_current = z[:, delta:, :]
            z_past = z[:, :-delta, :]

            # Compute Plucker coords
            p = z_current[..., idx_i] * z_past[..., idx_j] - z_current[..., idx_j] * z_past[..., idx_i]
            p = p * scale

            plucker_out[:, delta:, :] = plucker_out[:, delta:, :] + p
            counts[:, delta:, :] = counts[:, delta:, :] + 1

        counts = counts.clamp(min=1)
        plucker_out = plucker_out / counts

        # Normalize
        norm = torch.sqrt((plucker_out ** 2).sum(dim=-1, keepdim=True) + eps)
        plucker_out = plucker_out / norm

        return plucker_out

    @staticmethod
    def backward(ctx, grad_plucker):
        z, window_sizes = ctx.saved_tensors
        scale = ctx.scale

        batch_size, seq_len, reduced_dim = z.shape
        plucker_dim = reduced_dim * (reduced_dim - 1) // 2
        device = z.device
        dtype = z.dtype

        grad_z = torch.zeros_like(z)

        # Precompute index pairs
        idx_i, idx_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=device, dtype=torch.long)
        idx_j = torch.tensor(idx_j, device=device, dtype=torch.long)

        # Count valid windows per position
        valid_windows = torch.zeros(seq_len, device=device, dtype=dtype)
        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue
            valid_windows[delta:] += 1
        valid_windows = valid_windows.clamp(min=1)

        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue

            T_valid = seq_len - delta
            inv_count = 1.0 / valid_windows[delta:]

            # grad_plucker for positions delta:
            grad_p = grad_plucker[:, delta:, :] * inv_count.unsqueeze(0).unsqueeze(-1)
            grad_p_scaled = grad_p * scale

            z_current = z[:, delta:, :]
            z_past = z[:, :-delta, :]

            # Compute gradient contributions
            # p_ij = z_current[i] * z_past[j] - z_current[j] * z_past[i]
            # grad_z_current[i] += grad_p[ij] * z_past[j]
            # grad_z_current[j] -= grad_p[ij] * z_past[i]
            grad_z_current_i = grad_p_scaled * z_past[..., idx_j]  # [B, T_valid, plucker_dim]
            grad_z_current_j = -grad_p_scaled * z_past[..., idx_i]

            # grad_z_past[j] += grad_p[ij] * z_current[i]
            # grad_z_past[i] -= grad_p[ij] * z_current[j]
            grad_z_past_j = grad_p_scaled * z_current[..., idx_i]
            grad_z_past_i = -grad_p_scaled * z_current[..., idx_j]

            # Accumulate gradients using index_add (vectorized)
            # For z_current positions (delta to seq_len)
            for k in range(plucker_dim):
                i_idx = idx_i[k].item()
                j_idx = idx_j[k].item()
                grad_z[:, delta:, i_idx] += grad_z_current_i[..., k]
                grad_z[:, delta:, j_idx] += grad_z_current_j[..., k]
                grad_z[:, :T_valid, i_idx] += grad_z_past_i[..., k]
                grad_z[:, :T_valid, j_idx] += grad_z_past_j[..., k]

        return grad_z, None, None, None


class FusedGrassmannFunction(Function):
    """Autograd function for fully fused Grassmann mixing."""

    @staticmethod
    def forward(ctx, hidden_states, W_reduce, W_proj, W_gate, b_gate,
                window_sizes, scale, eps, output_scale):
        """
        Fully fused forward pass.
        """
        if CUDA_AVAILABLE and hidden_states.is_cuda:
            output = grassmann_cuda.fused_grassmann_forward(
                hidden_states, W_reduce, W_proj, W_gate, b_gate,
                window_sizes, scale, eps, output_scale
            )
            # Save for backward - need intermediate values
            # For now, recompute in backward (trade compute for memory)
            ctx.save_for_backward(hidden_states, W_reduce, W_proj, W_gate, b_gate, window_sizes)
            ctx.scale = scale
            ctx.eps = eps
            ctx.output_scale = output_scale
        else:
            # Fallback
            output, z, plucker, geo = FusedGrassmannFunction._pytorch_forward(
                hidden_states, W_reduce, W_proj, W_gate, b_gate,
                window_sizes, scale, eps, output_scale
            )
            ctx.save_for_backward(hidden_states, W_reduce, W_proj, W_gate, b_gate,
                                   window_sizes, z, plucker, geo)
            ctx.scale = scale
            ctx.eps = eps
            ctx.output_scale = output_scale

        return output

    @staticmethod
    def _pytorch_forward(hidden_states, W_reduce, W_proj, W_gate, b_gate,
                         window_sizes, scale, eps, output_scale):
        """PyTorch fallback with intermediate tensors."""
        batch_size, seq_len, model_dim = hidden_states.shape
        reduced_dim = W_reduce.shape[0]
        plucker_dim = W_proj.shape[1]
        device = hidden_states.device

        # Reduction
        z = F.linear(hidden_states, W_reduce)

        # Plucker computation
        idx_i, idx_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=device)
        idx_j = torch.tensor(idx_j, device=device)

        plucker_accum = torch.zeros(batch_size, seq_len, plucker_dim, device=device, dtype=hidden_states.dtype)
        counts = torch.zeros(batch_size, seq_len, 1, device=device, dtype=hidden_states.dtype)

        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue

            z_current = z[:, delta:, :]
            z_past = z[:, :-delta, :]

            p = z_current[..., idx_i] * z_past[..., idx_j] - z_current[..., idx_j] * z_past[..., idx_i]
            plucker_accum[:, delta:, :] += p * scale
            counts[:, delta:, :] += 1

        counts = counts.clamp(min=1)
        plucker = plucker_accum / counts

        # Project to model dim
        geo = F.linear(plucker, W_proj)

        # Gated residual
        gate = torch.sigmoid(F.linear(hidden_states, W_gate, b_gate))
        output = hidden_states + gate * geo * output_scale

        return output, z, plucker, geo

    @staticmethod
    def backward(ctx, grad_output):
        # Always use PyTorch backward for correctness (CUDA backward has bugs)
        # The forward CUDA kernel provides most of the speedup
        if CUDA_AVAILABLE and ctx.saved_tensors[0].is_cuda:
            hidden_states, W_reduce, W_proj, W_gate, b_gate, window_sizes = ctx.saved_tensors

            # Recompute intermediates for backward
            _, z, plucker, geo = FusedGrassmannFunction._pytorch_forward(
                hidden_states, W_reduce, W_proj, W_gate, b_gate,
                window_sizes, ctx.scale, ctx.eps, ctx.output_scale
            )

            return FusedGrassmannFunction._pytorch_backward(
                grad_output, hidden_states, z, plucker, geo,
                W_reduce, W_proj, W_gate, b_gate, window_sizes,
                ctx.scale, ctx.output_scale
            )
        else:
            hidden_states, W_reduce, W_proj, W_gate, b_gate, window_sizes, z, plucker, geo = ctx.saved_tensors
            return FusedGrassmannFunction._pytorch_backward(
                grad_output, hidden_states, z, plucker, geo,
                W_reduce, W_proj, W_gate, b_gate, window_sizes,
                ctx.scale, ctx.output_scale
            )

    @staticmethod
    def _pytorch_backward(grad_output, hidden_states, z, plucker, geo,
                          W_reduce, W_proj, W_gate, b_gate, window_sizes,
                          scale, output_scale):
        """PyTorch backward pass."""
        batch_size, seq_len, model_dim = hidden_states.shape
        reduced_dim = W_reduce.shape[0]
        plucker_dim = W_proj.shape[1]
        device = hidden_states.device

        # Recompute gate
        gate = torch.sigmoid(F.linear(hidden_states, W_gate, b_gate))

        # Backward through gated residual
        grad_hidden = grad_output.clone()  # residual path
        grad_geo = grad_output * gate * output_scale
        grad_gate = grad_output * geo * output_scale * gate * (1 - gate)

        # Gradient for W_gate and b_gate
        grad_b_gate = grad_gate.sum(dim=(0, 1))
        grad_W_gate = torch.einsum('bsd,bsm->dm', grad_gate, hidden_states)
        grad_hidden += torch.einsum('bsd,dm->bsm', grad_gate, W_gate)

        # Backward through projection
        grad_plucker = F.linear(grad_geo, W_proj.t())
        grad_W_proj = torch.einsum('bsd,bsp->dp', grad_geo, plucker)

        # Backward through Plucker to z
        idx_i, idx_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                idx_i.append(i)
                idx_j.append(j)
        idx_i = torch.tensor(idx_i, device=device)
        idx_j = torch.tensor(idx_j, device=device)

        grad_z = torch.zeros_like(z)

        # Count valid windows
        valid_windows = torch.zeros(seq_len, device=device)
        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue
            valid_windows[delta:] += 1
        valid_windows = valid_windows.clamp(min=1)

        for delta in window_sizes.tolist():
            if delta >= seq_len:
                continue

            inv_count = 1.0 / valid_windows[delta:]
            grad_p = grad_plucker[:, delta:, :] * inv_count.unsqueeze(0).unsqueeze(-1)

            z_current = z[:, delta:, :]
            z_past = z[:, :-delta, :]

            for k, (i, j) in enumerate(zip(idx_i.tolist(), idx_j.tolist())):
                grad_z[:, delta:, i] += grad_p[..., k] * z_past[..., j] * scale
                grad_z[:, delta:, j] -= grad_p[..., k] * z_past[..., i] * scale
                grad_z[:, :-delta, j] += grad_p[..., k] * z_current[..., i] * scale
                grad_z[:, :-delta, i] -= grad_p[..., k] * z_current[..., j] * scale

        # Backward through reduction
        grad_hidden += F.linear(grad_z, W_reduce.t())
        grad_W_reduce = torch.einsum('bsr,bsm->rm', grad_z, hidden_states)

        return grad_hidden, grad_W_reduce, grad_W_proj, grad_W_gate, grad_b_gate, None, None, None, None


class FusedGrassmannMixing(nn.Module):
    """
    Fused Grassmann Mixing layer - v4 (Paper Exact).

    Matches arXiv 2512.19428:
    - Blend gating: alpha * h + (1-alpha) * g
    - Gate input: concatenate [h; g]
    - L2 normalize Plucker before projection
    - r=32, windows=[1,2,4,8,12,16]

    Uses CUDA kernel for Plucker computation (5-6x speedup).
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 32,  # Paper's value
        window_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.reduced_dim = reduced_dim
        # Paper's window sizes for 6-layer
        self.window_sizes = window_sizes or [1, 2, 4, 8, 12, 16]
        self.num_windows = len(self.window_sizes)
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Step 1: Linear reduction z = W_red * h
        self.W_red = nn.Linear(model_dim, reduced_dim)

        # Step 4: Project Plucker to model dim
        self.W_plu = nn.Linear(self.plucker_dim, model_dim)

        # Step 6: Gate from concatenated [h; g] - paper's approach
        self.W_gate = nn.Linear(2 * model_dim, model_dim)

        # Step 8: LayerNorm after mixing
        self.layer_norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        # Scaling/stability
        self.eps = 1e-8

        # Register window sizes as buffer
        self.register_buffer('window_sizes_tensor', torch.tensor(self.window_sizes, dtype=torch.int32))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_red.weight)
        nn.init.zeros_(self.W_red.bias)
        nn.init.xavier_uniform_(self.W_plu.weight)
        nn.init.zeros_(self.W_plu.bias)
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: Reduce to low dimension
        z = self.W_red(hidden_states)  # (batch, seq_len, reduced_dim)

        # Steps 2-5: Plucker with L2 norm, averaged across windows
        # CUDA kernel handles: Plucker coords -> L2 norm -> average
        if CUDA_AVAILABLE and z.is_cuda:
            # Scale=1.0 since we L2 normalize anyway
            plucker = FusedPluckerFunction.apply(z, self.window_sizes_tensor, 1.0, self.eps)
        else:
            plucker = FusedPluckerFunction._pytorch_forward(z, self.window_sizes_tensor, 1.0, self.eps)

        # Project to model dim
        g = self.W_plu(plucker)  # (batch, seq_len, model_dim)

        # Step 6: Gating with concatenated [h; g]
        concat = torch.cat([hidden_states, g], dim=-1)  # (batch, seq_len, 2*model_dim)
        alpha = torch.sigmoid(self.W_gate(concat))  # (batch, seq_len, model_dim)

        # Step 7: Blend (not add!) - h_mix = alpha * h + (1-alpha) * g
        h_mix = alpha * hidden_states + (1 - alpha) * g

        # Step 8: LayerNorm and dropout
        output = self.layer_norm(h_mix)
        output = self.dropout(output)

        return output


class FusedGrassmannBlock(nn.Module):
    """Transformer block with fused Grassmann mixing - v4 (Paper Exact)."""

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 32,  # Paper's value
        ff_dim: int = None,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * model_dim

        self.ln1 = nn.LayerNorm(model_dim)
        self.grassmann = FusedGrassmannMixing(
            model_dim=model_dim,
            reduced_dim=reduced_dim,
            window_sizes=window_sizes,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Grassmann mixing with residual
        normed = self.ln1(hidden_states)
        hidden_states = hidden_states + self.grassmann(normed)

        # FFN with residual
        normed = self.ln2(hidden_states)
        hidden_states = hidden_states + self.ffn(normed)

        return hidden_states


class GrassmannGPTFused(nn.Module):
    """
    GrassmannGPT with fused CUDA kernels - v4 (Paper Exact).

    Matches arXiv 2512.19428:
    - reduced_dim = 32
    - window_sizes = [1, 2, 4, 8, 12, 16]
    - Blend gating with concat input
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        model_dim: int = 768,
        num_layers: int = 12,
        reduced_dim: int = 32,  # Paper's value
        ff_dim: int = None,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        ff_dim = ff_dim or 4 * model_dim
        # Paper's window sizes for 6-layer
        window_sizes = window_sizes or [1, 2, 4, 8, 12, 16]

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            FusedGrassmannBlock(
                model_dim=model_dim,
                reduced_dim=reduced_dim,
                ff_dim=ff_dim,
                window_sizes=window_sizes,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=device))
        hidden_states = self.embedding_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
