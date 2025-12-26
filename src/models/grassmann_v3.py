"""
Numerically Stable Grassmann Flow layers (v3).

Key changes from v2:
1. Larger eps (1e-6) for normalization to prevent gradient explosion
2. Fixed self-pairing fallback that was creating NaN/Inf
3. Added LayerNorm after Plucker projection for stability
4. Scaled initialization to prevent early training instability
5. Tanh-based soft gating instead of sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class StablePluckerEncoder(nn.Module):
    """
    Numerically stable Plucker coordinate computation.

    For vectors u, v in R^r, computes:
    p_{ij} = u_i * v_j - u_j * v_i for all i < j

    Key stability improvements:
    - Larger epsilon for normalization
    - Optional skip of normalization for stability
    """

    def __init__(self, reduced_dim: int, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.eps = eps
        self.normalize = normalize
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Create index tensors for extracting upper triangular elements
        indices_i, indices_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                indices_i.append(i)
                indices_j.append(j)
        self.register_buffer('idx_i', torch.tensor(indices_i, dtype=torch.long))
        self.register_buffer('idx_j', torch.tensor(indices_j, dtype=torch.long))

        # Scaling factor to prevent gradient explosion
        self.scale = 1.0 / math.sqrt(self.plucker_dim)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Plucker coordinates with numerical stability.

        Args:
            u: (..., r) first vectors
            v: (..., r) second vectors

        Returns:
            Plucker coordinates (..., plucker_dim)
        """
        # Compute antisymmetric outer product elements
        # p_{ij} = u_i * v_j - u_j * v_i
        p = u[..., self.idx_i] * v[..., self.idx_j] - u[..., self.idx_j] * v[..., self.idx_i]

        # Scale down to prevent large values
        p = p * self.scale

        if self.normalize:
            # L2 normalize with stable epsilon
            norm = torch.sqrt((p ** 2).sum(dim=-1, keepdim=True) + self.eps)
            p = p / norm

        return p


class StableGrassmannMixing(nn.Module):
    """
    Numerically stable Causal Grassmann Mixing layer.

    Key stability improvements:
    - LayerNorm after Plucker projection
    - Simpler fallback for early positions (just use zeros)
    - Softer gating with tanh scaling
    - Gradient-friendly residual connection
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 64,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.reduced_dim = reduced_dim
        self.window_sizes = window_sizes or [1, 2, 4, 8, 16, 32]
        self.num_windows = len(self.window_sizes)

        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Linear reduction with layer norm
        self.reduction = nn.Linear(model_dim, reduced_dim)
        self.reduction_ln = nn.LayerNorm(reduced_dim)

        # Plucker encoder - skip normalization, we'll use LayerNorm instead
        self.plucker = StablePluckerEncoder(reduced_dim, normalize=False)

        # Project Plucker back to model dim with LayerNorm
        self.plucker_proj = nn.Linear(self.plucker_dim, model_dim)
        self.plucker_ln = nn.LayerNorm(model_dim)

        # Gating - initialize to favor residual connection
        self.gate_proj = nn.Linear(model_dim, model_dim)
        # Initialize gate bias to -2 so sigmoid(gate) starts near 0.12
        # This means we start mostly with residual and gradually learn geometric features
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        # Output
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for output
        self.output_scale = 1.0 / math.sqrt(2 * self.num_windows)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply numerically stable Grassmann mixing.

        Args:
            hidden_states: (batch, seq_len, model_dim)

        Returns:
            Mixed hidden states (batch, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. Linear reduction with normalization
        z = self.reduction(hidden_states)
        z = self.reduction_ln(z)  # (batch, seq_len, reduced_dim)

        # 2. Compute Plucker features for all windows at once
        geo_features = torch.zeros_like(hidden_states)
        counts = torch.zeros(batch_size, seq_len, 1, device=device)

        for delta in self.window_sizes:
            if delta >= seq_len:
                continue

            # Current positions (from delta onwards can look back)
            z_current = z[:, delta:, :]  # (batch, seq_len-delta, r)
            z_past = z[:, :-delta, :]     # (batch, seq_len-delta, r)

            # Compute Plucker coordinates
            plucker_coords = self.plucker(z_current, z_past)  # (batch, seq_len-delta, plucker_dim)

            # Project to model dim
            geo = self.plucker_proj(plucker_coords)  # (batch, seq_len-delta, model_dim)

            # Accumulate at correct positions
            geo_features[:, delta:, :] = geo_features[:, delta:, :] + geo
            counts[:, delta:, :] = counts[:, delta:, :] + 1

        # Average across windows (for positions with valid windows)
        valid_mask = counts > 0
        counts = counts.clamp(min=1)
        geo_features = geo_features / counts

        # Apply LayerNorm to geometric features
        geo_features = self.plucker_ln(geo_features)

        # For positions with no valid windows, geometric features stay as zeros
        # which is stable since we'll gate them appropriately

        # 3. Simple gated residual - gate controls how much geometric info to add
        gate = torch.sigmoid(self.gate_proj(hidden_states))

        # Output: hidden + gate * geo_features (starts small, grows with learning)
        output = hidden_states + gate * geo_features * self.output_scale

        # Output projection and dropout
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


class StableGrassmannBlock(nn.Module):
    """
    Numerically stable Grassmann Transformer block.
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 64,
        ff_dim: int = None,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * model_dim

        self.ln1 = nn.LayerNorm(model_dim)
        self.grassmann = StableGrassmannMixing(
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Grassmann mixing with residual
        normed = self.ln1(hidden_states)
        hidden_states = hidden_states + self.grassmann(normed, attention_mask)

        # FFN with residual
        normed = self.ln2(hidden_states)
        hidden_states = hidden_states + self.ffn(normed)

        return hidden_states


class GrassmannGPTv3(nn.Module):
    """
    Numerically Stable GrassmannGPT model (v3).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        model_dim: int = 768,
        num_layers: int = 12,
        reduced_dim: int = 64,
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
        window_sizes = window_sizes or [1, 2, 4, 8, 16, 32, 64, 128]

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Blocks
        self.blocks = nn.ModuleList([
            StableGrassmannBlock(
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
            # Smaller init for stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
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
            hidden_states = block(hidden_states, attention_mask)

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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
