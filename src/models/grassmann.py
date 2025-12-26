"""
Grassmann Flow layers for attention-free sequence modeling.

Based on: "Attention Is Not What You Need: Grassmann Flows as an
Attention-Free Alternative for Sequence Modeling" (Zhang, 2024)

Core idea: Replace self-attention with Grassmann manifold operations:
1. Reduce hidden states to low-dimensional space
2. Encode local token pairs as 2D subspaces on Gr(2,r) via Plucker coordinates
3. Project geometric features back and fuse with hidden states via gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class PluckerEncoder(nn.Module):
    """
    Computes Plucker coordinates for pairs of vectors.

    Given two vectors u, v in R^r, the Plucker coordinates are:
    p_{ij} = u_i * v_j - u_j * v_i for 1 <= i < j <= r

    This encodes the 2D subspace spanned by {u, v} as a point on Gr(2,r).
    """

    def __init__(self, reduced_dim: int, eps: float = 1e-8):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.eps = eps
        # Number of Plucker coordinates: C(r, 2) = r*(r-1)/2
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Precompute index pairs for vectorized Plucker computation
        indices_i = []
        indices_j = []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                indices_i.append(i)
                indices_j.append(j)
        self.register_buffer('indices_i', torch.tensor(indices_i, dtype=torch.long))
        self.register_buffer('indices_j', torch.tensor(indices_j, dtype=torch.long))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized Plucker coordinates for pairs (u, v).

        Args:
            u: Tensor of shape (..., r) - first vectors
            v: Tensor of shape (..., r) - second vectors

        Returns:
            Normalized Plucker coordinates of shape (..., plucker_dim)
        """
        # p_{ij} = u_i * v_j - u_j * v_i
        u_i = u[..., self.indices_i]  # (..., plucker_dim)
        u_j = u[..., self.indices_j]
        v_i = v[..., self.indices_i]
        v_j = v[..., self.indices_j]

        plucker = u_i * v_j - u_j * v_i  # (..., plucker_dim)

        # Normalize for numerical stability
        norm = torch.norm(plucker, dim=-1, keepdim=True)
        plucker = plucker / torch.clamp(norm, min=self.eps)

        return plucker


class CausalGrassmannMixing(nn.Module):
    """
    Causal Grassmann Mixing layer - replaces self-attention.

    Operations:
    1. Linear reduction: h -> z (d -> r)
    2. Multi-scale local pairing with causal offsets
    3. Plucker encoding of pairs
    4. Project back to model dimension
    5. Aggregate across offsets
    6. Gated fusion with original hidden state
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
        self.dropout_p = dropout

        # Plucker dimension
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Linear reduction: d -> r
        self.reduction = nn.Linear(model_dim, reduced_dim)

        # Plucker encoder
        self.plucker = PluckerEncoder(reduced_dim)

        # Project Plucker coordinates back to model dimension
        self.plucker_proj = nn.Linear(self.plucker_dim, model_dim)

        # Gating mechanism
        self.gate_proj = nn.Linear(2 * model_dim, model_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Grassmann mixing.

        Args:
            hidden_states: (batch, seq_len, model_dim)
            attention_mask: Optional mask (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            Updated hidden states (batch, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. Linear reduction
        z = self.reduction(hidden_states)  # (batch, seq_len, reduced_dim)

        # 2-4. Compute Plucker features for each position across all valid offsets
        # We'll accumulate geometric features for each position
        geo_features = torch.zeros_like(hidden_states)  # (batch, seq_len, model_dim)
        offset_counts = torch.zeros(batch_size, seq_len, 1, device=device)

        for delta in self.window_sizes:
            if delta >= seq_len:
                continue

            # For causal modeling: position t pairs with t+delta (looking ahead)
            # But for language modeling, we typically want causal = looking back
            # The paper uses forward pairing, but we'll use backward for causal LM

            # Positions that can look back by delta
            valid_positions = seq_len - delta
            if valid_positions <= 0:
                continue

            # z_t pairs with z_{t-delta} (causal: looking at past)
            z_current = z[:, delta:, :]  # (batch, valid_positions, r)
            z_past = z[:, :valid_positions, :]  # (batch, valid_positions, r)

            # Compute Plucker coordinates
            plucker_coords = self.plucker(z_current, z_past)  # (batch, valid_positions, plucker_dim)

            # Project to model dimension
            geo_feat = self.plucker_proj(plucker_coords)  # (batch, valid_positions, model_dim)

            # Accumulate at the correct positions (starting from position delta)
            geo_features[:, delta:, :] += geo_feat
            offset_counts[:, delta:, :] += 1

        # Handle positions with no valid offsets (beginning of sequence)
        # Use the hidden state itself as the geometric feature
        no_offset_mask = (offset_counts == 0)
        offset_counts = torch.clamp(offset_counts, min=1)

        # 5. Average across offsets
        geo_features = geo_features / offset_counts

        # For positions with no offsets, use projected hidden state
        geo_features = torch.where(
            no_offset_mask.expand_as(geo_features),
            self.plucker_proj(self.plucker(z, z)),  # Self-pairing fallback
            geo_features
        )

        # 6. Gated fusion
        # u = [h; g]
        concat = torch.cat([hidden_states, geo_features], dim=-1)  # (batch, seq_len, 2*model_dim)

        # alpha = sigmoid(W_gate * u)
        gate = torch.sigmoid(self.gate_proj(concat))  # (batch, seq_len, model_dim)

        # mixed = alpha * h + (1 - alpha) * g
        mixed = gate * hidden_states + (1 - gate) * geo_features

        # Layer norm and dropout
        mixed = self.layer_norm(mixed)
        mixed = self.dropout(mixed)

        return mixed


class GrassmannBlock(nn.Module):
    """
    A complete Grassmann Transformer block.

    Equivalent to a Transformer block but with Grassmann mixing instead of attention.
    Structure: GrassmannMixing -> Residual -> FFN -> Residual
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 64,
        ff_dim: int = None,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.model_dim = model_dim
        ff_dim = ff_dim or 4 * model_dim

        # Grassmann mixing (replaces attention)
        self.grassmann_mixing = CausalGrassmannMixing(
            model_dim=model_dim,
            reduced_dim=reduced_dim,
            window_sizes=window_sizes,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture (like GPT-2)
        # Grassmann mixing with residual
        normed = self.ln1(hidden_states)
        mixed = self.grassmann_mixing(normed, attention_mask)
        hidden_states = hidden_states + mixed

        # FFN with residual
        normed = self.ln2(hidden_states)
        ff_out = self.ffn(normed)
        hidden_states = hidden_states + ff_out

        return hidden_states


class GrassmannGPT(nn.Module):
    """
    GPT-2 style language model using Grassmann flows instead of attention.

    Architecture:
    - Token + Position embeddings
    - N x GrassmannBlock
    - Final LayerNorm
    - LM Head (tied weights with embedding)
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
        self.num_layers = num_layers

        ff_dim = ff_dim or 4 * model_dim
        window_sizes = window_sizes or [1, 2, 4, 8, 16, 32, 64, 128]

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Grassmann blocks
        self.blocks = nn.ModuleList([
            GrassmannBlock(
                model_dim=model_dim,
                reduced_dim=reduced_dim,
                ff_dim=ff_dim,
                window_sizes=window_sizes,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(model_dim)

        # LM head
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Tie weights
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
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
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: Optional (batch, seq_len) mask
            labels: Optional (batch, seq_len) for computing loss

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Optional cross-entropy loss if labels provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        tok_emb = self.token_embedding(input_ids)  # (batch, seq_len, model_dim)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, model_dim)

        # Combined embeddings
        hidden_states = self.embedding_dropout(tok_emb + pos_emb)

        # Grassmann blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            idx_cond = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
