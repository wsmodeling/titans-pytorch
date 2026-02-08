"""
KDA (Kimi Dynamic Attention) based memory module
Replaces the Neural Memory module with linear attention mechanism
"""

from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter, Linear
import torch.nn.functional as F
from einops import rearrange, repeat

# KDA state
KDAState = namedtuple('KDAState', [
    'seq_index',
    'hidden_state',  # The recurrent state S
])

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# Naive recurrent KDA implementation
def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """
    Recurrent KDA processing token by token

    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        g: gates (log-space) [B, T, H, K]
        beta: beta modulation [B, T, H, K]
        scale: query scaling factor
        initial_state: initial state [B, H, K, V]
        output_final_state: whether to return final state

    Returns:
        o: output [B, T, H, V]
        S: final state [B, H, K, V] or None
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)

    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        # Exponential decay
        S = S * g_i[..., None].exp()
        # Rank-one update: S += beta * k * (v - k^T S)
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        # Query the state
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """
    Chunked KDA for efficient parallel processing

    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        g: gates (log-space) [B, T, H, K]
        beta: beta modulation [B, T, H, K]
        scale: query scaling factor
        initial_state: initial state [B, H, K, V]
        output_final_state: whether to return final state
        chunk_size: size of chunks for processing

    Returns:
        o: output [B, T, H, V]
        S: final state [B, H, K, V] or None
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0, f"Sequence length {T} must be divisible by chunk_size {BT}"

    q, k, v, g, beta = map(lambda x: rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float), [q, k, v, g, beta])
    q = q * scale
    g = g.cumsum(-2)

    # Causal mask (diagonal is masked)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # Compute A matrix for within-chunk interactions
    A = torch.zeros(*q.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    # Aggregated keys and values
    w = A @ (g.exp() * k)
    u = A @ v

    # Initialize state
    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)

    for i in range(0, NT):
        # [B, H, BT, ...]
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        A = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            A[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        A = A.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + A @ v_i
        # Update state
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i

    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


class KDAMemory(Module):
    """
    KDA-based memory module to replace NeuralMemory

    Similar interface to NeuralMemory but uses linear attention mechanism
    """
    def __init__(
        self,
        dim,
        dim_head = None,
        heads = 1,
        chunk_size = 64,
        use_chunk = True,  # Use chunked version (faster) vs recurrent (slower but more flexible)
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        activation: Module | None = None,
    ):
        super().__init__()
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.heads = heads
        self.chunk_size = chunk_size
        self.use_chunk = use_chunk

        # Norms
        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        # Multi-headed projections
        dim_inner = dim_head * heads

        self.split_heads = nn.Identity() if heads == 1 else lambda x: rearrange(x, 'b n (h d) -> b n h d', h=heads)
        self.merge_heads = nn.Identity() if heads == 1 else lambda x: rearrange(x, 'b n h d -> b n (h d)')

        # Q, K, V projections
        self.to_queries = nn.Sequential(
            Linear(dim, dim_inner, bias=False),
            activation if activation else nn.Identity()
        )
        self.to_keys = nn.Sequential(
            Linear(dim, dim_inner, bias=False),
            activation if activation else nn.Identity()
        )
        self.to_values = nn.Sequential(
            Linear(dim, dim_inner, bias=False),
            activation if activation else nn.Identity()
        )

        # Gate and beta projections
        self.to_gates = Linear(dim, dim_inner, bias=False)
        self.to_beta = Linear(dim, heads, bias=False)  # Beta is per-head scalar!

        # Output projection
        self.to_out = Linear(dim_inner, dim, bias=False) if heads > 1 else nn.Identity()

        self.register_buffer('zero', torch.tensor(0.), persistent=False)

    def forward(
        self,
        seq,
        state: KDAState | None = None,
        return_state = True,
    ):
        """
        Forward pass of KDA memory

        Args:
            seq: input sequence [batch, seq_len, dim]
            state: previous KDA state (for inference)
            return_state: whether to return new state

        Returns:
            retrieved: output [batch, seq_len, dim]
            new_state: KDAState or None
        """
        batch, seq_len = seq.shape[:2]

        # Handle state
        if not exists(state):
            state = KDAState(0, None)

        seq_index, hidden_state = state

        # Norm
        seq = self.store_norm(seq)

        # Project to Q, K, V
        queries = self.to_queries(seq)
        keys = self.to_keys(seq)
        values = self.to_values(seq)

        # Gates and beta
        gates = self.to_gates(seq)
        beta = self.to_beta(seq).sigmoid()  # [B, N, H] - per-head scalar!

        # Split heads: [B, N, (H D)] -> [B, N, H, D]
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        gates = self.split_heads(gates)
        # Beta is already [B, N, H] - correct shape!

        # Apply norms
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Call KDA function
        # KDA expects [B, T, H, D] where last dim is the key/value dim
        if self.use_chunk and seq_len % self.chunk_size == 0:
            retrieved, new_hidden_state = naive_chunk_kda(
                q=queries,     # [B, T, H, D]
                k=keys,        # [B, T, H, D]
                v=values,      # [B, T, H, D]
                g=gates,       # [B, T, H, D]
                beta=beta,     # [B, T, H, D]
                initial_state=hidden_state,
                output_final_state=return_state,
                chunk_size=self.chunk_size
            )
        else:
            retrieved, new_hidden_state = naive_recurrent_kda(
                q=queries,     # [B, T, H, D]
                k=keys,        # [B, T, H, D]
                v=values,      # [B, T, H, D]
                g=gates,       # [B, T, H, D]
                beta=beta,     # [B, T, H, D]
                initial_state=hidden_state,
                output_final_state=return_state,
            )

        # retrieved is now [B, T, H, D]
        # Post norm
        retrieved = self.multihead_rmsnorm(retrieved)

        # Merge heads
        retrieved = self.merge_heads(retrieved)

        # Output projection
        retrieved = self.to_out(retrieved)

        # Create new state
        if return_state:
            new_state = KDAState(seq_index + seq_len, new_hidden_state)
        else:
            new_state = None

        return retrieved, new_state


class MultiheadRMSNorm(Module):
    """Multi-head RMSNorm (copied from neural_memory.py)"""
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine=False)
        self.gamma = Parameter(torch.zeros(1, 1, heads, dim))  # [1, 1, H, D]

    def forward(self, x):
        # x shape: [B, N, H, D]
        return self.rmsnorm(x) * (self.gamma + 1.)


def create_kda_memory_for_mac(
    dim,
    chunk_size=64,
    use_chunk=True,
    qk_rmsnorm=True,
    **kwargs
):
    """
    Factory function to create KDAMemory compatible with MAC Transformer

    This is a convenience function that matches the interface of creating
    a memory model for NeuralMemory (like MemoryMLP).

    Args:
        dim: Model dimension (will be used as both dim and dim_head)
        chunk_size: Chunk size for KDA processing
        use_chunk: Whether to use chunked KDA
        qk_rmsnorm: Whether to use QK RMSNorm
        **kwargs: Additional arguments passed to KDAMemory

    Returns:
        KDAMemory instance configured for MAC Transformer
    """
    return KDAMemory(
        dim=dim,
        dim_head=dim,  # Use same dimension for heads
        heads=4,       # Default 4 heads
        chunk_size=chunk_size,
        use_chunk=use_chunk,
        pre_rmsnorm=True,
        qk_rmsnorm=qk_rmsnorm,
        **kwargs
    )
