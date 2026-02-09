"""
KDA (Kimi Delta Attention) based memory module.

Closely follows FLA (Flash Linear Attention) KimiDeltaAttention:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py
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
    'hidden_state',  # The recurrent state S [B, H, K, V]
])

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d


# ---------------------------------------------------------------------------
# ShortConvolution — causal depthwise conv1d + activation
# (replaces fla.modules.ShortConvolution without Triton dependency)
# ---------------------------------------------------------------------------

class ShortConvolution(Module):
    """Causal depthwise 1D convolution with optional activation."""

    def __init__(self, hidden_size, kernel_size=4, bias=False, activation='silu'):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal: left-pad
            groups=hidden_size,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.activation = dict(silu=F.silu, swish=F.silu).get(activation)

    def forward(self, x):
        # x: [B, T, D]
        x = rearrange(x, 'b t d -> b d t')
        x = self.conv(x)[..., :x.shape[-1]]  # trim right-pad for causal
        x = rearrange(x, 'b d t -> b t d')
        if self.activation is not None:
            x = self.activation(x)
        return x


# ---------------------------------------------------------------------------
# Gate computation (from fla/ops/kda/gate.py — naive_kda_gate)
# ---------------------------------------------------------------------------

def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KDA gate: g = -exp(A_log) * softplus(g + dt_bias)

    Always <= 0, so exp(g) in (0, 1] — stable contractive decay.

    Args:
        g: gate projections [..., H, K]
        A_log: learnable per-head log-scale [H]
        dt_bias: learnable per-channel bias [H*K]
    Returns:
        g: gated values [..., H, K], always <= 0
    """
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return -A_log.view(H, 1).float().exp() * F.softplus(g)


# ---------------------------------------------------------------------------
# Naive recurrent KDA (from fla/ops/kda/naive.py — verbatim)
# ---------------------------------------------------------------------------

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

    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

    if not output_final_state:
        S = None
    return o.to(dtype), S


# ---------------------------------------------------------------------------
# Naive chunked KDA (from fla/ops/kda/naive.py — verbatim)
# ---------------------------------------------------------------------------

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

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

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

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)

    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        A = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            A[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        A = A.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + A @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i

    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


# ---------------------------------------------------------------------------
# KDAMemory — follows FLA's KimiDeltaAttention as closely as possible
# ---------------------------------------------------------------------------

class KDAMemory(Module):
    """
    Kimi Delta Attention (KDA) memory module.

    Follows FLA's KimiDeltaAttention architecture:
    - Q, K, V: Linear + ShortConvolution(causal conv1d + SiLU) or Linear + SiLU
    - Gate (f_proj): low-rank projection + naive_kda_gate
    - A_log: learnable per-head decay scale, init log(uniform(1, 16))
    - dt_bias: learnable per-channel gate bias
    - Beta (b_proj): per-head sigmoid
    - Q/K L2 normalization (use_qk_l2norm_in_kernel)
    - Output: FusedRMSNormGated(sigmoid) = RMSNorm(o) * sigmoid(g_proj(x))
    - o_proj: final linear

    Reference: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py
    """
    def __init__(
        self,
        dim,
        dim_head = None,
        heads = 4,
        chunk_size = 64,
        use_chunk = True,
        use_short_conv = True,
        conv_size = 4,
        conv_bias = False,
        allow_neg_eigval = False,
        norm_eps = 1e-5,
    ):
        super().__init__()
        dim_head = default(dim_head, dim // heads)

        self.heads = heads
        self.dim_head = dim_head
        self.chunk_size = chunk_size
        self.use_chunk = use_chunk
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval

        head_k_dim = dim_head
        head_v_dim = dim_head
        key_dim = heads * head_k_dim
        value_dim = heads * head_v_dim

        # Input norm
        self.norm = nn.RMSNorm(dim)

        # Q, K, V projections
        self.q_proj = Linear(dim, key_dim, bias=False)
        self.k_proj = Linear(dim, key_dim, bias=False)
        self.v_proj = Linear(dim, value_dim, bias=False)

        # Optional short convolution (FLA default: use_short_conv=True)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')

        # Gate projection: low-rank (FLA: hidden_size -> head_v_dim -> key_dim)
        self.f_proj = nn.Sequential(
            Linear(dim, head_v_dim, bias=False),
            Linear(head_v_dim, key_dim, bias=False),
        )

        # A_log: learnable per-head decay base, init as log(uniform(1, 16))
        self.A_log = Parameter(torch.log(torch.empty(heads, dtype=torch.float32).uniform_(1, 16)))
        # dt_bias: learnable per-channel bias for gates
        self.dt_bias = Parameter(torch.zeros(key_dim, dtype=torch.float32))

        # Beta: per-head sigmoid
        self.b_proj = Linear(dim, heads, bias=False)

        # Output gate: low-rank (FLA: hidden_size -> head_v_dim -> value_dim, with bias)
        self.g_proj = nn.Sequential(
            Linear(dim, head_v_dim, bias=False),
            Linear(head_v_dim, value_dim, bias=True),
        )
        # FusedRMSNormGated(head_v_dim, activation='sigmoid') equivalent
        self.o_norm = nn.RMSNorm(head_v_dim, eps=norm_eps)
        self.o_proj = Linear(value_dim, dim, bias=False)

        self.register_buffer('zero', torch.tensor(0.), persistent=False)

    def forward(
        self,
        seq,
        state: KDAState | None = None,
        return_state = True,
    ):
        """
        Args:
            seq: [batch, seq_len, dim]
            state: previous KDAState (for inference)
            return_state: whether to return new state
        Returns:
            output: [batch, seq_len, dim]
            new_state: KDAState or None
        """
        batch, seq_len = seq.shape[:2]

        if not exists(state):
            state = KDAState(0, None)
        seq_index, hidden_state = state

        # Input norm
        normed = self.norm(seq)

        # Q, K, V
        if self.use_short_conv:
            q = self.q_conv1d(self.q_proj(normed))
            k = self.k_conv1d(self.k_proj(normed))
            v = self.v_conv1d(self.v_proj(normed))
        else:
            q = F.silu(self.q_proj(normed))
            k = F.silu(self.k_proj(normed))
            v = F.silu(self.v_proj(normed))

        # Gate: low-rank projection
        g = self.f_proj(normed)
        # Beta: per-head sigmoid
        beta = self.b_proj(normed).sigmoid()  # [B, T, H]

        # Split heads: [B, T, (H D)] -> [B, T, H, D]
        q, k, g = (rearrange(x, 'b t (h d) -> b t h d', h=self.heads) for x in (q, k, g))
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.heads)

        # Allow negative eigenvalues (FLA option)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Apply KDA gate: g = -exp(A_log) * softplus(g + dt_bias), always <= 0
        g = naive_kda_gate(g, self.A_log, self.dt_bias)

        # L2-normalize Q and K (use_qk_l2norm_in_kernel=True in FLA)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Run KDA kernel
        if self.use_chunk and seq_len % self.chunk_size == 0:
            o, new_hidden_state = naive_chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=hidden_state,
                output_final_state=return_state,
                chunk_size=self.chunk_size,
            )
        else:
            o, new_hidden_state = naive_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=hidden_state,
                output_final_state=return_state,
            )

        # Output: FusedRMSNormGated(head_v_dim, activation='sigmoid')
        # = RMSNorm(o) * sigmoid(g_proj(x))
        gate = rearrange(self.g_proj(normed), 'b t (h d) -> b t h d', h=self.heads).sigmoid()
        o = self.o_norm(o) * gate

        # Merge heads and output projection
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if return_state:
            new_state = KDAState(seq_index + seq_len, new_hidden_state)
        else:
            new_state = None

        return o, new_state


# ---------------------------------------------------------------------------
# MultiheadRMSNorm (kept for mac_transformer import compatibility)
# ---------------------------------------------------------------------------

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine=False)
        self.gamma = Parameter(torch.zeros(1, 1, heads, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)


# ---------------------------------------------------------------------------
# Factory function for MAC Transformer
# ---------------------------------------------------------------------------

def create_kda_memory_for_mac(
    dim,
    chunk_size=64,
    use_chunk=True,
    heads=4,
    use_short_conv=True,
    **kwargs
):
    """
    Create KDAMemory configured for MAC Transformer.

    Args:
        dim: Model dimension (template dim; MAC recreates with transformer dim)
        chunk_size: Chunk size for KDA processing
        use_chunk: Whether to use chunked KDA
        heads: Number of attention heads
        use_short_conv: Whether to use causal short convolution on Q, K, V
    Returns:
        KDAMemory instance
    """
    return KDAMemory(
        dim=dim,
        heads=heads,
        chunk_size=chunk_size,
        use_chunk=use_chunk,
        use_short_conv=use_short_conv,
        **kwargs
    )
