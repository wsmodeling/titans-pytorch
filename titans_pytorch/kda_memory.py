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
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange, repeat

# KDA state
KDAState = namedtuple('KDAState', [
    'seq_index',
    'hidden_state',  # The recurrent state S [B, H, K, V]
])

# Sparse KDA state: hidden_state is [B, N, H, K, V] (N memory slots)
SparseKDAState = namedtuple('SparseKDAState', [
    'seq_index',
    'hidden_state',         # [B, N, H, K, V]
    'shared_hidden_state',  # [B, H, K, V] or None (only used when use_shared_memory=True)
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

@torch.compiler.disable
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

@torch.compiler.disable
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
# Chunked recurrent KDA with gradient checkpointing
# ---------------------------------------------------------------------------

@torch.compiler.disable
def _kda_recurrent_chunk(q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, S_in):
    """Process one chunk of the KDA recurrence (for gradient checkpointing)."""
    dtype = v_chunk.dtype
    B, T_c, H, K = q_chunk.shape
    V = v_chunk.shape[-1]

    q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk = map(
        lambda x: x.to(torch.float), [q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk]
    )
    S = S_in.to(torch.float)
    o_chunk = torch.zeros(B, T_c, H, V, dtype=torch.float, device=q_chunk.device)

    for i in range(T_c):
        q_i, k_i, v_i, g_i, b_i = q_chunk[:, i], k_chunk[:, i], v_chunk[:, i], g_chunk[:, i], beta_chunk[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o_chunk[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

    return o_chunk.to(dtype), S.to(dtype)


@torch.compiler.disable
def kda_recurrent_checkpointed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 32,
):
    """
    Chunked KDA recurrence with gradient checkpointing for memory efficiency.

    Processes the sequence in chunks of `chunk_size` tokens, applying
    gradient checkpointing at each chunk boundary to avoid storing all
    intermediate activations for the backward pass.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    q = q * scale

    if initial_state is not None:
        S = initial_state.float()
    else:
        S = q.new_zeros(B, H, K, V)

    num_chunks = (T + chunk_size - 1) // chunk_size
    o_parts = []

    for c in range(num_chunks):
        t_start = c * chunk_size
        t_end = min(t_start + chunk_size, T)

        q_c = q[:, t_start:t_end]
        k_c = k[:, t_start:t_end]
        v_c = v[:, t_start:t_end]
        g_c = g[:, t_start:t_end]
        b_c = beta[:, t_start:t_end]

        if torch.is_grad_enabled():
            o_c, S = grad_checkpoint(
                _kda_recurrent_chunk,
                q_c, k_c, v_c, g_c, b_c, S,
                use_reentrant=False,
            )
        else:
            o_c, S = _kda_recurrent_chunk(q_c, k_c, v_c, g_c, b_c, S)

        o_parts.append(o_c)

    o = torch.cat(o_parts, dim=1)  # [B, T, H, V]

    if not output_final_state:
        S = None
    return o.to(dtype), (S.to(dtype) if S is not None else None)


# ---------------------------------------------------------------------------
# KDASlot — a single KDA memory slot (K/V/gate/beta projections)
# Shared by both SparseKDAMemory (N slots) and the optional dense shared slot.
# Does NOT hold recurrent state — state is managed by the parent module.
# ---------------------------------------------------------------------------

class KDASlot(Module):
    """
    A single KDA memory slot: projects input to K, V, gate (g), and beta.
    The query (Q) is shared across all slots and computed by the parent module.

    Returns (k, v, g, beta) ready for the KDA recurrence kernel.
    """
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        allow_neg_eigval=False,
    ):
        super().__init__()
        self.heads = heads
        self.allow_neg_eigval = allow_neg_eigval
        self.use_short_conv = use_short_conv

        key_dim = heads * dim_head
        value_dim = heads * dim_head  # expand_v=1 for simplicity

        self.k_proj = Linear(dim, key_dim, bias=False)
        self.v_proj = Linear(dim, value_dim, bias=False)
        self.f_proj = nn.Sequential(
            Linear(dim, dim_head, bias=False),
            Linear(dim_head, key_dim, bias=False),
        )
        self.A_log = Parameter(torch.log(torch.empty(heads, dtype=torch.float32).uniform_(1, 16)))
        self.dt_bias = Parameter(torch.zeros(key_dim, dtype=torch.float32))
        self.b_proj = Linear(dim, heads, bias=False)

        if use_short_conv:
            self.k_conv1d = ShortConvolution(key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(value_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')

    def forward(self, normed):
        """
        Args:
            normed: [B, T, dim]  (pre-normed input)
        Returns:
            k:    [B, T, H, D]
            v:    [B, T, H, D]
            g:    [B, T, H, D]  (KDA gate, always <= 0)
            beta: [B, T, H]
        """
        if self.use_short_conv:
            k = self.k_conv1d(self.k_proj(normed))
            v = self.v_conv1d(self.v_proj(normed))
        else:
            k = F.silu(self.k_proj(normed))
            v = F.silu(self.v_proj(normed))

        g = self.f_proj(normed)
        beta = self.b_proj(normed).sigmoid()

        k, g = (rearrange(x, 'b t (h d) -> b t h d', h=self.heads) for x in (k, g))
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = naive_kda_gate(g, self.A_log, self.dt_bias)
        k = F.normalize(k, dim=-1)

        return k, v, g, beta


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
        use_grad_checkpoint = True,
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
        self.use_grad_checkpoint = use_grad_checkpoint
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
            # TODO: Should we add silu for QKV? 
            q = F.silu(self.q_proj(normed))
            k = F.silu(self.k_proj(normed))
            v = F.silu(self.v_proj(normed))

        # Gate: low-rank projection
        g = self.f_proj(normed) # [b t h d]
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
        if self.use_grad_checkpoint:
            o, new_hidden_state = kda_recurrent_checkpointed(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=hidden_state,
                output_final_state=return_state,
                chunk_size=self.chunk_size,
            )
        elif self.use_chunk and seq_len % self.chunk_size == 0:
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
    chunk_size=32,
    use_chunk=True,
    use_grad_checkpoint=True,
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
        use_grad_checkpoint=use_grad_checkpoint,
        use_short_conv=use_short_conv,
        **kwargs
    )


# ---------------------------------------------------------------------------
# SM-KDA: Sparse Mixture-of-KDA (naive recurrent)
# ---------------------------------------------------------------------------

@torch.compiler.disable
def naive_recurrent_sparse_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    router_weights: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """
    Sparse Mixture-of-KDA: routes each token to top-k of N memory slots.

    Args:
        q:            [B, T, H, D]  — shared query across all slots
        k, v, g:      [N, B, T, H, D]  — per-slot keys / values / gates
        beta:         [N, B, T, H]      — per-slot betas
        router_weights: [B, T, N]  — sparse routing weights (top-k masked, sums to 1)
        initial_state: [B, N, H, K, V]
        output_final_state: whether to return final state
    Returns:
        o: [B, T, H, V]
        S: [B, N, H, K, V] or None
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = router_weights.shape[-1]  # number of memory slots

    if scale is None:
        scale = K ** -0.5

    q = q.to(torch.float) * scale
    k, v, g, beta = map(lambda x: x.to(torch.float), [k, v, g, beta])
    router_weights = router_weights.to(torch.float)

    # S: [B, N, H, K, V]
    S = q.new_zeros(B, N, H, K, V)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros(B, T, H, V, dtype=torch.float, device=q.device)

    for i in range(T):
        q_i  = q[:, i]          # [B, H, K]
        k_i  = k[:, :, i]       # [N, B, H, K]
        v_i  = v[:, :, i]       # [N, B, H, V]
        g_i  = g[:, :, i]       # [N, B, H, K]
        b_i  = beta[:, :, i]    # [N, B, H]
        r_i  = router_weights[:, i]  # [B, N]

        # --- Write: decay then delta update (per-slot), before read ---
        # Decay: each slot uses its own per-slot gate
        # g_i: [N, B, H, K] -> exp: [N, B, H, K, 1] broadcast with S [B, N, H, K, V]
        S = S * g_i.permute(1, 0, 2, 3)[:, :, :, :, None].exp()  # [B, N, H, K, V]

        # Delta rule per slot:
        # kS[n] = k_i[n] @ S[n]  ->  [B, N, H, V]
        kS = torch.einsum('n b h k, b n h k v -> b n h v', k_i, S)
        # residual[n] = v_i[n] - kS[n]: [B, N, H, V]
        residual = v_i.permute(1, 0, 2, 3) - kS                           # [B, N, H, V]
        # delta[n] = b_i[n] * residual[n] outer k_i[n]: [B, N, H, K, V]
        delta = torch.einsum(
            'n b h, b n h v, n b h k -> b n h k v',
            b_i, residual, k_i
        )
        # Weighted update: r_i[:,n] scales each slot's delta
        S = S + r_i[:, :, None, None, None] * delta                       # [B, N, H, K, V]

        # --- Read: weighted sum over slots ---
        retrieved = torch.einsum('b h k, b n h k v -> b n h v', q_i, S)   # [B, N, H, V]
        o[:, i] = torch.einsum('b n, b n h v -> b h v', r_i, retrieved)   # [B, H, V]

    if not output_final_state:
        S = None
    return o.to(dtype), S


# ---------------------------------------------------------------------------
# Chunked recurrent sparse KDA with gradient checkpointing
# ---------------------------------------------------------------------------

@torch.compiler.disable
def _sparse_kda_chunk(q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, r_chunk, S_in, topk_indices, write_scale=1.0):
    """
    Process one chunk of the sparse KDA recurrence (for gradient checkpointing).

    Each slot only decays and updates when it is selected (top-k). Unselected slots
    are frozen — their state is preserved exactly until the next time they are activated.
    This prevents unselected slots from decaying to zero over long sequences.

    Args:
        q_chunk:    [B, T_c, H, D]
        k_chunk:    [N, B, T_c, H, D]  — per-slot keys
        v_chunk:    [N, B, T_c, H, D]  — per-slot values
        g_chunk:    [N, B, T_c, H, D]  — per-slot gates
        beta_chunk: [N, B, T_c, H]     — per-slot betas
        r_chunk:    [B, T_c, N]        — routing weights
        S_in:       [B, N, H, D, D]    — recurrent state
        topk_indices: [B, T_c, k]
    """
    dtype = v_chunk.dtype
    T_c = q_chunk.shape[1]
    B, _, H, K = q_chunk.shape
    V = v_chunk.shape[-1]
    N = k_chunk.shape[0]

    q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, r_chunk = map(
        lambda x: x.to(torch.float), [q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, r_chunk]
    )
    topk_indices = topk_indices.long()  # [B, T_c, k]

    S = S_in.to(torch.float)
    o_chunk = torch.zeros(B, T_c, H, V, dtype=torch.float, device=q_chunk.device)

    for i in range(T_c):
        q_i       = q_chunk[:, i]           # [B, H, K]
        r_i       = r_chunk[:, i]           # [B, N]
        topk_idx_i = topk_indices[:, i]     # [B, k]
        top_k = topk_idx_i.shape[1]

        # idx: [B, k, H, K, V] — expanded indices for gather/scatter on S
        idx = topk_idx_i[:, :, None, None, None].expand(-1, -1, H, K, V)

        # Gather top-k slots from recurrent state
        S_topk = S.gather(1, idx)           # [B, k, H, K, V]

        # Per-slot k/v/g/beta for the selected top-k slots only
        # k_chunk[n] = [B, T_c, H, D], gather along slot dim
        # topk_idx_i: [B, k] — need to index k_chunk along N dim per batch
        # Rearrange: k_chunk[:, :, i] -> [N, B, H, D], then gather top-k
        k_all_i    = k_chunk[:, :, i]       # [N, B, H, D]
        v_all_i    = v_chunk[:, :, i]       # [N, B, H, D]
        g_all_i    = g_chunk[:, :, i]       # [N, B, H, D]
        beta_all_i = beta_chunk[:, :, i]    # [N, B, H]

        # Gather per-batch top-k slots: [B, k, H, D]
        # k_all_i: [N, B, H, D] -> [B, N, H, D], then gather along dim=1 with [B, k, H, D] index
        k_all_i_b = k_all_i.permute(1, 0, 2, 3)      # [B, N, H, D]
        v_all_i_b = v_all_i.permute(1, 0, 2, 3)      # [B, N, H, D]
        g_all_i_b = g_all_i.permute(1, 0, 2, 3)      # [B, N, H, D]
        beta_all_i_b = beta_all_i.permute(1, 0, 2)   # [B, N, H]

        idx_kd = topk_idx_i[:, :, None, None].expand(-1, -1, H, K)  # [B, k, H, D]
        idx_kv = topk_idx_i[:, :, None, None].expand(-1, -1, H, V)  # [B, k, H, V]
        idx_kh = topk_idx_i[:, :, None].expand(-1, -1, H)           # [B, k, H]

        k_topk    = k_all_i_b.gather(1, idx_kd)      # [B, k, H, D]
        v_topk    = v_all_i_b.gather(1, idx_kv)      # [B, k, H, V]
        g_topk    = g_all_i_b.gather(1, idx_kd)      # [B, k, H, D]
        beta_topk = beta_all_i_b.gather(1, idx_kh)   # [B, k, H]

        # Decay: only top-k slots decay — unselected slots are frozen
        S_topk = S_topk * g_topk[:, :, :, :, None].exp()   # [B, k, H, D, 1] broadcast

        # Write: delta rule update for top-k slots only, using per-slot k/v/beta
        # (write before read, matching regular KDA order: decay -> write -> read)
        r_i_topk = r_i.gather(1, topk_idx_i)                                   # [B, k]
        kS      = torch.einsum('b s h d, b s h d v -> b s h v', k_topk, S_topk)   # [B, k, H, V]
        residual = v_topk - kS                                                      # [B, k, H, V]
        delta    = torch.einsum('b s h, b s h v, b s h d -> b s h d v', beta_topk, residual, k_topk)
        weighted_delta = r_i_topk[:, :, None, None, None] * delta
        S_topk = S_topk + weighted_delta

        # Read: query against top-k slots, weighted sum
        # Note: 's' = top-k slot index, 'd' = key dim
        retrieved = torch.einsum('b h d, b s h d v -> b s h v', q_i, S_topk)  # [B, k, H, V]
        o_chunk[:, i] = torch.einsum('b s, b s h v -> b h v', r_i_topk, retrieved)

        # Scatter updated top-k slots back (non-inplace to avoid grad checkpoint issues)
        S = S.scatter(1, idx, S_topk)

    return o_chunk.to(dtype), S.to(dtype)


@torch.compiler.disable
def naive_recurrent_sparse_kda_checkpointed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    router_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 32,
    write_scale: float = 1.0,
):
    """
    Chunked sparse KDA with gradient checkpointing for memory efficiency.

    Processes the sequence in chunks of `chunk_size` tokens, applying
    gradient checkpointing at each chunk boundary to reduce peak memory.
    Only the top-k selected slots per token are read/written (MoE-style).

    Args:
        write_scale: multiplier on write weights to compensate for reduced
                     per-slot update frequency when top_k < N. Typically N/top_k.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V_dim = v.shape[-1]
    N = router_weights.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # Scale queries
    q = q * scale

    # Initialize state
    if initial_state is not None:
        S = initial_state.float()
    else:
        S = q.new_zeros(B, N, H, K, V_dim)

    # Process in chunks
    num_chunks = (T + chunk_size - 1) // chunk_size
    o_parts = []

    for c in range(num_chunks):
        t_start = c * chunk_size
        t_end = min(t_start + chunk_size, T)

        q_c    = q[:, t_start:t_end]
        k_c    = k[:, :, t_start:t_end]       # [N, B, chunk, H, D]
        v_c    = v[:, :, t_start:t_end]
        g_c    = g[:, :, t_start:t_end]
        b_c    = beta[:, :, t_start:t_end]    # [N, B, chunk, H]
        r_c    = router_weights[:, t_start:t_end]
        topk_c = topk_indices[:, t_start:t_end]   # [B, chunk_size, k]

        # grad_checkpoint passes all args through and saves/restores them
        # It only recomputes the forward during backward, not storing inner activations
        if torch.is_grad_enabled():
            o_c, S = grad_checkpoint(
                _sparse_kda_chunk,
                q_c, k_c, v_c, g_c, b_c, r_c, S, topk_c, write_scale,
                use_reentrant=False,
            )
        else:
            o_c, S = _sparse_kda_chunk(q_c, k_c, v_c, g_c, b_c, r_c, S, topk_c, write_scale)

        o_parts.append(o_c)

    o = torch.cat(o_parts, dim=1)  # [B, T, H, V]

    if not output_final_state:
        S = None
    return o.to(dtype), (S.to(dtype) if S is not None else None)


# ---------------------------------------------------------------------------
# SparseKDAMemory — SM-KDA module
# ---------------------------------------------------------------------------

class SparseKDAMemory(Module):
    """
    Sparse Mixture-of-KDA (SM-KDA) memory module.

    Extends KDAMemory with N memory slots and top-k sparse routing.
    Each token is routed to k out of N memory matrices via a learned gate.

    Args:
        dim: model dimension
        heads: number of attention heads
        num_memory_slots: N — total number of memory matrices
        top_k: k — how many slots each token activates
        use_short_conv: causal conv1d on Q/K/V
        conv_size: kernel size for short conv
        norm_eps: epsilon for RMSNorm
    """
    def __init__(
        self,
        dim,
        dim_head=None,
        heads=4,
        num_memory_slots=8,
        top_k=2,
        use_short_conv=True,
        conv_size=4,
        conv_bias=False,
        allow_neg_eigval=False,
        norm_eps=1e-5,
        use_shared_memory=False,
    ):
        super().__init__()
        dim_head = default(dim_head, dim // heads)

        self.heads = heads
        self.dim_head = dim_head
        self.num_memory_slots = num_memory_slots
        self.top_k = top_k
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.use_shared_memory = use_shared_memory

        head_k_dim = dim_head
        head_v_dim = dim_head
        key_dim = heads * head_k_dim
        value_dim = heads * head_v_dim

        self.norm = nn.RMSNorm(dim)

        # Q projection (shared across all slots — all slots use the same query)
        self.q_proj = Linear(dim, key_dim, bias=False)
        if use_short_conv:
            self.q_conv1d = ShortConvolution(key_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')

        # N sparse slots — each is an independent KDASlot
        slot_kwargs = dict(
            dim=dim, heads=heads, dim_head=dim_head,
            use_short_conv=use_short_conv, conv_size=conv_size, conv_bias=conv_bias,
            allow_neg_eigval=allow_neg_eigval,
        )
        self.slots = nn.ModuleList([KDASlot(**slot_kwargs) for _ in range(num_memory_slots)])

        # Router: projects x -> N logits, then top-k sparse softmax
        self.router = Linear(dim, num_memory_slots, bias=False)

        # Optional shared (dense) slot: all tokens read/write, independent KDASlot
        self.shared_slot = KDASlot(**slot_kwargs) if use_shared_memory else None

        # Output gate
        self.g_proj = nn.Sequential(
            Linear(dim, head_v_dim, bias=False),
            Linear(head_v_dim, value_dim, bias=True),
        )
        self.o_norm = nn.RMSNorm(head_v_dim, eps=norm_eps)
        self.o_proj = Linear(value_dim, dim, bias=False)

        self.register_buffer('_slot_hit_counts', torch.zeros(num_memory_slots), persistent=False)
        self.register_buffer('_slot_weight_sum', torch.zeros(num_memory_slots), persistent=False)
        self.register_buffer('_slot_logit_sum', torch.zeros(num_memory_slots), persistent=False)
        self.register_buffer('_slot_logit_sq_sum', torch.zeros(num_memory_slots), persistent=False)
        self._slot_weight_n = 0  # number of (batch, token) pairs accumulated

        # Switch Transformer load balance auxiliary loss accumulator
        self._aux_loss_accum = None  # summed aux_loss tensors (stays on compute graph)
        self._aux_loss_count = 0     # number of forward passes accumulated

    def _route(self, x):
        """
        Compute sparse top-k routing weights.
        Args:
            x: [B, T, dim]
        Returns:
            router_weights: [B, T, N]  — sparse, sums to 1 over selected k slots
            topk_idx: [B, T, k]  — indices of the k selected slots per token
        """
        logits = self.router(x)                          # [B, T, N]
        # MoM-style routing: softmax first, then top-k, then renormalize
        scores = logits.float().softmax(dim=-1)          # [B, T, N], bounded in (0, 1)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)  # [B, T, k]
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # renormalize within top-k
        weights = torch.zeros_like(scores).scatter_(-1, topk_idx, topk_vals).to(logits.dtype)  # [B, T, N]

        with torch.no_grad():
            # Hit counts: how many tokens selected each slot
            counts = torch.zeros(self.num_memory_slots, device=x.device)
            counts.scatter_add_(0, topk_idx.reshape(-1), torch.ones(topk_idx.numel(), device=x.device))
            self._slot_hit_counts += counts

            # Average weights
            self._slot_weight_sum += weights.sum(dim=(0, 1))  # [N]

            # Logit mean and variance across all tokens
            logits_flat = logits.float()  # [B, T, N]
            self._slot_logit_sum += logits_flat.sum(dim=(0, 1))       # [N]
            self._slot_logit_sq_sum += (logits_flat ** 2).sum(dim=(0, 1))  # [N]

            self._slot_weight_n += x.shape[0] * x.shape[1]   # B * T

        # Switch Transformer load balance auxiliary loss
        # L_aux = N * sum_i( f_i * P_i )
        # f_i: fraction of tokens routed to slot i (stop-gradient)
        # P_i: mean full softmax prob for slot i (differentiable, provides gradient)
        # TODO: Switch Transformer aux loss has known issues with load collapse in practice.
        #       DeepSeek-V2/V3 tech reports describe an improved auxiliary-loss-free strategy
        #       using a per-slot bias term added to the router logits before top-k selection
        #       (not affecting the routing weights themselves), updated with a simple online
        #       rule: bias_i += gamma if slot i is overloaded, else bias_i -= gamma.
        #       Consider replacing this loss with that bias-based approach to avoid the
        #       tension between the task loss gradient and the aux loss gradient.
        B, T, N = logits.shape
        topk_onehot = torch.zeros_like(logits)
        topk_onehot.scatter_(-1, topk_idx, 1.0)
        f = topk_onehot.detach().float().sum(dim=(0, 1)) / (B * T * self.top_k)  # [N], sums to 1
        P = scores.mean(dim=(0, 1))                                                # [N], differentiable
        aux_loss = N * (f * P).sum()

        if self._aux_loss_accum is None:
            self._aux_loss_accum = aux_loss
        else:
            self._aux_loss_accum = self._aux_loss_accum + aux_loss
        self._aux_loss_count += 1

        return weights, topk_idx

    def reset_slot_stats(self):
        self._slot_hit_counts.zero_()
        self._slot_weight_sum.zero_()
        self._slot_logit_sum.zero_()
        self._slot_logit_sq_sum.zero_()
        self._slot_weight_n = 0

    def get_aux_loss(self):
        """Return the accumulated mean load balance aux loss (retains compute graph for backprop)."""
        if self._aux_loss_accum is None or self._aux_loss_count == 0:
            return None
        return self._aux_loss_accum / self._aux_loss_count

    def reset_aux_loss(self):
        self._aux_loss_accum = None
        self._aux_loss_count = 0

    def get_slot_hit_rates(self):
        """Fraction of tokens that selected each slot via top-k (normalized, sums to 1)."""
        counts = self._slot_hit_counts
        total = counts.sum().clamp(min=1)
        return counts / total

    def get_slot_avg_weights(self):
        """Mean routing weight per slot (post-softmax)."""
        n = max(self._slot_weight_n, 1)
        return self._slot_weight_sum / n

    def get_slot_logit_stats(self):
        """Mean and std of router logits per slot. Higher std indicates more router specialization."""
        n = max(self._slot_weight_n, 1)
        mean = self._slot_logit_sum / n
        variance = (self._slot_logit_sq_sum / n) - mean ** 2
        std = variance.clamp(min=0).sqrt()
        return mean, std

    @torch.no_grad()
    def _run_oracle_debug(self, q, k, v, topk_indices, hidden_state):
        """
        Oracle debug: compare each slot's memory output vs its causal attention oracle.

        For each slot i, the oracle is causal attention using slot i's own K/V projections:
            oracle_i = softmax(q K_i^T / sqrt(d)) V_i   (causal, over full sequence)

        This measures how well each slot's recurrent state approximates its ideal attention output.
        Also checks whether the router selects the slot closest to its own oracle.

        Args:
            q:            [B, T, H, D]   shared query
            k:            [N, B, T, H, D] per-slot keys
            v:            [N, B, T, H, D] per-slot values
            topk_indices: [B, T, top_k]  router selections
            hidden_state: [B, N, H, D, D] or None — initial recurrent state
        """
        N, B, T, H, D = k.shape
        V = v.shape[-1]

        # Causal attention oracle for each slot using slot i's own K/V
        # q: [B, T, H, D] -> [B, H, T, D]
        q_bhtd = q.float().permute(0, 2, 1, 3)
        causal_mask = torch.full((T, T), float('-inf'), device=q.device).triu(1)

        oracle_os = []
        for i in range(N):
            k_i = k[i].float().permute(0, 2, 1, 3)  # [B, H, T, D]
            v_i = v[i].float().permute(0, 2, 1, 3)  # [B, H, T, V]
            oracle_i = F.scaled_dot_product_attention(q_bhtd, k_i, v_i, attn_mask=causal_mask)
            oracle_os.append(oracle_i.permute(0, 2, 1, 3))  # [B, T, H, V]
        oracle_os = torch.stack(oracle_os, dim=0)  # [N, B, T, H, V]

        # Memory output for each slot independently (using current hidden state)
        # Run full recurrent forward for each slot, treating it as a single KDA
        S = hidden_state.float() if hidden_state is not None else \
            torch.zeros(B, N, H, D, V, device=q.device, dtype=torch.float)

        mem_os = []
        for i in range(N):
            S_i = S[:, i]  # [B, H, D, V]
            o_i_list = []
            for t in range(T):
                q_t  = q[:, t].float()          # [B, H, D]
                o_t  = torch.einsum('b h d, b h d v -> b h v', q_t, S_i)
                o_i_list.append(o_t)
            mem_os.append(torch.stack(o_i_list, dim=1))  # [B, T, H, V]
        mem_os = torch.stack(mem_os, dim=0)  # [N, B, T, H, V]

        # Per-slot quality: how well memory approximates its own oracle
        # dist[i] = mean ||mem_i - oracle_i|| over (B, T, H, V)
        diff = mem_os - oracle_os                            # [N, B, T, H, V]
        slot_dist = diff.pow(2).mean(dim=(-1, -2))          # [N, B, T]
        slot_dist_mean = slot_dist.mean(dim=(1, 2))         # [N]  — per-slot quality

        # Best slot per token: which slot has lowest dist to its oracle
        best_slot = slot_dist.argmin(dim=0)                 # [B, T]

        # Router top-1 accuracy vs oracle best
        router_top1 = topk_indices[..., 0]                  # [B, T]
        match = (router_top1 == best_slot).float()
        random_baseline = self.top_k / N

        print(f"[oracle debug]  router top-1 accuracy : {match.mean():.3f}  "
              f"(random baseline: {random_baseline:.3f})")
        print(f"[oracle debug]  per-slot dist to oracle: "
              f"{[f'{d:.4f}' for d in slot_dist_mean.tolist()]}")
        best_dist = [f'{(best_slot == i).float().mean().item():.3f}' for i in range(N)]
        print(f"[oracle debug]  best-slot distribution : {best_dist}")

    def forward(
        self,
        seq,
        state: SparseKDAState | None = None,
        return_state=True,
    ):
        debug_oracle = getattr(self, '_oracle_debug_next', False)
        self._oracle_debug_next = False  # consume the flag — only fires once
        batch, seq_len = seq.shape[:2]

        if not exists(state):
            state = SparseKDAState(0, None, None)
        seq_index, hidden_state, shared_hidden_state = state

        normed = self.norm(seq)

        # Q: shared across all slots
        if self.use_short_conv:
            q = self.q_conv1d(self.q_proj(normed))
        else:
            q = F.silu(self.q_proj(normed))
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.heads)
        q = F.normalize(q, dim=-1)

        # K, V, g, beta: each sparse slot computes its own via KDASlot
        # Stack results: k/v/g [N, B, T, H, D], beta [N, B, T, H]
        slot_outputs = [slot(normed) for slot in self.slots]
        k    = torch.stack([o[0] for o in slot_outputs], dim=0)   # [N, B, T, H, D]
        v    = torch.stack([o[1] for o in slot_outputs], dim=0)
        g    = torch.stack([o[2] for o in slot_outputs], dim=0)
        beta = torch.stack([o[3] for o in slot_outputs], dim=0)   # [N, B, T, H]

        # Sparse routing weights [B, T, N] and top-k indices [B, T, k]
        router_weights, topk_indices = self._route(normed)

        if debug_oracle:
            self._run_oracle_debug(q, k, v, topk_indices, hidden_state)

        o, new_hidden_state = naive_recurrent_sparse_kda_checkpointed(
            q=q, k=k, v=v, g=g, beta=beta,
            router_weights=router_weights,
            topk_indices=topk_indices,
            initial_state=hidden_state,
            output_final_state=return_state,
            chunk_size=32,
            write_scale=1.0,
        )

        # Shared (dense) slot: all tokens read/write, q reused from above
        # Add shared output before o_norm, matching MoM: o_norm(sparse_o + shared_o) * gate
        if self.shared_slot is not None:
            sk, sv, sg, sbeta = self.shared_slot(normed)
            shared_o, new_shared_hidden_state = kda_recurrent_checkpointed(
                q=q, k=sk, v=sv, g=sg, beta=sbeta,
                initial_state=shared_hidden_state,
                output_final_state=return_state,
                chunk_size=32,
            )
            o = o + shared_o
        else:
            new_shared_hidden_state = None

        gate = rearrange(self.g_proj(normed), 'b t (h d) -> b t h d', h=self.heads).sigmoid()
        o = self.o_norm(o) * gate

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if return_state:
            new_state = SparseKDAState(seq_index + seq_len, new_hidden_state, new_shared_hidden_state)
        else:
            new_state = None

        return o, new_state


def create_sparse_kda_memory_for_mac(
    dim,
    heads=4,
    num_memory_slots=8,
    top_k=2,
    use_short_conv=True,
    use_shared_memory=False,
    **kwargs
):
    """
    Create SparseKDAMemory configured for MAC Transformer.

    Args:
        dim: template model dimension
        heads: number of attention heads
        num_memory_slots: N — total memory slots
        top_k: k — slots activated per token
        use_short_conv: causal conv on Q/K/V
    """
    return SparseKDAMemory(
        dim=dim,
        heads=heads,
        num_memory_slots=num_memory_slots,
        top_k=top_k,
        use_short_conv=use_short_conv,
        use_shared_memory=use_shared_memory,
        **kwargs
    )
