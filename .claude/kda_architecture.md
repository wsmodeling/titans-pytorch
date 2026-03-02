# KDA Memory Architecture Notes

## Delta Rule (Memory Read/Write)

```
# Per-token update (simplified):
S: [B, H, K, V]   # memory matrix, K=V=dim_head
k_i: [B, H, K]    # key for token i
v_i: [B, H, V]    # value for token i
q_i: [B, H, K]    # query for token i

# Write (delta rule):
residual = v_i - einsum('b h k, b h k v -> b h v', k_i, S)  # prediction error
S = S * exp(g_i) + einsum('b h k, b h v -> b h k v', beta_i * k_i, residual)

# Read:
o_i = einsum('b h k, b h k v -> b h v', q_i, S)
```

**Key constraint**: `k_i` shape `[B, H, K]` must match S's `H` and `K` dimensions.
`H = heads`, `K = dim_head`. Both are set by KDA's own projection layers (independent of transformer attention).

## KDA vs SparseKDA Comparison (N=1, k=1)

| Parameter | KDAMemory | SparseKDAMemory |
|-----------|-----------|-----------------|
| S shape | [B, H, K, V] | [B, N, H, K, V] (N=1) |
| heads | 4 | 4 |
| dim_head | 64 | 64 |
| use_chunk | True (faster) | Not supported (always recurrent) |
| router | None | Linear(dim, N) — extra param, unused when N=1 |
| write_scale | 1.0 | N/top_k = 1.0 (when N=k=1) |

Mathematically equivalent when N=1, k=1. Minor differences: extra router param, no chunked compute.

## Parameter Independence from Transformer Attention

KDA memory has **its own** W_q, W_k, W_v projection layers.
- Input: transformer hidden states of shape [B, T, dim=384]
- Output: [B, T, heads*dim_head]
- `heads` and `dim_head` are independent of transformer's attention heads/dim_head
- Only `dim` (=384) must match the transformer

Changing `KDA_HEADS` / `KDA_DIM_HEAD` changes memory matrix size = `heads * dim_head^2`.
`KDA_DIM_HEAD` is shared between kda and sparse_kda for fair comparison.

## Sparse Routing (SparseKDAMemory)

- Router: `Linear(dim, N)` → logits → top-k sparse softmax → routing weights `r_i`
- `write_scale = N / top_k`: compensates for reduced per-slot update frequency
  - KDA: r_i = 1.0 per token
  - SparseKDA top-k=4, N=8: r_i ≈ 0.25 on average → scale by 8/4=2 to match

## Switch Transformer Aux Loss

```
L_aux = N * sum_i(f_i * P_i)
```
- `f_i`: fraction of tokens routed to slot i (stop-gradient)
- `P_i`: mean router softmax prob for slot i (differentiable)
- Gradient: `∂L_aux/∂logit_i = N * f_i * P_i * (1 - P_i)` — f_i weights gradient toward overloaded slots

**Known issue**: When top_k=N, f_i=1/N always → aux loss cannot distinguish overloaded slots.
Requires top_k < N for aux loss to work.

**TODO**: DeepSeek-V2/V3 uses auxiliary-loss-free strategy:
per-slot bias added to router logits before top-k (not affecting routing weights),
updated online: `bias_i += gamma if overloaded, else bias_i -= gamma`.

## mac_transformer.py Integration

- Template memory model created in `train_mac.py` with small `dim=64`
- `MemoryAsContextTransformer.__init__` recreates one instance per neural memory layer with `dim=384`
- Critical: must pass `dim_head=template.dim_head` when recreating (bug was missing this)
- Forward interface: `mem.forward(x, state=..., return_state=True)` → `(retrieved, new_state)`
- `retrieved` is added to residual stream via hyper connections
