# Titans PyTorch Project Memory

## Project Overview
**Research project (academic, paper-oriented)**: Improving Kimi's linear attention paper (KDA) with sparse memory.

Core idea: extend KDA (Kimi Delta Attention / linear recurrent memory) with a sparse MoE routing mechanism
— multiple memory slots (S matrices), each token routes to top-k slots via a learned router.
Hypothesis: sparse memory improves long-context modeling capacity over dense single-slot KDA.
Goal: publish as a paper.

Using Titans codebase purely as an experiment harness (MAC Transformer training loop, enwik8 benchmark).
The actual contribution is the SparseKDA memory design, not the Titans architecture.

Main files: `train_mac.py`, `titans_pytorch/kda_memory.py`, `titans_pytorch/mac_transformer.py`.

**Memory sync**: Always keep `/workspace/wshao/titans-pytorch/.claude/` in sync with this directory (for source control).

## Memory Types
- `neural`: TTT-based MLP memory (NeuralMemory)
- `kda`: KDA (Kimi Delta Attention) linear attention memory
- `sparse_kda`: Sparse MoE version of KDA with N slots, top-k routing

## Key Architecture Notes

See `kda_architecture.md` for detailed KDA/SparseKDA architecture notes.

## Known Bugs Fixed

### 1. dim_head not passed in mac_transformer.py (FIXED)
When recreating KDAMemory/SparseKDAMemory per layer, `dim_head` was missing:
- `mac_transformer.py:577` SparseKDAMemory — added `dim_head=template.dim_head`
- `mac_transformer.py:591` KDAMemory — added `dim_head=template.dim_head`
Without this, dim_head defaulted to `dim // heads = 384 // 4 = 96` instead of 64.

### 2. sparse_kda template dim_head was 16 (FIXED)
`create_sparse_kda_memory_for_mac(dim=64, heads=4)` without explicit `dim_head`
→ `dim_head = 64 // 4 = 16` (not 64). Fixed by passing `dim_head=KDA_DIM_HEAD`.

### 3. Read/write order reversed in sparse KDA (FIXED)
Both `naive_recurrent_sparse_kda` and `_sparse_kda_chunk` were doing **read before write**.
Regular KDA order is: decay → write → read.
Sparse KDA was: decay → **read** → **write** (wrong).
Fix: swap order in both functions to match regular KDA.

### 4. Batch gather bug in `_sparse_kda_chunk` (FIXED — critical for N>1)
Original code used `topk_idx_i.t()` with `gather(dim=0)` on `k_all_i [N, B, H, D]`.
This is correct only for B=1. For B>1, it applies batch-0's slot indices to all batches.
Fix: permute `[N,B,H,D] → [B,N,H,D]` then `gather(dim=1)` with `topk_idx_i [B,k]` expanded properly.
This bug was invisible at N=1 (always slot 0) but would corrupt all k/v/g/beta for N>1.

### 5. `naive_recurrent_sparse_kda` wrong interface (FIXED)
Previously accepted shared `k/v/g/beta [B,T,H,D]` like regular KDA.
But `SparseKDAMemory` computes per-slot `[N,B,T,H,D]` tensors.
Fix: updated to accept per-slot shapes; decay now uses per-slot gates instead of shared gate.

## Current Config (train_mac.py)
- `MEMORY_TYPE = 'sparse_kda'`
- `SPARSE_KDA_NUM_SLOTS = 1`, `SPARSE_KDA_TOP_K = 1` (sanity check vs kda)
- `SPARSE_KDA_AUX_LOSS_WEIGHT = 0.0`
- `KDA_HEADS = 4`, `KDA_DIM_HEAD = 64` (shared by both kda and sparse_kda)
- Transformer: `dim=384, heads=8, dim_head=64, depth=8`

## Experiment Status
- N=1, k=1: sanity check vs kda — was ~0.2 loss gap due to bugs 3+4 above; after fix should converge
- N=8, k=4: main experiment config (bug 4 was critical here)
- sparse_kda does not support chunked compute (`use_chunk`), always recurrent checkpointed
- `naive_recurrent_sparse_kda` is NOT called in forward; only `naive_recurrent_sparse_kda_checkpointed` is used

## MoM Reference (refs/MoM/)
- `refs/MoM/mom/layers/mom.py` — full MomAttention (gated delta rule + MoM routing)
- `refs/MoM/mom_naive/layers/mom_gsa.py` — MomGatedSlotAttention (GSA-based)
- MoM shared_mem: add sparse_o + shared_o **before** o_norm, then `o_norm(combined) * gate` — our impl matches
