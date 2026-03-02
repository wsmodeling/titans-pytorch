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

## Current Config (train_mac.py)
- `MEMORY_TYPE = 'sparse_kda'`
- `SPARSE_KDA_NUM_SLOTS = 1`, `SPARSE_KDA_TOP_K = 1` (sanity check vs kda)
- `SPARSE_KDA_AUX_LOSS_WEIGHT = 0.0`
- `KDA_HEADS = 4`, `KDA_DIM_HEAD = 64` (shared by both kda and sparse_kda)
- Transformer: `dim=384, heads=8, dim_head=64, depth=8`

## Experiment Status
- Trying N=1 k=1 sparse_kda vs kda as sanity check (should be ~equivalent)
- sparse_kda has extra `router Linear(384,1)` parameter (negligible difference)
- sparse_kda does not support chunked compute (`use_chunk`), always recurrent
