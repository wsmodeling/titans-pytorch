# titans-pytorch — Claude Code Context

Project context and architecture notes are maintained in `.claude/`:
- `.claude/MEMORY.md` — project overview, bug history, experiment status
- `.claude/kda_architecture.md` — KDA/SparseKDA architecture details, tensor shapes

## Quick Reference

**Main files**: `train_mac.py`, `titans_pytorch/kda_memory.py`, `titans_pytorch/mac_transformer.py`

**Switch memory type** in `train_mac.py`: `MEMORY_TYPE = 'neural' | 'kda' | 'sparse_kda'`

**Current branch**: `sparse-memory` — implementing SM-KDA (Sparse Mixture-of-KDA)

**KDA recurrence order** (per token): decay → write (delta rule) → read
```
S = S * exp(g_i)
S = S + beta * (v - k@S) ⊗ k   # delta rule write
o = q @ S                        # read
```

**Sparse KDA gather pattern** (bug-safe for B>1):
```python
# k_all_i [N,B,H,D] → permute → [B,N,H,D] → gather(dim=1, idx=[B,k,H,D])
k_all_i_b = k_all_i.permute(1, 0, 2, 3)
k_topk = k_all_i_b.gather(1, topk_idx_i[:,:,None,None].expand(-1,-1,H,K))
```
