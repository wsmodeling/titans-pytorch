# KDA Memory - Kimi Linear Attention Memory Module

This is a replacement for the Neural Memory module using Kimi's Dynamic Attention (KDA) linear attention mechanism.

## Overview

KDA Memory provides an alternative to the test-time-training based Neural Memory by using a linear attention mechanism with:
- Gated recurrent updates
- Beta modulation for dynamic information flow
- Efficient chunked processing
- O(n) complexity instead of O(n²) attention

## Quick Start

### Basic Usage

```python
from titans_pytorch import KDAMemory

# Create KDA memory module
kda_mem = KDAMemory(
    dim=256,           # Model dimension
    dim_head=64,       # Dimension per head
    heads=4,           # Number of attention heads
    chunk_size=64,     # Chunk size for parallel processing
    use_chunk=True,    # Use chunked version (faster)
    pre_rmsnorm=True,  # RMS normalization before processing
    qk_rmsnorm=True,   # RMS norm on queries and keys
).cuda()

# Forward pass
x = torch.randn(batch, seq_len, dim).cuda()
output, state = kda_mem(x, state=None, return_state=True)
```

### Incremental/Autoregressive Inference

```python
# First chunk
x1 = torch.randn(batch, 64, dim).cuda()
out1, state1 = kda_mem(x1, state=None, return_state=True)

# Continue with state (like autoregressive generation)
x2 = torch.randn(batch, 1, dim).cuda()
out2, state2 = kda_mem(x2, state=state1, return_state=True)
```

## Replacing Neural Memory in train_mac.py

To use KDA Memory instead of Neural Memory in the MAC Transformer:

### Option 1: Modify train_mac.py directly

```python
# Change line 54
USE_MEM_ATTENTION_MODEL = False  # Keep this
USE_KDA_MEMORY = True  # Add this new flag

# Then in the model configuration section (around line 98-107):
if USE_KDA_MEMORY:
    from titans_pytorch import KDAMemory
    neural_memory_model = KDAMemory(
        dim=64,
        dim_head=64,
        heads=4,
        chunk_size=NEURAL_MEM_SEGMENT_LEN * 4,  # Adjust as needed
        use_chunk=True,
        pre_rmsnorm=True,
        qk_rmsnorm=NEURAL_MEM_QK_NORM,
    )
elif USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(dim=64)
else:
    neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)
```

### Option 2: Create a new training script

Copy `train_mac.py` to `train_mac_kda.py` and make the changes above.

## Key Differences from Neural Memory

| Feature | Neural Memory | KDA Memory |
|---------|--------------|------------|
| **Mechanism** | Test-time training with MLP weights as memory | Linear attention with gated recurrence |
| **Complexity** | O(params × updates) | O(n × d²) |
| **State** | MLP weights + momentum | Recurrent hidden state [B, H, D, D] |
| **Updates** | Gradient-based weight updates | Gated rank-one updates |
| **Chunking** | Based on batch_size boundaries | Based on chunk_size |
| **Memory** | Compressed into MLP parameters | Explicit state matrix |

## Parameters

- `dim`: Model dimension
- `dim_head`: Dimension per attention head (default: same as dim)
- `heads`: Number of attention heads (default: 1)
- `chunk_size`: Size of chunks for parallel processing (default: 64)
- `use_chunk`: Use chunked version (True) vs recurrent (False)
- `pre_rmsnorm`: Apply RMSNorm before processing
- `post_rmsnorm`: Apply RMSNorm after processing
- `qk_rmsnorm`: Apply RMSNorm to queries and keys
- `activation`: Optional activation function for projections

## Implementation Details

### Core Algorithm

KDA uses a gated recurrent mechanism:

```
S_t = S_{t-1} * exp(g_t) + β_t * k_t * (v_t - k_t^T S_{t-1})
o_t = q_t^T S_t
```

Where:
- `S_t`: Recurrent state matrix [H, D, D]
- `g_t`: Gate (controls decay)
- `β_t`: Beta modulation (per-head scalar)
- `k_t, v_t`: Keys and values
- `q_t`: Query

### Chunked Processing

For efficiency, sequences are processed in chunks with:
- Within-chunk causal attention
- Cross-chunk recurrent state propagation
- Parallel computation within chunks

## Performance Tips

1. **Chunk Size**: Larger chunks (64-128) are more efficient but use more memory
2. **Use Chunked Mode**: `use_chunk=True` is much faster than recurrent for long sequences
3. **Inference**: For single-token inference, the recurrent path is used automatically
4. **Mixed Precision**: KDA works with fp16/bf16 (converts internally to fp32 for stability)

## Testing

Run the test suite:

```bash
python test_kda_memory.py
```

All tests should pass:
- ✓ Basic forward pass
- ✓ Incremental processing
- ✓ Shape compatibility with NeuralMemory
- ✓ Backward pass (gradient flow)
- ✓ Recurrent vs chunk implementation

## References

- Kimi Linear Attention Paper: https://arxiv.org/abs/2410.23029
- FLA Repository: https://github.com/fla-org/flash-linear-attention
- Original Titans Paper: https://arxiv.org/abs/2408.06654

## Notes

- KDA Memory is a drop-in replacement for Neural Memory with the same interface
- State management is different (single matrix vs weight dictionaries)
- Both provide long-range memory capabilities but through different mechanisms
- KDA may be faster for very long sequences due to linear complexity
