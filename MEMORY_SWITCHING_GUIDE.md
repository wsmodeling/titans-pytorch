# Memory Type Switching Guide

This guide explains how to switch between Neural Memory (TTT-based) and KDA Memory (Linear Attention) in the MAC Transformer.

## Quick Start

In `train_mac.py`, simply change the `MEMORY_TYPE` flag:

```python
# Line ~46 in train_mac.py

# Choose memory type: 'neural' (TTT-based) or 'kda' (linear attention)
MEMORY_TYPE = 'neural'  # Change this to 'kda' to use KDA Memory
```

That's it! The script will automatically configure the correct memory module.

## Configuration

### Neural Memory (Default)

```python
MEMORY_TYPE = 'neural'
```

Uses test-time training based memory with MLP weights:
- Stores memories in neural network parameters
- Updates via gradient descent during forward pass
- Configured via `NEURAL_MEMORY_DEPTH`, `NEURAL_MEM_BATCH_SIZE`, etc.

### KDA Memory (Linear Attention)

```python
MEMORY_TYPE = 'kda'
```

Uses Kimi's linear attention mechanism:
- Gated recurrent updates with rank-one modifications
- Linear complexity O(n×d²) instead of quadratic
- Configured via `KDA_CHUNK_SIZE` and `KDA_USE_CHUNK`

## KDA-Specific Settings

When using `MEMORY_TYPE = 'kda'`, you can configure:

```python
# KDA memory specific settings (line ~71)
KDA_CHUNK_SIZE = NEURAL_MEM_SEGMENT_LEN * 8     # Chunk size for parallel processing
KDA_USE_CHUNK = True                            # Use chunked (faster) vs recurrent
```

**Recommendations:**
- **Chunk Size**: Larger values (64-128) are more efficient but use more memory
- **Use Chunk**: Keep `True` for training, automatically switches to recurrent for single-token inference

## Parameter Comparison

For the default `train_mac.py` configuration:

| Memory Type | Parameters | Difference |
|------------|-----------|------------|
| Neural Memory | 19,980,207 | Baseline |
| KDA Memory | 20,056,377 | +76,170 (+0.38%) |

The parameter counts are very similar, making them directly comparable.

## Performance Characteristics

### Neural Memory
- ✅ Explicit test-time training mechanism
- ✅ Proven in Titans paper
- ✅ Supports advanced features (weight residual, QKV layer selection)
- ⚠️ Complexity depends on number of updates

### KDA Memory
- ✅ Linear attention complexity
- ✅ Potentially faster for very long sequences
- ✅ Simpler formulation
- ⚠️ Newer, less tested in this context
- ⚠️ Doesn't support QKV layer selection yet

## Running Examples

### Demo Script

Run the demonstration to verify both memory types work:

```bash
python demo_memory_switching.py
```

Expected output:
```
============================================================
Demo: Memory Type Switching in MAC Transformer
============================================================

============================================================
Test 1: Neural Memory (TTT-based)
============================================================
...
✓ NEURAL memory works!

============================================================
Test 2: KDA Memory (Linear Attention)
============================================================
...
✓ KDA memory works!

============================================================
✓ Both memory types work!
============================================================
```

### Training Script

To train with Neural Memory (default):
```bash
# train_mac.py should have MEMORY_TYPE = 'neural'
uv run train_mac.py
```

To train with KDA Memory:
```bash
# Change MEMORY_TYPE = 'kda' in train_mac.py, then:
uv run train_mac.py
```

## Under the Hood

### How It Works

1. **Template Creation**: `train_mac.py` creates a memory model template with the base dimension (64)

2. **MAC Transformer Detection**: When building the model, `MemoryAsContextTransformer` detects if the template is a `KDAMemory` instance

3. **Proper Initialization**:
   - For `NeuralMemory`: Wraps the template in `NeuralMemory` class
   - For `KDAMemory`: Recreates with transformer's dimension (384) instead of wrapping

4. **Forward Pass**: The forward pass automatically uses the correct interface for each memory type

### Code Flow

```python
# In train_mac.py
if MEMORY_TYPE == 'kda':
    neural_memory_model = create_kda_memory_for_mac(...)  # Template
else:
    neural_memory_model = MemoryMLP(...)  # Template

# In mac_transformer.py __init__
if isinstance(neural_memory_model, KDAMemory):
    # Recreate with correct dimensions
    mem = KDAMemory(dim=transformer_dim, ...)
else:
    # Wrap in NeuralMemory
    mem = NeuralMemory(model=neural_memory_model, ...)

# In mac_transformer.py forward
if isinstance(mem, KDAMemory):
    retrieved, state = mem.forward(seq, state=state)
else:
    retrieved, state = mem.forward(qkv_input, state=state, prev_weights=...)
```

## Troubleshooting

### Issue: Dimension Mismatch

**Error**: `RuntimeError: Given normalized_shape=[64], expected input with shape [*64]`

**Solution**: Make sure the memory model template is created with `dim_head` (64), not transformer `dim` (384). The MAC Transformer will automatically recreate it with the correct dimensions.

### Issue: Import Error

**Error**: `ModuleNotFoundError: No module named 'titans_pytorch.kda_memory'`

**Solution**: Make sure you're using the updated codebase with `kda_memory.py` installed.

### Issue: CUDA Out of Memory

**Solution**:
1. Reduce `KDA_CHUNK_SIZE` (for KDA Memory)
2. Reduce `NEURAL_MEM_BATCH_SIZE` (for Neural Memory)
3. Reduce batch size or sequence length

## References

- **Neural Memory**: [Titans Paper](https://arxiv.org/abs/2408.06654)
- **KDA**: [Kimi Linear Attention Paper](https://arxiv.org/abs/2410.23029)
- **Implementation**: [FLA Repository](https://github.com/fla-org/flash-linear-attention)

## Testing

All memory switching functionality is tested in:
- `test_kda_memory.py` - Tests KDA Memory module
- `demo_memory_switching.py` - Tests both memory types in MAC Transformer
- `test_memory_switch.py` - Tests configuration switching

Run tests:
```bash
python test_kda_memory.py
python demo_memory_switching.py
```
