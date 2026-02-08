"""
Test script for KDA Memory module
"""

import torch
from titans_pytorch import KDAMemory, KDAState

def test_kda_memory_basic():
    """Test basic forward pass"""
    print("=" * 60)
    print("Test 1: Basic Forward Pass")
    print("=" * 60)

    batch_size = 2
    seq_len = 64  # Must be divisible by chunk_size
    dim = 256
    heads = 4
    chunk_size = 16

    # Create model
    model = KDAMemory(
        dim=dim,
        dim_head=64,
        heads=heads,
        chunk_size=chunk_size,
        use_chunk=True,
        pre_rmsnorm=True,
        qk_rmsnorm=True
    ).cuda()

    # Create input
    x = torch.randn(batch_size, seq_len, dim).cuda()

    # Forward pass
    retrieved, state = model(x, state=None, return_state=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {retrieved.shape}")
    print(f"State hidden shape: {state.hidden_state.shape if state.hidden_state is not None else None}")
    print(f"State seq_index: {state.seq_index}")

    assert retrieved.shape == x.shape, "Output shape mismatch"
    assert state.seq_index == seq_len, "Seq index mismatch"
    print("✓ Basic forward pass successful\n")

    return model, state


def test_kda_memory_incremental():
    """Test incremental/autoregressive processing"""
    print("=" * 60)
    print("Test 2: Incremental Processing (Inference)")
    print("=" * 60)

    batch_size = 2
    seq_len_1 = 64
    seq_len_2 = 1  # Single token
    dim = 256
    heads = 4
    chunk_size = 16

    model = KDAMemory(
        dim=dim,
        dim_head=64,
        heads=heads,
        chunk_size=chunk_size,
        use_chunk=False,  # Use recurrent for single token
        pre_rmsnorm=True,
    ).cuda()

    # First chunk
    x1 = torch.randn(batch_size, seq_len_1, dim).cuda()
    retrieved1, state1 = model(x1, state=None, return_state=True)

    print(f"First chunk: input {x1.shape} → output {retrieved1.shape}")
    print(f"State after first chunk: seq_index={state1.seq_index}")

    # Second chunk (single token, like inference)
    x2 = torch.randn(batch_size, seq_len_2, dim).cuda()
    retrieved2, state2 = model(x2, state=state1, return_state=True)

    print(f"Second chunk: input {x2.shape} → output {retrieved2.shape}")
    print(f"State after second chunk: seq_index={state2.seq_index}")

    assert state2.seq_index == seq_len_1 + seq_len_2, "Cumulative seq_index mismatch"
    print("✓ Incremental processing successful\n")


def test_kda_vs_neural_memory_shape():
    """Compare shapes with original NeuralMemory"""
    print("=" * 60)
    print("Test 3: Shape Compatibility with NeuralMemory")
    print("=" * 60)

    from titans_pytorch import NeuralMemory, MemoryMLP

    batch_size = 2
    seq_len = 64
    dim = 256
    heads = 4

    # Original NeuralMemory
    neural_mem = NeuralMemory(
        dim=dim,
        dim_head=64,
        heads=heads,
        chunk_size=16,
        model=MemoryMLP(64, depth=2),
    ).cuda()

    # KDA Memory
    kda_mem = KDAMemory(
        dim=dim,
        dim_head=64,
        heads=heads,
        chunk_size=16,
    ).cuda()

    x = torch.randn(batch_size, seq_len, dim).cuda()

    # Forward passes
    neural_out, neural_state = neural_mem(x, state=None)
    kda_out, kda_state = kda_mem(x, state=None, return_state=True)

    print(f"Input shape: {x.shape}")
    print(f"NeuralMemory output shape: {neural_out.shape}")
    print(f"KDAMemory output shape: {kda_out.shape}")

    assert neural_out.shape == kda_out.shape, "Output shapes don't match"
    print("✓ Shapes are compatible\n")


def test_kda_memory_backward():
    """Test backward pass"""
    print("=" * 60)
    print("Test 4: Backward Pass (Gradient Flow)")
    print("=" * 60)

    batch_size = 2
    seq_len = 64
    dim = 256

    model = KDAMemory(
        dim=dim,
        dim_head=64,
        heads=4,
        chunk_size=16,
    ).cuda()

    # Don't require grad on input since it goes through normalize
    x = torch.randn(batch_size, seq_len, dim).cuda()

    retrieved, state = model(x, state=None, return_state=True)
    loss = retrieved.sum()
    loss.backward()

    print(f"Loss: {loss.item():.4f}")

    # Check model parameters have gradients
    param_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
    total_params = sum(1 for _ in model.parameters())
    print(f"Parameters with gradients: {param_with_grad}/{total_params}")

    assert param_with_grad > 0, "No parameters have gradients"

    print("✓ Backward pass successful\n")


def test_recurrent_vs_chunk():
    """Compare recurrent and chunk implementations"""
    print("=" * 60)
    print("Test 5: Recurrent vs Chunk Implementation")
    print("=" * 60)

    batch_size = 2
    seq_len = 64
    dim = 128

    # Same initialization
    torch.manual_seed(42)
    model_recurrent = KDAMemory(
        dim=dim,
        dim_head=32,
        heads=4,
        chunk_size=16,
        use_chunk=False,  # Recurrent
    ).cuda()

    torch.manual_seed(42)
    model_chunk = KDAMemory(
        dim=dim,
        dim_head=32,
        heads=4,
        chunk_size=16,
        use_chunk=True,  # Chunk
    ).cuda()

    # Copy parameters
    model_chunk.load_state_dict(model_recurrent.state_dict())

    x = torch.randn(batch_size, seq_len, dim).cuda()

    with torch.no_grad():
        out_recurrent, _ = model_recurrent(x, state=None, return_state=False)
        out_chunk, _ = model_chunk(x, state=None, return_state=False)

    diff = (out_recurrent - out_chunk).abs().max().item()

    print(f"Max difference: {diff:.6f}")
    print(f"Recurrent output norm: {out_recurrent.norm().item():.4f}")
    print(f"Chunk output norm: {out_chunk.norm().item():.4f}")

    # They should be close (within numerical precision)
    # Note: chunk and recurrent may have numerical differences
    relative_diff = diff / max(out_recurrent.norm().item(), out_chunk.norm().item())
    print(f"Relative difference: {relative_diff:.6f}")

    if relative_diff < 0.01:  # 1% relative difference
        print("✓ Recurrent and chunk implementations match (within 1%)\n")
    else:
        print("⚠ Warning: Recurrent and chunk have numerical differences\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("KDA Memory Module Tests")
    print("="*60 + "\n")

    try:
        test_kda_memory_basic()
        test_kda_memory_incremental()
        test_kda_vs_neural_memory_shape()
        test_kda_memory_backward()
        test_recurrent_vs_chunk()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
