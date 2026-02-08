"""
Demo: Switching between Neural Memory and KDA Memory in MAC Transformer
"""

import torch
from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    create_kda_memory_for_mac
)

def create_model_with_memory_type(memory_type='neural'):
    """Create MAC Transformer with specified memory type"""

    # Common model config
    dim = 384
    dim_head = 64
    heads = dim // dim_head  # 6 heads

    config = dict(
        num_tokens=256,
        dim=dim,
        depth=8,
        segment_len=32,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(2, 4, 6),
    )

    if memory_type == 'kda':
        print(f"Creating model with KDA Memory (Linear Attention)")
        neural_memory_model = create_kda_memory_for_mac(
            dim=dim_head,  # Template dim (will be recreated with correct dim in MAC)
            chunk_size=32,
            use_chunk=True,
            qk_rmsnorm=True,
        )
    elif memory_type == 'neural':
        print(f"Creating model with Neural Memory (TTT-based MLP)")
        neural_memory_model = MemoryMLP(
            dim=dim_head,  # Must match dim_head!
            depth=2
        )
    else:
        raise ValueError(f"Unknown memory_type: {memory_type}")

    model = MemoryAsContextTransformer(
        **config,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=dim_head,
            heads=heads,
        )
    ).cuda()

    return model


def test_model(model, memory_type):
    """Test model forward pass"""
    print(f"\nTesting {memory_type} memory...")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, 256, (batch_size, seq_len)).cuda()

    with torch.no_grad():
        logits = model(x)

    print(f"  Forward pass: {x.shape} -> {logits.shape}")
    print(f"  ✓ {memory_type.upper()} memory works!")


if __name__ == "__main__":
    print("="*60)
    print("Demo: Memory Type Switching in MAC Transformer")
    print("="*60)

    # Test Neural Memory
    print("\n" + "="*60)
    print("Test 1: Neural Memory (TTT-based)")
    print("="*60)
    model_neural = create_model_with_memory_type('neural')
    test_model(model_neural, 'neural')

    # Test KDA Memory
    print("\n" + "="*60)
    print("Test 2: KDA Memory (Linear Attention)")
    print("="*60)
    model_kda = create_model_with_memory_type('kda')
    test_model(model_kda, 'kda')

    # Compare
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    params_neural = sum(p.numel() for p in model_neural.parameters())
    params_kda = sum(p.numel() for p in model_kda.parameters())

    print(f"Neural Memory: {params_neural:,} parameters")
    print(f"KDA Memory:    {params_kda:,} parameters")
    diff = params_kda - params_neural
    print(f"Difference:    {diff:+,} parameters ({diff/params_neural*100:+.2f}%)")

    print("\n" + "="*60)
    print("✓ Both memory types work!")
    print("="*60)
    print("\nTo switch in train_mac.py, change:")
    print("  MEMORY_TYPE = 'neural'  # for Neural Memory (TTT)")
    print("  MEMORY_TYPE = 'kda'     # for KDA Memory (Linear Attention)")
