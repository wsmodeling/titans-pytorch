"""
Test script to verify memory type switching in train_mac.py
"""

import sys
import importlib.util

def test_memory_type(memory_type):
    """Test loading train_mac.py with different memory types"""
    print(f"\n{'='*60}")
    print(f"Testing with MEMORY_TYPE = '{memory_type}'")
    print('='*60)

    # Read the file
    with open('train_mac.py', 'r') as f:
        code = f.read()

    # Replace MEMORY_TYPE
    code = code.replace(
        "MEMORY_TYPE = 'neural'",
        f"MEMORY_TYPE = '{memory_type}'"
    )

    # Create a temporary module
    spec = importlib.util.spec_from_loader('train_mac_test', loader=None)
    module = importlib.util.module_from_spec(spec)

    try:
        exec(code, module.__dict__)

        print(f"✓ Successfully loaded with {memory_type} memory")
        print(f"  Neural memory model type: {type(module.neural_memory_model).__name__}")
        print(f"  Model created: {module.model.__class__.__name__}")
        print(f"  Run name: {module.RUN_NAME}")

        # Check model parameters
        total_params = sum(p.numel() for p in module.model.parameters())
        print(f"  Total parameters: {total_params:,}")

        return True
    except Exception as e:
        print(f"✗ Failed to load with {memory_type} memory")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Memory Type Switching in train_mac.py")

    results = {}

    # Test Neural Memory (TTT-based)
    results['neural'] = test_memory_type('neural')

    # Test KDA Memory (Linear Attention)
    results['kda'] = test_memory_type('kda')

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)

    for mem_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{mem_type:10s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print(f"\n{'='*60}")
        print("All tests passed! ✓")
        print('='*60)
        print("\nYou can now switch between memory types by changing:")
        print("  MEMORY_TYPE = 'neural'  # TTT-based Neural Memory")
        print("  MEMORY_TYPE = 'kda'     # KDA Linear Attention")
    else:
        print(f"\n{'='*60}")
        print("Some tests failed! ✗")
        print('='*60)
        sys.exit(1)
