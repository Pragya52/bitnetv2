# examples/quick_start.py
"""
Quick start example for BitNet v2
This script demonstrates basic model creation and testing
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2 import BitNetV2ForCausalLM, get_model_config

def main():
    print("BitNet v2 Quick Start Example")
    print("=" * 40)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a small model for testing
    print("\n1. Creating BitNet v2 model...")
    config = get_model_config('400M')  # Start with smallest model
    model = BitNetV2ForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Move to device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Test model forward pass
    print("\n2. Testing model forward pass...")
    try:
        # Create dummy input
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits']
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print("✓ Forward pass successful!")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test activation quantization switching
    print("\n3. Testing activation quantization switching...")
    try:
        # Check initial state (should be 8-bit)
        sample_module = None
        for module in model.modules():
            if hasattr(module, 'activation_quantizer'):
                sample_module = module
                break
        
        if sample_module:
            print(f"Initial activation bits: {sample_module.activation_quantizer.bits}")
            
            # Switch to 4-bit
            for module in model.modules():
                if hasattr(module, 'activation_quantizer'):
                    module.activation_quantizer.bits = 4
            
            print(f"After switching: {sample_module.activation_quantizer.bits}")
            print("✓ Activation quantization switching successful!")
        else:
            print("✗ No activation quantizer found")
            
    except Exception as e:
        print(f"✗ Activation switching failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✓ Quick start example completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python training/train.py --model_size 400M --quick_test")
    print("2. Run evaluation: python evaluation/evaluate.py --model_path <path> --quick_test")
    print("3. Try examples: python examples/inference_example.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
