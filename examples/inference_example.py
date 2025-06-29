# examples/inference_example.py
"""
Simple inference example for BitNet v2
Demonstrates text generation and basic usage
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2 import BitNetV2ForCausalLM, get_model_config
from transformers import AutoTokenizer

def main():
    print("BitNet v2 Inference Example")
    print("=" * 40)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model (using smallest size for demo)
    config = get_model_config('400M')
    model = BitNetV2ForCausalLM(config).to(device)
    
    # Setup tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important lesson I learned today is",
        "Once upon a time in a magical forest,",
        "The solution to climate change requires"
    ]
    
    print("\nGenerating text with 8-bit activations:")
    print("-" * 50)
    
    # Generate with 8-bit activations
    model.eval()
    for prompt in prompts:
        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.size(1) + 30,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()
    
    # Switch to 4-bit activations
    print("\nSwitching to 4-bit activations...")
    for module in model.modules():
        if hasattr(module, 'activation_quantizer'):
            module.activation_quantizer.bits = 4
    
    print("\nGenerating text with 4-bit activations:")
    print("-" * 50)
    
    # Generate with 4-bit activations
    for prompt in prompts[:2]:  # Just first 2 for demo
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.size(1) + 30,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print()
    
    print("Inference example completed!")

if __name__ == "__main__":
    main()

# examples/ablation_study.py
"""
Ablation study example for BitNet v2
Tests the impact of Hadamard transformation and different configurations
"""

import torch
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2 import BitNetV2ForCausalLM, BitNetConfig
from evaluation.evaluator import ComprehensiveEvaluator
from transformers import AutoTokenizer

def create_ablation_configs():
    """Create different model configurations for ablation study"""
    base_config = {
        'hidden_size': 1024,
        'intermediate_size': 4096,
        'num_attention_heads': 16,
        'num_hidden_layers': 12,  # Smaller for faster testing
        'vocab_size': 32000
    }
    
    configs = {
        'baseline': BitNetConfig(**base_config, use_4bit_activations=False),
        'with_4bit': BitNetConfig(**base_config, use_4bit_activations=True),
    }
    
    return configs

def run_ablation_study(device='cuda', num_samples=50):
    """Run ablation study comparing different configurations"""
    print("BitNet v2 Ablation Study")
    print("=" * 40)
    
    # Setup tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create configurations
    configs = create_ablation_configs()
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting configuration: {config_name}")
        print("-" * 30)
        
        # Create model
        model = BitNetV2ForCausalLM(config).to(device)
        
        # Create evaluator
        evaluator = ComprehensiveEvaluator(model, tokenizer, device)
        
        # Run evaluation (quick test with limited samples)
        eval_results = evaluator.run_full_evaluation(num_samples_per_task=num_samples)
        results[config_name] = eval_results
        
        print(f"Results for {config_name}:")
        print(f"  Perplexity: {eval_results.perplexity:.2f}")
        print(f"  Average accuracy: {eval_results.average:.2f}%")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Peak GPU memory: {memory_used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"{'Config':<15} {'PPL↓':<8} {'Avg↑':<8} {'ARC-C↑':<8} {'HS↑':<8}")
    print("-" * 60)
    
    for config_name, result in results.items():
        print(f"{config_name:<15} {result.perplexity:<8.2f} {result.average:<8.2f} "
              f"{result.arc_challenge:<8.2f} {result.hellaswag:<8.2f}")
    
    # Analysis
    print("\nAnalysis:")
    if len(results) >= 2:
        baseline = results['baseline']
        with_4bit = results['with_4bit']
        
        ppl_diff = with_4bit.perplexity - baseline.perplexity
        acc_diff = with_4bit.average - baseline.average
        
        print(f"4-bit vs baseline:")
        print(f"  Perplexity change: {ppl_diff:+.2f}")
        print(f"  Accuracy change: {acc_diff:+.2f}%")
        
        if abs(ppl_diff) < 0.5 and abs(acc_diff) < 2.0:
            print("  → Minimal performance degradation with 4-bit activations ✓")
        else:
            print("  → Significant performance change with 4-bit activations")
    
    return results

def test_hadamard_impact():
    """Test the impact of Hadamard transformation on activation distributions"""
    print("\nTesting Hadamard transformation impact...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a small model
    config = BitNetConfig(
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=6
    )
    
    model = BitNetV2ForCausalLM(config).to(device)
    
    # Create dummy input
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass to get intermediate activations
    model.eval()
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        
        # Pass through first layer to get activations
        layer = model.model.layers[0]
        attn_output = layer.self_attn(hidden_states)
        
        print("Activation statistics:")
        print(f"  Mean: {attn_output.mean().item():.4f}")
        print(f"  Std: {attn_output.std().item():.4f}")
        print(f"  Min: {attn_output.min().item():.4f}")
        print(f"  Max: {attn_output.max().item():.4f}")
        
        # Check for outliers (values > 3 std from mean)
        mean = attn_output.mean()
        std = attn_output.std()
        outliers = torch.abs(attn_output - mean) > 3 * std
        outlier_percentage = outliers.float().mean().item() * 100
        
        print(f"  Outliers (>3σ): {outlier_percentage:.2f}%")
        
        if outlier_percentage < 5.0:
            print("  → Good distribution for quantization ✓")
        else:
            print("  → High outlier rate, may affect quantization")

def main():
    parser = argparse.ArgumentParser(description="Run BitNet v2 ablation study")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per task")
    parser.add_argument("--test_hadamard", action='store_true', help="Test Hadamard transformation")
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Run ablation study
    results = run_ablation_study(device, args.num_samples)
    
    # Test Hadamard transformation if requested
    if args.test_hadamard:
        test_hadamard_impact()
    
    print("\nAblation study completed!")

if __name__ == "__main__":
    main()

# examples/model_comparison.py
"""
Compare different BitNet v2 model sizes and configurations
"""

import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2 import BitNetV2ForCausalLM, get_model_config
from evaluation.evaluator import ComprehensiveEvaluator, compare_models
from transformers import AutoTokenizer

def create_models(device='cuda'):
    """Create different model sizes for comparison"""
    models = {}
    
    # Available model sizes (using smaller ones for demo)
    model_sizes = ['400M', '1.3B']
    
    for size in model_sizes:
        print(f"Creating {size} model...")
        config = get_model_config(size)
        config.num_hidden_layers = min(config.num_hidden_layers, 12)  # Reduce for demo
        
        model = BitNetV2ForCausalLM(config).to(device)
        models[f"BitNet-{size}"] = model
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")
    
    return models

def benchmark_models(models, device='cuda'):
    """Benchmark inference speed of different models"""
    print("\nBenchmarking inference speed...")
    print("-" * 40)
    
    sequence_lengths = [128, 512, 1024]
    batch_size = 1
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nBenchmarking {model_name}:")
        model_results = {}
        
        model.eval()
        with torch.no_grad():
            for seq_len in sequence_lengths:
                # Create dummy input
                input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(device)
                
                # Warmup
                for _ in range(3):
                    _ = model(input_ids)
                
                # Benchmark
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                num_runs = 10
                
                for _ in range(num_runs):
                    _ = model(input_ids)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / num_runs
                throughput = (batch_size * seq_len) / avg_time
                
                model_results[seq_len] = {
                    'time': avg_time,
                    'throughput': throughput
                }
                
                print(f"  Seq {seq_len}: {avg_time:.4f}s, {throughput:.1f} tokens/s")
        
        results[model_name] = model_results
    
    return results

def memory_analysis(models, device='cuda'):
    """Analyze memory usage of different models"""
    print("\nMemory analysis...")
    print("-" * 40)
    
    if device.type != 'cuda':
        print("Memory analysis only available on CUDA")
        return
    
    for model_name, model in models.items():
        torch.cuda.reset_peak_memory_stats()
        
        # Test with different input sizes
        input_sizes = [128, 512, 1024]
        
        print(f"\n{model_name}:")
        
        for seq_len in input_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            input_ids = torch.randint(0, 32000, (1, seq_len)).to(device)
            
            model.eval()
            with torch.no_grad():
                _ = model(input_ids)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  Seq {seq_len}: {peak_memory:.2f} GB")

def main():
    print("BitNet v2 Model Comparison")
    print("=" * 40)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create models
    models = create_models(device)
    
    # Benchmark speed
    speed_results = benchmark_models(models, device)
    
    # Memory analysis
    memory_analysis(models, device)
    
    # Quick evaluation comparison
    print("\nQuick evaluation comparison...")
    print("-" * 40)
    
    eval_results = compare_models(models, tokenizer, device, num_samples=10)
    
    print("\nComparison completed!")

if __name__ == "__main__":
    main()

# examples/quantization_demo.py
"""
Demonstration of quantization effects in BitNet v2
Shows the impact of different quantization settings
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2.quantization import WeightQuantizer, ActivationQuantizer, hadamard_transform

def demonstrate_weight_quantization():
    """Demonstrate weight quantization to ternary values"""
    print("Weight Quantization Demonstration")
    print("-" * 40)
    
    # Create sample weights
    weights = torch.randn(100, 100) * 0.1
    
    # Apply quantization
    quantizer = WeightQuantizer()
    quantized_weights = quantizer(weights)
    
    print(f"Original weights - Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
    print(f"Quantized weights - Mean: {quantized_weights.mean():.4f}, Std: {quantized_weights.std():.4f}")
    
    # Check quantization values
    unique_values = torch.unique(quantized_weights)
    print(f"Unique values in quantized weights: {unique_values.tolist()}")
    
    # Calculate quantization error
    error = torch.abs(weights - quantized_weights).mean()
    print(f"Mean quantization error: {error:.4f}")

def demonstrate_activation_quantization():
    """Demonstrate activation quantization for different bit widths"""
    print("\nActivation Quantization Demonstration")
    print("-" * 40)
    
    # Create sample activations (simulating typical LLM activations)
    activations = torch.randn(32, 128) * 2.0 + 0.5
    
    # Test different bit widths
    for bits in [8, 4]:
        quantizer = ActivationQuantizer(bits=bits)
        quantized = quantizer(activations)
        
        error = torch.abs(activations - quantized).mean()
        print(f"{bits}-bit quantization - Mean error: {error:.4f}")
        
        # Show dynamic range
        print(f"  Original range: [{activations.min():.3f}, {activations.max():.3f}]")
        print(f"  Quantized range: [{quantized.min():.3f}, {quantized.max():.3f}]")

def demonstrate_hadamard_transform():
    """Demonstrate the effect of Hadamard transformation on activation distributions"""
    print("\nHadamard Transformation Demonstration")
    print("-" * 40)
    
    # Create activations with outliers (simulating problematic distributions)
    size = 256  # Must be power of 2
    activations = torch.randn(32, size)
    
    # Add some outliers
    outlier_indices = torch.randint(0, size, (5,))
    activations[:, outlier_indices] *= 10  # Create outliers
    
    print(f"Original activations:")
    print(f"  Mean: {activations.mean():.4f}")
    print(f"  Std: {activations.std():.4f}")
    print(f"  Min: {activations.min():.4f}")
    print(f"  Max: {activations.max():.4f}")
    
    # Check outlier percentage (>3 standard deviations)
    mean = activations.mean()
    std = activations.std()
    outliers = torch.abs(activations - mean) > 3 * std
    outlier_pct = outliers.float().mean() * 100
    print(f"  Outliers (>3σ): {outlier_pct:.2f}%")
    
    # Apply Hadamard transformation
    transformed = hadamard_transform(activations)
    
    print(f"\nAfter Hadamard transformation:")
    print(f"  Mean: {transformed.mean():.4f}")
    print(f"  Std: {transformed.std():.4f}")
    print(f"  Min: {transformed.min():.4f}")
    print(f"  Max: {transformed.max():.4f}")
    
    # Check outliers after transformation
    t_mean = transformed.mean()
    t_std = transformed.std()
    t_outliers = torch.abs(transformed - t_mean) > 3 * t_std
    t_outlier_pct = t_outliers.float().mean() * 100
    print(f"  Outliers (>3σ): {t_outlier_pct:.2f}%")
    
    print(f"\nOutlier reduction: {outlier_pct:.2f}% → {t_outlier_pct:.2f}%")

def plot_distributions():
    """Plot activation distributions before and after Hadamard transformation"""
    try:
        import matplotlib.pyplot as plt
        
        print("\nPlotting distributions...")
        
        # Create sample data
        size = 256
        activations = torch.randn(1000, size)
        
        # Add outliers
        outlier_indices = torch.randint(0, size, (50,))
        activations[:, outlier_indices] *= 5
        
        # Apply transformation
        transformed = hadamard_transform(activations)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original distribution
        ax1.hist(activations.flatten().numpy(), bins=50, alpha=0.7, density=True)
        ax1.set_title('Original Activations')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Transformed distribution
        ax2.hist(transformed.flatten().numpy(), bins=50, alpha=0.7, density=True)
        ax2.set_title('After Hadamard Transform')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('activation_distributions.png', dpi=150, bbox_inches='tight')
        print("Distribution plot saved as 'activation_distributions.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")

def main():
    print("BitNet v2 Quantization Demonstration")
    print("=" * 50)
    
    # Demonstrate different quantization techniques
    demonstrate_weight_quantization()
    demonstrate_activation_quantization()
    demonstrate_hadamard_transform()
    
    # Plot distributions if matplotlib is available
    plot_distributions()
    
    print("\nQuantization demonstration completed!")
    print("\nKey takeaways:")
    print("1. Weight quantization reduces precision but maintains relative magnitudes")
    print("2. Lower bit-width activation quantization increases error")
    print("3. Hadamard transformation reduces outliers, enabling better quantization")

if __name__ == "__main__":
    main()
