"""
Main evaluation script for BitNet v2
Supports comprehensive evaluation on all tasks from the paper
"""

import argparse
import os
import sys
import torch
import json
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2 import BitNetV2ForCausalLM, get_model_config
from evaluation.evaluator import ComprehensiveEvaluator, create_results_table
from evaluation.tasks import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate BitNet v2 model")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint or directory")
    parser.add_argument("--model_size", type=str, default="1.3B",
                       choices=['400M', '1.3B', '3B', '7B'],
                       help="Model size (used if loading config)")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config file")
    
    # Evaluation configuration
    parser.add_argument("--tasks", type=str, default="all",
                       help="Comma-separated list of tasks or 'all'")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples per task (None for full dataset)")
    parser.add_argument("--perplexity_samples", type=int, default=1000,
                       help="Number of samples for perplexity calculation")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--save_results", action='store_true',
                       help="Save results to JSON file")
    parser.add_argument("--verbose", action='store_true',
                       help="Verbose output")
    
    # Special evaluation modes
    parser.add_argument("--compare_activations", action='store_true',
                       help="Compare 8-bit vs 4-bit activations")
    parser.add_argument("--benchmark_speed", action='store_true',
                       help="Benchmark inference speed")
    parser.add_argument("--generate_samples", action='store_true',
                       help="Generate sample outputs")
    
    # Quick testing
    parser.add_argument("--quick_test", action='store_true',
                       help="Quick test with small number of samples")
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device for evaluation"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Available memory: {torch.cuda.get_device_properties(device).total_memory // 1024**3} GB")
    
    return device

def load_model(args, device):
    """Load model from checkpoint"""
    print(f"Loading model from {args.model_path}...")
    
    # Determine if it's a checkpoint file or directory
    if os.path.isfile(args.model_path):
        # Single checkpoint file
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'config' in checkpoint:
            # Checkpoint contains config
            config_dict = checkpoint['config']
            if 'model_configs' in config_dict:
                # Training config format
                model_config_dict = config_dict['model_configs'][args.model_size]
                from bitnet_v2.config import BitNetConfig
                config = BitNetConfig(**model_config_dict)
            else:
                # Direct config format
                from bitnet_v2.config import BitNetConfig
                config = BitNetConfig(**config_dict)
        else:
            # No config in checkpoint, use default
            config = get_model_config(args.model_size)
        
        # Create model
        model = BitNetV2ForCausalLM(config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    elif os.path.isdir(args.model_path):
        # Model directory
        config_path = os.path.join(args.model_path, "config.json")
        model_path = os.path.join(args.model_path, "pytorch_model.bin")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            from bitnet_v2.config import BitNetConfig
            config = BitNetConfig(**config_dict)
        else:
            config = get_model_config(args.model_size)
        
        model = BitNetV2ForCausalLM(config)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
    
    else:
        raise ValueError(f"Model path not found: {args.model_path}")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters")
    
    return model

def setup_tokenizer():
    """Setup tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except:
        print("Warning: Using fallback tokenizer")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def get_task_list(tasks_arg):
    """Parse task list from argument"""
    available_tasks = [
        'arc_challenge', 'arc_easy', 'hellaswag', 
        'piqa', 'winogrande', 'lambada'
    ]
    
    if tasks_arg.lower() == 'all':
        return available_tasks
    else:
        task_list = [task.strip() for task in tasks_arg.split(',')]
        # Validate tasks
        for task in task_list:
            if task not in available_tasks:
                raise ValueError(f"Unknown task: {task}. Available: {available_tasks}")
        return task_list

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    tokenizer = setup_tokenizer()
    model = load_model(args, device)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, tokenizer, device, args.max_length)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set number of samples for quick test
    if args.quick_test:
        num_samples = 10
        perplexity_samples = 50
        print("Running quick test with reduced samples...")
    else:
        num_samples = args.num_samples
        perplexity_samples = args.perplexity_samples
    
    # Run evaluation based on mode
    if args.compare_activations:
        print("Comparing activation bit-widths...")
        results = evaluator.compare_activation_bits(num_samples or 100)
        
        print("\nComparison Results:")
        create_results_table(results)
        
        if args.save_results:
            results_path = os.path.join(args.output_dir, "activation_comparison.json")
            with open(results_path, 'w') as f:
                # Convert results to JSON-serializable format
                json_results = {}
                for key, result in results.items():
                    json_results[key] = {
                        'perplexity': result.perplexity,
                        'arc_challenge': result.arc_challenge,
                        'arc_easy': result.arc_easy,
                        'hellaswag': result.hellaswag,
                        'piqa': result.piqa,
                        'winogrande': result.winogrande,
                        'lambada': result.lambada,
                        'average': result.average
                    }
                json.dump(json_results, f, indent=2)
            print(f"Results saved to {results_path}")
    
    elif args.benchmark_speed:
        print("Benchmarking inference speed...")
        speed_results = evaluator.benchmark_inference_speed()
        
        print("\nSpeed Benchmark Results:")
        for key, result in speed_results.items():
            print(f"{key}: {result['avg_time']:.4f}s, {result['throughput']:.1f} tokens/s")
        
        if args.save_results:
            speed_path = os.path.join(args.output_dir, "speed_benchmark.json")
            with open(speed_path, 'w') as f:
                json.dump(speed_results, f, indent=2)
            print(f"Speed results saved to {speed_path}")
    
    elif args.generate_samples:
        print("Generating sample outputs...")
        prompts = [
            "The quick brown fox",
            "In a world where artificial intelligence",
            "The meaning of life is",
            "Once upon a time in a distant galaxy",
            "The most important discovery in science"
        ]
        
        outputs = evaluator.generate_sample_outputs(prompts, max_length=50)
        
        print("\nSample Outputs:")
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print("-" * 50)
        
        if args.save_results:
            samples_path = os.path.join(args.output_dir, "sample_outputs.json")
            sample_data = [{"prompt": p, "output": o} for p, o in zip(prompts, outputs)]
            with open(samples_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            print(f"Sample outputs saved to {samples_path}")
    
    else:
        # Standard evaluation
        print("Running comprehensive evaluation...")
        
        if args.tasks != "all":
            # Run specific tasks
            task_list = get_task_list(args.tasks)
            results = {}
            
            # Calculate perplexity
            results['perplexity'] = evaluator.calculate_perplexity(num_samples=perplexity_samples)
            
            # Run selected tasks
            task_scores = []
            for task_name in task_list:
                score = evaluator.evaluate_task(task_name, num_samples)
                results[task_name] = score * 100
                task_scores.append(score * 100)
                print(f"{task_name}: {score * 100:.2f}%")
            
            # Calculate average
            results['average'] = sum(task_scores) / len(task_scores) if task_scores else 0.0
            
        else:
            # Run full evaluation
            results = evaluator.run_full_evaluation(num_samples, perplexity_samples)
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Perplexity: {results.perplexity:.2f}")
        print(f"ARC Challenge: {results.arc_challenge:.2f}%")
        print(f"ARC Easy: {results.arc_easy:.2f}%")
        print(f"HellaSwag: {results.hellaswag:.2f}%")
        print(f"PIQA: {results.piqa:.2f}%")
        print(f"WinoGrande: {results.winogrande:.2f}%")
        print(f"LAMBADA: {results.lambada:.2f}%")
        print(f"Average: {results.average:.2f}%")
        print("=" * 60)
        
        # Save results
        if args.save_results:
            results_path = os.path.join(args.output_dir, "evaluation_results.json")
            results_dict = {
                'perplexity': results.perplexity,
                'arc_challenge': results.arc_challenge,
                'arc_easy': results.arc_easy,
                'hellaswag': results.hellaswag,
                'piqa': results.piqa,
                'winogrande': results.winogrande,
                'lambada': results.lambada,
                'average': results.average,
                'metadata': {
                    'model_path': args.model_path,
                    'model_size': args.model_size,
                    'num_samples': num_samples,
                    'perplexity_samples': perplexity_samples,
                    'tasks': args.tasks
                }
            }
            
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
