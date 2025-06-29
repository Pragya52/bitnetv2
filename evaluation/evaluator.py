"""
Comprehensive evaluation suite for BitNet v2
Implements all evaluation tasks from the paper
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import load_dataset
from .tasks import (
    ARCChallengeTask, ARCEasyTask, HellaSwagTask,
    PIQATask, WinoGrandeTask, LAMBADATask
)

@dataclass
class EvaluationResults:
    """Store evaluation results"""
    perplexity: float
    arc_challenge: float
    arc_easy: float
    hellaswag: float
    piqa: float
    winogrande: float
    lambada: float
    average: float

class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for BitNet v2"""
    
    def __init__(self, model, tokenizer, device='cuda', max_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Initialize task evaluators
        self.tasks = {
            'arc_challenge': ARCChallengeTask(model, tokenizer, device),
            'arc_easy': ARCEasyTask(model, tokenizer, device),
            'hellaswag': HellaSwagTask(model, tokenizer, device),
            'piqa': PIQATask(model, tokenizer, device),
            'winogrande': WinoGrandeTask(model, tokenizer, device),
            'lambada': LAMBADATask(model, tokenizer, device)
        }
    
    def calculate_perplexity(self, dataset_name="c4", split="validation", num_samples=1000):
        """Calculate perplexity on a dataset"""
        print(f"Calculating perplexity on {dataset_name} {split}...")
        
        try:
            # Load dataset
            if dataset_name == "c4":
                dataset = load_dataset("c4", "en", split=split, streaming=True)
            elif dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            total_loss = 0
            total_tokens = 0
            processed_samples = 0
            
            with torch.no_grad():
                for item in dataset:
                    if processed_samples >= num_samples:
                        break
                    
                    text = item['text'] if 'text' in item else str(item)
                    
                    # Skip empty or very short texts
                    if len(text.strip()) < 10:
                        continue
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(text, max_length=self.max_length, 
                                                 truncation=True, add_special_tokens=True)
                    if len(tokens) < 2:
                        continue
                    
                    tokens = torch.tensor([tokens]).to(self.device)
                    
                    # Calculate loss
                    outputs = self.model(tokens, labels=tokens)
                    loss = outputs['loss']
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item() * (tokens.size(1) - 1)
                        total_tokens += tokens.size(1) - 1
                        processed_samples += 1
                    
                    if processed_samples % 100 == 0:
                        current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                        print(f"Processed {processed_samples}/{num_samples} samples, PPL: {current_ppl:.2f}")
            
            if total_tokens == 0:
                return float('inf')
            
            perplexity = math.exp(total_loss / total_tokens)
            print(f"Final Perplexity: {perplexity:.2f}")
            return perplexity
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def evaluate_task(self, task_name: str, num_samples: Optional[int] = None) -> float:
        """Evaluate a specific task"""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        print(f"Evaluating {task_name}...")
        return self.tasks[task_name].evaluate(num_samples)
    
    def run_full_evaluation(self, num_samples_per_task: Optional[int] = None, 
                          perplexity_samples: int = 1000) -> EvaluationResults:
        """Run complete evaluation suite"""
        print("Starting comprehensive evaluation...")
        print("=" * 50)
        
        results = {}
        
        # Calculate perplexity
        results['perplexity'] = self.calculate_perplexity(num_samples=perplexity_samples)
        
        # Run all tasks
        task_results = {}
        for task_name in self.tasks.keys():
            try:
                accuracy = self.evaluate_task(task_name, num_samples_per_task)
                task_results[task_name] = accuracy * 100  # Convert to percentage
                print(f"{task_name}: {accuracy * 100:.2f}%")
            except Exception as e:
                print(f"Error evaluating {task_name}: {e}")
                task_results[task_name] = 0.0
        
        # Store results
        results.update(task_results)
        
        # Calculate average (excluding perplexity)
        task_scores = list(task_results.values())
        results['average'] = sum(task_scores) / len(task_scores) if task_scores else 0.0
        
        return EvaluationResults(**results)
    
    def compare_activation_bits(self, num_samples_per_task: int = 100) -> Dict[str, EvaluationResults]:
        """Compare performance with different activation bit-widths"""
        print("Comparing activation bit-widths...")
        
        results = {}
        
        # Test 8-bit activations
        print("Testing with 8-bit activations...")
        self._set_activation_bits(8)
        results['8-bit'] = self.run_full_evaluation(num_samples_per_task)
        
        # Test 4-bit activations
        print("Testing with 4-bit activations...")
        self._set_activation_bits(4)
        results['4-bit'] = self.run_full_evaluation(num_samples_per_task)
        
        return results
    
    def _set_activation_bits(self, bits: int):
        """Set activation quantization bits for all modules"""
        for module in self.model.modules():
            if hasattr(module, 'activation_quantizer'):
                module.activation_quantizer.bits = bits
    
    def generate_sample_outputs(self, prompts: List[str], max_length: int = 50) -> List[str]:
        """Generate sample outputs for qualitative evaluation"""
        self.model.eval()
        outputs = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Generate
                generated = self.model.generate(
                    tokens, 
                    max_length=tokens.size(1) + max_length,
                    temperature=0.8,
                    do_sample=True
                )
                
                # Decode
                output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                outputs.append(output_text)
        
        return outputs
    
    def benchmark_inference_speed(self, sequence_lengths: List[int] = [128, 512, 1024, 2048],
                                batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict:
        """Benchmark inference speed"""
        print("Benchmarking inference speed...")
        
        results = {}
        self.model.eval()
        
        with torch.no_grad():
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    # Create dummy input
                    input_ids = torch.randint(0, self.tokenizer.vocab_size, 
                                            (batch_size, seq_len)).to(self.device)
                    
                    # Warmup
                    for _ in range(3):
                        _ = self.model(input_ids)
                    
                    # Benchmark
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                    end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                    
                    if self.device.type == 'cuda':
                        start_time.record()
                    
                    num_runs = 10
                    for _ in range(num_runs):
                        _ = self.model(input_ids)
                    
                    if self.device.type == 'cuda':
                        end_time.record()
                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                    else:
                        import time
                        start = time.time()
                        for _ in range(num_runs):
                            _ = self.model(input_ids)
                        elapsed_time = time.time() - start
                    
                    avg_time = elapsed_time / num_runs
                    throughput = (batch_size * seq_len) / avg_time  # tokens per second
                    
                    key = f"bs{batch_size}_seq{seq_len}"
                    results[key] = {
                        'avg_time': avg_time,
                        'throughput': throughput,
                        'batch_size': batch_size,
                        'sequence_length': seq_len
                    }
                    
                    print(f"Batch size: {batch_size}, Seq length: {seq_len}, "
                          f"Time: {avg_time:.4f}s, Throughput: {throughput:.1f} tokens/s")
        
        return results

def create_results_table(results_dict: Dict[str, EvaluationResults]):
    """Create a formatted results table like in the paper"""
    print("\nResults Table:")
    print("=" * 80)
    print(f"{'Model':<15} {'PPL↓':<8} {'ARCc↑':<8} {'ARCe↑':<8} {'HS↑':<8} "
          f"{'PQ↑':<8} {'WGe↑':<8} {'LBA↑':<8} {'Avg↑':<8}")
    print("-" * 80)
    
    for model_name, results in results_dict.items():
        print(f"{model_name:<15} {results.perplexity:<8.2f} {results.arc_challenge:<8.2f} "
              f"{results.arc_easy:<8.2f} {results.hellaswag:<8.2f} {results.piqa:<8.2f} "
              f"{results.winogrande:<8.2f} {results.lambada:<8.2f} {results.average:<8.2f}")

def compare_models(models_dict: Dict[str, torch.nn.Module], tokenizer, 
                  device='cuda', num_samples=100) -> Dict[str, EvaluationResults]:
    """Compare multiple models on all tasks"""
    print("Comparing models...")
    print("=" * 60)
    
    all_results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        evaluator = ComprehensiveEvaluator(model, tokenizer, device)
        results = evaluator.run_full_evaluation(num_samples_per_task=num_samples)
        all_results[model_name] = results
        
        print(f"\nResults for {model_name}:")
        print(f"Perplexity: {results.perplexity:.2f}")
        print(f"Average accuracy: {results.average:.2f}%")
        print("-" * 40)
    
    create_results_table(all_results)
    return all_results
