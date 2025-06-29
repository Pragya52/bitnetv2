"""
Main training script for BitNet v2
Supports two-stage training as described in the paper
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_v2 import BitNetV2ForCausalLM, get_model_config
from training.trainer import BitNetTrainer, BitNetTrainingConfig
from training.dataset import get_dataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BitNet v2 model")
    
    # Model configuration
    parser.add_argument("--model_size", choices=['400M', '1.3B', '3B', '7B'], 
                       default='1.3B', help="Model size")
    parser.add_argument("--vocab_size", type=int, default=32000, 
                       help="Vocabulary size")
    
    # Training configuration
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                       help="Training stage (1: 8-bit activations, 2: 4-bit activations)")
    parser.add_argument("--dataset", type=str, default="redpajama",
                       choices=["redpajama", "c4", "wikitext"],
                       help="Dataset to use for training")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save frequency")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps (overrides token count)")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Checkpoint path to resume from")
    parser.add_argument("--save_model", action='store_true',
                       help="Save final model")
    
    # Hardware and optimization
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--mixed_precision", action='store_true',
                       help="Use mixed precision training")
    parser.add_argument("--compile", action='store_true',
                       help="Compile model with torch.compile")
    
    # Logging and monitoring
    parser.add_argument("--use_wandb", action='store_true',
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="bitnet-v2",
                       help="W&B project name")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Testing and debugging
    parser.add_argument("--quick_test", action='store_true',
                       help="Run a quick test with small dataset")
    parser.add_argument("--eval_only", action='store_true',
                       help="Only run evaluation")
    parser.add_argument("--dry_run", action='store_true',
                       help="Dry run without actual training")
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup device for training"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory // 1024**3} GB")
    
    return device

def setup_tokenizer():
    """Setup tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    except:
        # Fallback tokenizer
        print("Warning: Using basic tokenizer")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def create_model(args, device):
    """Create and initialize model"""
    print(f"Creating {args.model_size} BitNet v2 model...")
    
    # Get model configuration
    config = get_model_config(args.model_size)
    config.vocab_size = args.vocab_size
    config.max_position_embeddings = args.max_length
    
    # Set activation bits based on stage
    if args.stage == 2:
        config.use_4bit_activations = True
    
    # Create model
    model = BitNetV2ForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Compile model if requested
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    return model.to(device)

def create_datasets(args, tokenizer):
    """Create training and evaluation datasets"""
    print(f"Loading {args.dataset} dataset...")
    
    if args.quick_test:
        # Use small subset for testing
        train_dataset = get_dataset(args.dataset, tokenizer, split='train', 
                                  max_length=args.max_length, subset=1000)
        eval_dataset = get_dataset(args.dataset, tokenizer, split='validation',
                                 max_length=args.max_length, subset=100)
    else:
        train_dataset = get_dataset(args.dataset, tokenizer, split='train',
                                  max_length=args.max_length)
        eval_dataset = get_dataset(args.dataset, tokenizer, split='validation',
                                 max_length=args.max_length, max_samples=10000)
    
    return train_dataset, eval_dataset

def main():
    """Main training function"""
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    tokenizer = setup_tokenizer()
    
    # Create model
    model = create_model(args, device)
    
    # Create datasets
    train_dataset, eval_dataset = create_datasets(args, tokenizer)
    
    # Create training configuration
    training_config = BitNetTrainingConfig(args.model_size)
    
    # Override max_steps if provided
    if args.max_steps:
        training_config.total_steps = args.max_steps
        training_config.stage1_steps = args.max_steps // 2
        training_config.stage2_steps = args.max_steps // 2
    
    # Create trainer
    trainer = BitNetTrainer(
        config=training_config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if provided
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        print(f"Resumed from checkpoint: {args.resume_from}")
    
    # Set stage
    if args.stage == 2:
        trainer.switch_to_4bit_activations()
    
    if args.dry_run:
        print("Dry run completed successfully!")
        return
    
    if args.eval_only:
        print("Running evaluation only...")
        eval_loss = trainer.evaluate(eval_dataset)
        perplexity = torch.exp(torch.tensor(eval_loss))
        print(f"Evaluation Loss: {eval_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        return
    
    # Start training
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        save_dir=args.output_dir,
        eval_interval=args.eval_steps,
        save_interval=args.save_steps
    )
    
    # Save final model if requested
    if args.save_model:
        model_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Final model saved to {model_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
