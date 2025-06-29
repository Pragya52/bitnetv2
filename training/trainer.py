"""
Training utilities for BitNet v2
Following the exact training protocol from the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import time
from typing import Dict, Optional
import wandb
from dataclasses import dataclass

@dataclass
class BitNetTrainingConfig:
    """Training configuration following the paper"""
    
    def __init__(self, model_size='1.3B'):
        self.model_size = model_size
        
        # Model configurations from Table 8
        self.model_configs = {
            '400M': {
                'hidden_size': 1024,
                'intermediate_size': 4096,
                'num_attention_heads': 16,
                'num_hidden_layers': 24,
                'batch_size': 1_000_000,  # 1M tokens per batch
            },
            '1.3B': {
                'hidden_size': 2048,
                'intermediate_size': 8192,
                'num_attention_heads': 32,
                'num_hidden_layers': 18,
                'batch_size': 1_000_000,
            },
            '3B': {
                'hidden_size': 4096,
                'intermediate_size': 8192,
                'num_attention_heads': 32,
                'num_hidden_layers': 20,
                'batch_size': 1_000_000,
            },
            '7B': {
                'hidden_size': 4096,
                'intermediate_size': 16384,
                'num_attention_heads': 32,
                'num_hidden_layers': 24,
                'batch_size': 1_000_000,
            }
        }
        
        # Training hyperparameters from Table 9
        self.lr_configs = {
            '400M': {'initial_lr': 1.8e-3, 'final_lr': 1.2e-3},
            '1.3B': {'initial_lr': 1.2e-3, 'final_lr': 8e-4},
            '3B': {'initial_lr': 1.2e-3, 'final_lr': 6.4e-4},
            '7B': {'initial_lr': 1e-3, 'final_lr': 6e-4},
        }
        
        # Common settings
        self.total_tokens = 100_000_000_000  # 100B tokens
        self.stage1_tokens = 95_000_000_000   # 95B tokens with 8-bit activations
        self.stage2_tokens = 5_000_000_000    # 5B tokens with 4-bit activations
        
        self.weight_decay_initial = 0.1
        self.weight_decay_final = 0.0
        self.warmup_steps = 375
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.max_grad_norm = 1.0
        self.seq_length = 2048

class BitNetTrainer:
    """Enhanced trainer following the paper's training protocol"""
    
    def __init__(self, config: BitNetTrainingConfig, model, tokenizer, device='cuda', use_wandb=False):
        self.config = config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.step = 0
        self.stage = 1  # 1 for 8-bit, 2 for 4-bit activations
        self.total_loss = 0.0
        
        # Calculate steps
        tokens_per_step = config.model_configs[config.model_size]['batch_size']
        self.stage1_steps = config.stage1_tokens // tokens_per_step
        self.stage2_steps = config.stage2_tokens // tokens_per_step
        self.total_steps = self.stage1_steps + self.stage2_steps
        
        if use_wandb:
            wandb.init(
                project=f"bitnet-v2-{config.model_size}",
                config=vars(config)
            )
    
    def _create_optimizer(self):
        """Create optimizer with weight decay scheduling"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay_initial},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        lr_config = self.config.lr_configs[self.config.model_size]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr_config['initial_lr'],
            betas=(self.config.adam_beta1, self.config.adam_beta2)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with two-stage decay"""
        lr_config = self.config.lr_configs[self.config.model_size]
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Warmup
                return step / self.config.warmup_steps
            elif step < self.stage1_steps:
                # Stage 1: Cosine decay
                progress = (step - self.config.warmup_steps) / (self.stage1_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
            else:
                # Stage 2: Continue from stage 1 final LR
                stage2_progress = (step - self.stage1_steps) / self.stage2_steps
                final_ratio = lr_config['final_lr'] / lr_config['initial_lr']
                return final_ratio * (0.5 * (1 + math.cos(math.pi * stage2_progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _update_weight_decay(self):
        """Update weight decay according to schedule"""
        progress = self.step / self.total_steps
        current_weight_decay = (
            self.config.weight_decay_initial * (1 - progress) + 
            self.config.weight_decay_final * progress
        )
        
        for param_group in self.optimizer.param_groups:
            if param_group['weight_decay'] > 0:
                param_group['weight_decay'] = current_weight_decay
    
    def switch_to_4bit_activations(self):
        """Switch to 4-bit activations for stage 2"""
        print("Switching to 4-bit activations...")
        self.stage = 2
        
        # Update all activation quantizers
        for module in self.model.modules():
            if hasattr(module, 'activation_quantizer'):
                module.activation_quantizer.bits = 4
        
        print("Now using 4-bit activations")
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update weight decay
        self._update_weight_decay()
        
        # Update step counter
        self.step += 1
        self.total_loss += loss.item()
        
        # Check if we need to switch to stage 2
        if self.step == self.stage1_steps and self.stage == 1:
            self.switch_to_4bit_activations()
        
        return loss.item()
    
    def evaluate(self, eval_dataloader, num_batches=100):
        """Evaluation step"""
        self.model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss']
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        
        return total_loss / num_samples if num_samples > 0 else float('inf')
    
    def save_checkpoint(self, path):
        """Save training checkpoint"""
        checkpoint = {
            'step': self.step,
            'stage': self.stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_loss': self.total_loss,
            'config': vars(self.config)
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step = checkpoint['step']
        self.stage = checkpoint['stage']
        self.total_loss = checkpoint['total_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # If we're in stage 2, make sure activations are 4-bit
        if self.stage == 2:
            for module in self.model.modules():
                if hasattr(module, 'activation_quantizer'):
                    module.activation_quantizer.bits = 4
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self, train_dataset, eval_dataset=None, save_dir="checkpoints", eval_interval=1000, save_interval=5000):
        """Main training loop"""
        print(f"Starting training for {self.config.model_size} model")
        print(f"Total steps: {self.total_steps}")
        print(f"Stage 1 (8-bit): {self.stage1_steps} steps")
        print(f"Stage 2 (4-bit): {self.stage2_steps} steps")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loader
        from torch.utils.data import DataLoader
        from .dataset import collate_fn
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=8,  # Actual batch size (will accumulate to reach target tokens)
            collate_fn=collate_fn,
            num_workers=4
        )
        
        eval_dataloader = None
        if eval_dataset:
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=8,
                collate_fn=collate_fn,
                num_workers=2
            )
        
        start_time = time.time()
        data_iter = iter(dataloader)
        
        while self.step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Training step
            loss = self.train_step(batch)
            
            # Logging
            if self.step % 100 == 0:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]
                stage_name = "8-bit" if self.stage == 1 else "4-bit"
                
                print(f"Step {self.step}/{self.total_steps} ({stage_name}) | "
                      f"Loss: {loss:.4f} | LR: {lr:.2e} | "
                      f"Time: {elapsed:.1f}s")
                
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss,
                        'train/learning_rate': lr,
                        'train/stage': self.stage,
                        'train/step': self.step
                    })
            
            # Evaluation
            if eval_dataloader and self.step % eval_interval == 0:
                eval_loss = self.evaluate(eval_dataloader)
                perplexity = math.exp(eval_loss)
                
                print(f"Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
                
                if self.use_wandb:
                    wandb.log({
                        'eval/loss': eval_loss,
                        'eval/perplexity': perplexity
                    })
            
            # Save checkpoint
            if self.step % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{self.step}.pt")
                self.save_checkpoint(checkpoint_path)
        
        # Final checkpoint
        final_checkpoint_path = os.path.join(save_dir, "final_checkpoint.pt")
        self.save_checkpoint(final_checkpoint_path)
        
        print("Training completed!")
