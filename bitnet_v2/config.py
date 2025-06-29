"""
Configuration classes and utilities for BitNet v2
"""

import yaml
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class BitNetConfig:
    """Configuration for BitNet v2 model"""
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 18
    num_attention_heads: int = 32
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_4bit_activations: bool = False
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"

def get_model_config(model_size: str) -> BitNetConfig:
    """Get model configuration for different sizes"""
    
    configs = {
        '400M': BitNetConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24
        ),
        '1.3B': BitNetConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=32,
            num_hidden_layers=18
        ),
        '3B': BitNetConfig(
            hidden_size=4096,
            intermediate_size=8192,
            num_attention_heads=32,
            num_hidden_layers=20
        ),
        '7B': BitNetConfig(
            hidden_size=4096,
            intermediate_size=16384,
            num_attention_heads=32,
            num_hidden_layers=24
        )
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from: {list(configs.keys())}")
    
    return configs[model_size]

def load_config_from_yaml(config_path: str) -> BitNetConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return BitNetConfig(**config_dict)

def save_config_to_yaml(config: BitNetConfig, config_path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'vocab_size': config.vocab_size,
        'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size,
        'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'max_position_embeddings': config.max_position_embeddings,
        'rms_norm_eps': config.rms_norm_eps,
        'rope_theta': config.rope_theta,
        'use_4bit_activations': config.use_4bit_activations
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
