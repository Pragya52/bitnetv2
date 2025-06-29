"""
Quantization utilities for BitNet v2
Includes weight and activation quantization, and Hadamard transformation
"""

import torch
import torch.nn as nn
import math

def round_clip(x, a, b):
    """Round and clip function as defined in the paper"""
    return torch.clamp(torch.round(x), a, b)

def hadamard_transform(x):
    """
    Fast Hadamard Transform implementation
    Assumes x has shape (..., n) where n is a power of 2
    """
    # Try to use the fast implementation if available
    try:
        from fast_hadamard_transform import hadamard_transform as fast_hadamard
        return fast_hadamard(x)
    except ImportError:
        # Fallback to manual implementation
        pass
    
    n = x.shape[-1]
    if n == 1:
        return x
    
    # Ensure n is power of 2
    assert n & (n - 1) == 0, f"Last dimension must be power of 2, got {n}"
    
    x = x.clone()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[..., j]
                v = x[..., j + h]
                x[..., j] = u + v
                x[..., j + h] = u - v
        h *= 2
    
    return x / math.sqrt(n)

class WeightQuantizer(nn.Module):
    """Weight quantization to ternary values {-1, 0, 1}"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, weight):
        # Per-tensor absmean quantization
        alpha = torch.mean(torch.abs(weight))
        eps = 1e-8
        
        # Quantize to {-1, 0, 1}
        normalized_weight = weight / (alpha + eps)
        quantized_weight = round_clip(normalized_weight, -1, 1)
        
        # Scale back
        return alpha * quantized_weight

class ActivationQuantizer(nn.Module):
    """Activation quantization for INT8 and INT4"""
    
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        
    def forward(self, x):
        if self.bits == 8:
            # Per-token absmax quantization for INT8
            gamma = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
            eps = 1e-8
            
            normalized_x = 127 * x / (gamma + eps)
            quantized_x = round_clip(normalized_x, -128, 127)
            
            return gamma * quantized_x / 127
            
        elif self.bits == 4:
            # Per-token absmean quantization for INT4
            beta = torch.mean(torch.abs(x), dim=-1, keepdim=True)
            eps = 1e-8
            
            normalized_x = math.sqrt(7) * x / (beta + eps)
            quantized_x = round_clip(normalized_x, -8, 7)
            
            return beta * quantized_x / math.sqrt(7)
        
        else:
            raise ValueError(f"Unsupported bit width: {self.bits}")

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
