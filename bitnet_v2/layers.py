"""
Custom layers for BitNet v2
Includes BitLinear, H-BitLinear, RMSNorm, and RotaryEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .quantization import WeightQuantizer, ActivationQuantizer, hadamard_transform

class BitLinear(nn.Module):
    """Standard BitLinear layer for 1.58-bit weights"""
    
    def __init__(self, in_features, out_features, bias=False, activation_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Quantizers
        self.weight_quantizer = WeightQuantizer()
        self.activation_quantizer = ActivationQuantizer(bits=activation_bits)
        
        # For mixed precision training
        self.register_buffer('weight_scale', torch.ones(1))
        
    def forward(self, x):
        # Quantize activation
        x_quant = self.activation_quantizer(x)
        
        # Quantize weight during forward pass
        if self.training:
            # Use straight-through estimator
            weight_quant = self.weight_quantizer(self.weight)
            weight_quant = weight_quant + self.weight - self.weight.detach()
        else:
            weight_quant = self.weight_quantizer(self.weight)
        
        # Linear transformation
        output = F.linear(x_quant, weight_quant, self.bias)
        return output

class HBitLinear(nn.Module):
    """H-BitLinear with Hadamard transformation for outlier reduction"""
    
    def __init__(self, in_features, out_features, bias=False, activation_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Ensure input features is power of 2 for Hadamard transform
        if in_features & (in_features - 1) != 0:
            # Pad to next power of 2
            self.padded_in_features = 2 ** math.ceil(math.log2(in_features))
            self.needs_padding = True
        else:
            self.padded_in_features = in_features
            self.needs_padding = False
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_features, self.padded_in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Layer normalization before Hadamard transform
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)
        
        # Quantizers
        self.weight_quantizer = WeightQuantizer()
        self.activation_quantizer = ActivationQuantizer(bits=activation_bits)
        
    def forward(self, x):
        # Apply layer normalization
        x_norm = self.layer_norm(x)
        
        # Pad if necessary
        if self.needs_padding:
            pad_size = self.padded_in_features - self.in_features
            x_norm = F.pad(x_norm, (0, pad_size))
        
        # Apply Hadamard transformation
        x_hadamard = hadamard_transform(x_norm)
        
        # Quantize activation
        x_quant = self.activation_quantizer(x_hadamard)
        
        # Quantize weight
        if self.training:
            weight_quant = self.weight_quantizer(self.weight)
            weight_quant = weight_quant + self.weight - self.weight.detach()
        else:
            weight_quant = self.weight_quantizer(self.weight)
        
        # Linear transformation
        output = F.linear(x_quant, weight_quant, self.bias)
        return output

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cos and sin cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len=None):
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )
