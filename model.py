"""
BitNet v2: Core model implementation
Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import BitNetConfig
from .layers import BitLinear, HBitLinear, RMSNorm, RotaryEmbedding
from .quantization import apply_rotary_pos_emb

class BitNetAttention(nn.Module):
    """Multi-head attention with BitLinear and H-BitLinear"""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        
        activation_bits = 4 if config.use_4bit_activations else 8
        
        # Query, Key, Value projections
        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim, 
                               activation_bits=activation_bits)
        self.k_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim,
                               activation_bits=activation_bits)
        self.v_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim,
                               activation_bits=activation_bits)
        
        # Output projection with H-BitLinear
        self.o_proj = HBitLinear(self.num_heads * self.head_dim, self.hidden_size,
                                activation_bits=activation_bits)
        
        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(self.head_dim, 
                                         config.max_position_embeddings,
                                         config.rope_theta)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        
        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embedding
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class BitNetMLP(nn.Module):
    """Feed-forward network with SwishGLU and H-BitLinear"""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        activation_bits = 4 if config.use_4bit_activations else 8
        
        # Gate and up projections
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size,
                                  activation_bits=activation_bits)
        self.up_proj = BitLinear(self.hidden_size, self.intermediate_size,
                                activation_bits=activation_bits)
        
        # Down projection with H-BitLinear
        self.down_proj = HBitLinear(self.intermediate_size, self.hidden_size,
                                   activation_bits=activation_bits)
    
    def forward(self, x):
        # SwishGLU activation
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = F.silu(gate) * up
        
        # Down projection
        output = self.down_proj(intermediate)
        return output

class BitNetDecoderLayer(nn.Module):
    """Single decoder layer"""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        
        # Self attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class BitNetV2Model(nn.Module):
    """BitNet v2 Model"""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            BitNetDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
        
        # Convert attention mask to causal mask
        causal_mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
        causal_mask = causal_mask.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        
        hidden_states = inputs_embeds
        
        # Pass through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, causal_mask, position_ids)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class BitNetV2ForCausalLM(nn.Module):
    """BitNet v2 for Causal Language Modeling"""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        
        self.model = BitNetV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, do_sample=True):
        """Simple generation method"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(generated)
                logits = outputs['logits'][:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token (assuming token_id 2)
                if next_token.item() == 2:
                    break
        
        return generated
      
