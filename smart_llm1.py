#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import pickle
import os
import random
from tqdm import tqdm
import logging
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
@dataclass
class ModelConfig:
    # --- Micro-sized model with all features enabled for CPU testing ---
    vocab_size: int = 16000
    d_model: int = 64                   # DRASTICALLY REDUCED from 512
    n_layers: int = 2                   # DRASTICALLY REDUCED from 8
    n_heads: int = 4                    # REDUCED from 8
    n_kv_heads: int = 2                 # Compatible with n_heads
    d_ff: int = 128                     # DRASTICALLY REDUCED from 1024
    max_seq_len: int = 64               # DRASTICALLY REDUCED from 1024
    dropout: float = 0.1

    # --- Advanced features are KEPT ENABLED as requested ---
    use_rotary_embeddings: bool = True
    use_mixture_of_experts: bool = True     # ENABLED
    num_experts: int = 2                    # REDUCED from 4 to save memory
    num_experts_per_token: int = 1          # REDUCED from 2
    load_balancing_weight: float = 0.01

    sliding_window_size: int = 128
    router_z_loss_weight: float = 0.001
    num_shared_experts: int = 1
    use_flash_attention: bool = True        # ENABLED (will be ignored safely on CPU)
    use_sliding_window_attention: bool = True # ENABLED

    use_rezero: bool = True                 # ENABLED
    layer_drop_prob: float = 0.05
    use_gradient_checkpointing: bool = True # ENABLED



class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class RotaryPositionalEmbedding(nn.Module):
    """Enhanced Rotary Positional Embedding (RoPE) with caching."""
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 4096):
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            self._cached_seq_len = max(seq_len, self.max_seq_len)
            t = torch.arange(self._cached_seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
            
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cache(seq_len, device)
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies rotary embeddings to queries and keys."""
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, Seq, 1, Dim]
    sin = sin.unsqueeze(0).unsqueeze(2) # [1, Seq, 1, Dim]
    q_out = (q * cos) + (torch.cat([-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]], dim=-1) * sin)
    k_out = (k * cos) + (torch.cat([-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]], dim=-1) * sin)
    return q_out, k_out

class SwiGLU(nn.Module):
    """State-of-the-art SwiGLU activation - superior to GatedMLP"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU uses three linear layers instead of GatedMLP's approach
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Up projection  
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: SiLU(W1 @ x) ⊙ (W2 @ x) then W3
        gate = F.silu(self.w1(x))  # Apply SiLU activation to gate
        up = self.w2(x)            # Linear transformation for up
        return self.w3(self.dropout(gate * up))  # Element-wise multiply and project down
class GroupedQueryAttention(nn.Module):
    """
    Optimized GQA with vectorized Sliding Window Attention.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.sliding_window_size = config.sliding_window_size
        self.use_sliding_window = config.use_sliding_window_attention
        
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(config.d_model, config.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(config.n_heads * self.d_head, config.d_model, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.d_head, max_seq_len=config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device):
        """
        Creates a sliding window causal mask in a highly efficient, vectorized way.
        No Python for-loops.
        """
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        q_indices = torch.arange(seq_len, device=device).view(-1, 1)
        k_indices = torch.arange(seq_len, device=device).view(1, -1)
        window_mask = (q_indices > k_indices + self.sliding_window_size)
        return torch.logical_or(causal_mask, window_mask)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.shape
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_kv_heads, self.d_head)
        v = v.view(B, T, self.n_kv_heads, self.d_head)

        cos, sin = self.rotary_emb(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv_cache = (k, v)
        
        # Transpose and repeat for GQA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        
        current_seq_len = k.shape[2]
        is_causal_needed = T > 1
        attn_mask = None
        if self.use_sliding_window and current_seq_len > self.sliding_window_size:
            is_causal_needed = False # The SWA mask includes causality.
            attn_mask = self._create_sliding_window_mask(T, x.device)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal_needed
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(attn_out), new_kv_cache

class MixtureOfExperts(nn.Module):
    """
    CRITICAL FIX: A fully parallel MoE implementation.
    This version avoids all major Python loops for massive speedup.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        self.shared_expert = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.routed_experts = nn.ModuleList([SwiGLU(config.d_model, config.d_ff, config.dropout) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        
        self.load_balancing_weight = config.load_balancing_weight
        self.router_z_loss_weight = config.router_z_loss_weight

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        num_tokens = x_flat.shape[0]

        shared_output = self.shared_expert(x_flat)
        router_logits = self.gate(x_flat)

        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_token, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)

        # Create a flat tensor of tokens and their expert assignments
        flat_selected_experts = selected_experts.view(-1)
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.num_experts_per_token)

        # Initialize final output tensor
        final_routed_output = torch.zeros_like(x_flat)
        
        # Batched processing of experts
        # This loop is over the number of experts (e.g., 4), NOT the number of tokens. This is key.
        for i, expert in enumerate(self.routed_experts):
            # Find all assignments for the current expert
            expert_mask = (flat_selected_experts == i)
            if expert_mask.any():
                # Get the tokens and their original positions for this expert
                tokens_for_expert = x_flat[flat_token_indices[expert_mask]]
                
                # Get the routing weights for these specific assignments
                weights_for_expert = routing_weights.view(-1)[expert_mask].unsqueeze(1)
                
                # Process the batch of tokens through the expert
                expert_out = expert(tokens_for_expert)
                
                # Apply the weights and add the results back to the final output tensor
                final_routed_output.index_add_(0, flat_token_indices[expert_mask], expert_out * weights_for_expert)
        
        # Final combination and loss calculation
        final_output = shared_output + final_routed_output
        aux_loss = self._compute_auxiliary_losses(router_logits) if self.training else torch.tensor(0.0, device=x.device)
        
        return final_output.view(B, T, C), aux_loss

    def _compute_auxiliary_losses(self, router_logits: torch.Tensor):
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        expert_load = router_probs.sum(dim=0)
        expert_fraction = expert_load / expert_load.sum()
        load_balancing_loss = self.num_experts * torch.sum(expert_fraction * expert_fraction) # Simplified and effective
        
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        return (self.load_balancing_weight * load_balancing_loss) + (self.router_z_loss_weight * z_loss)

class TransformerBlock(nn.Module):
    """
    The definitive State-of-the-Art Transformer Block, integrating:
    1. ReZero: Learnable scaling for extremely stable residual connections
    2. LayerDrop: Powerful regularization technique to prevent overfitting
    3. Gradient Checkpointing: Memory-efficient training for large models
    4. SOTA Sub-modules: SwiGLU (instead of GatedMLP), advanced MoE, and GQA
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Store configuration flags
        self.use_rezero = config.use_rezero
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
        # Normalization layers (keeping same names as your original)
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        
        # Attention module (upgraded with sliding window + FlashAttention)
        self.attention = GroupedQueryAttention(config)
        
        # MLP module - now uses SwiGLU instead of GatedMLP for better performance
        if config.use_mixture_of_experts:
            self.mlp = MixtureOfExperts(config)  # Advanced MoE with shared experts
        else:
            # Use SwiGLU instead of GatedMLP - it's proven superior
            self.mlp = SwiGLU(config.d_model, config.d_ff, config.dropout)
        
        # --- 1. ReZero Parameters ---
        # Learnable parameters initialized to zero for each residual connection
        # This allows the model to learn the optimal contribution of each block
        if self.use_rezero:
            self.rezero_alpha_attn = nn.Parameter(torch.zeros(1))
            self.rezero_alpha_mlp = nn.Parameter(torch.zeros(1))
        
        # --- 2. LayerDrop Module ---
        # Randomly "drops" the output of blocks during training for regularization
        self.layer_drop = nn.Dropout(config.layer_drop_prob)
    
    def _forward_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor], 
                          kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        """Helper function for attention block - needed for gradient checkpointing"""
        return self.attention(self.ln1(x), mask, kv_cache)
    
    def _forward_mlp(self, x: torch.Tensor):
        """Helper function for MLP block - needed for gradient checkpointing"""
        if isinstance(self.mlp, MixtureOfExperts):
            return self.mlp(self.ln2(x))
        else:
            return self.mlp(self.ln2(x)), torch.tensor(0.0, device=x.device)
    
    def forward(self, x, mask=None, kv_cache=None):
        """
        Forward pass with all advanced features:
        - ReZero for stable training
        - LayerDrop for regularization
        - Gradient checkpointing for memory efficiency
        """
        aux_loss = torch.tensor(0.0, device=x.device)
        
        # --- ATTENTION BLOCK ---
        residual = x
        
        # Use gradient checkpointing during training to save memory
        use_checkpoint = self.training and self.use_gradient_checkpointing
        
        if use_checkpoint:
            # Gradient checkpointing: trade compute for memory
            attn_out, new_kv_cache = checkpoint(
                self._forward_attention, x, mask, kv_cache, 
                use_reentrant=False
            )
        else:
            # Normal forward pass
            attn_out, new_kv_cache = self._forward_attention(x, mask, kv_cache)
        
        # Apply LayerDrop: randomly zero out the output during training
        if self.training:
            attn_out = self.layer_drop(attn_out)
        
        # Apply ReZero scaling or standard residual connection
        if self.use_rezero:
            # ReZero: learnable scaling (starts at 0, grows during training)
            x = residual + self.rezero_alpha_attn * attn_out
        else:
            # Standard residual connection
            x = residual + attn_out
        
        # --- MLP/MOE BLOCK ---
        residual = x
        
        if use_checkpoint:
            # Gradient checkpointing for MLP block
            mlp_out, mlp_aux_loss = checkpoint(
                self._forward_mlp, x, 
                use_reentrant=False
            )
        else:
            # Normal forward pass
            if isinstance(self.mlp, MixtureOfExperts):
                mlp_out, mlp_aux_loss = self.mlp(self.ln2(x))
            else:
                mlp_out = self.mlp(self.ln2(x))
                mlp_aux_loss = torch.tensor(0.0, device=x.device)
        
        # Accumulate auxiliary loss from MoE
        aux_loss = aux_loss + mlp_aux_loss
        
        # Apply LayerDrop to MLP output
        if self.training:
            mlp_out = self.layer_drop(mlp_out)
        
        # Apply ReZero scaling or standard residual connection
        if self.use_rezero:
            # ReZero: learnable scaling for MLP block
            x = residual + self.rezero_alpha_mlp * mlp_out
        else:
            # Standard residual connection
            x = residual + mlp_out
        
        return x, new_kv_cache, aux_loss


class SmartLLM(nn.Module):
    """Advanced Smart LLM with emergent capabilities and state-of-the-art enhancements"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Stacking the state-of-the-art Transformer Blocks
        # The TransformerBlock itself now contains ReZero, LayerDrop, SWA, etc.
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final normalization layer (using more descriptive name)
        self.ln_f = RMSNorm(config.d_model)
        
        # Final language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying for efficiency
        self.lm_head.weight = self.token_embedding.weight
        
        # Apply state-of-the-art weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        State-of-the-art weight initialization (inspired by Llama).
        This is crucial for stable training of deep networks.
        """
        if isinstance(module, nn.Linear):
            # Initialize weights with a normal distribution, but scale std by num_layers
            # This prevents activations from exploding in the residual stream.
            std = 0.02 / math.sqrt(2.0 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, kv_caches=None):
        """
        Enhanced forward pass with proper KV cache handling and auxiliary loss aggregation
        """
        B, T = tokens.shape
        x = self.token_embedding(tokens)
        
        # Initialize auxiliary loss as tensor on correct device
        total_aux_loss = torch.tensor(0.0, device=tokens.device)
        
        # Initialize KV caches if this is the first pass
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers
            
        new_kv_caches = []
        
        # Efficiently process all transformer layers
        for i, layer in enumerate(self.layers):
            x, new_kv, aux_loss = layer(x, kv_cache=kv_caches[i])
            total_aux_loss += aux_loss
            new_kv_caches.append(new_kv)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Use proper ignore index for padding tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            # Add the auxiliary loss from the MoE layers
            if total_aux_loss.item() > 0:
                loss += total_aux_loss
        
        # Return KV caches consistently
        return logits, loss, new_kv_caches

    def resize_token_embeddings(self, new_vocab_size):
        """
        Resize token embeddings when vocabulary is expanded.
        Essential for adding new tokens during fine-tuning.
        """
        old_vocab_size = self.config.vocab_size
        self.config.vocab_size = new_vocab_size
        
        # Create new embedding layer
        new_token_embedding = nn.Embedding(new_vocab_size, self.config.d_model)
        new_lm_head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
        
        # Copy old weights
        with torch.no_grad():
            new_token_embedding.weight[:old_vocab_size] = self.token_embedding.weight
            new_lm_head.weight[:old_vocab_size] = self.lm_head.weight
            
            # Initialize new token embeddings
            if new_vocab_size > old_vocab_size:
                torch.nn.init.normal_(
                    new_token_embedding.weight[old_vocab_size:], 
                    mean=0.0, std=0.02
                )
                torch.nn.init.normal_(
                    new_lm_head.weight[old_vocab_size:], 
                    mean=0.0, std=0.02
                )
        
        # Replace layers
        self.token_embedding = new_token_embedding
        self.lm_head = new_lm_head
        
        # Maintain weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        logger.info(f"Resized token embeddings from {old_vocab_size} to {new_vocab_size}")


class SmartBPETokenizer:
    """Advanced BPE Tokenizer with Byte-Level BPE for universal robustness"""

    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        # Special tokens are added during training
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<system>"]
        
    def train_from_texts(self, texts, save_path="smart-bpe-tokenizer.json"):
        """Train BPE tokenizer with state-of-the-art Byte-Level BPE from a list of texts"""
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required for BPE tokenizer")
        
        # 1. Initialize a BPE Model
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # 2. State-of-the-Art Normalization and Pre-tokenization
        # NFD Unicode normalization + Lowercase + Strip Accents
        self.tokenizer.normalizer = normalizers.Sequence([
            NFD(), 
            Lowercase(), 
            StripAccents()
        ])
        
        # ByteLevel pre-tokenizer is the KEY to handling all possible text.
        # This ensures the tokenizer can handle any character without <unk> tokens
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(),
            pre_tokenizers.Digits(individual_digits=True)
        ])
        
        # 3. Train the Model
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2
        )
        
        self.tokenizer.train_from_iterator(texts, trainer)
        self.tokenizer.save(save_path)
        logger.info(f"BPE tokenizer trained with {self.tokenizer.get_vocab_size()} tokens and saved to {save_path}")
        
    def load(self, path="smart-bpe-tokenizer.json"):
        """Load trained tokenizer"""
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required for BPE tokenizer")
            
        if os.path.exists(path):
            self.tokenizer = Tokenizer.from_file(path)
            logger.info(f"Loaded BPE tokenizer from {path}")
        else:
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
    
    def encode(self, text):
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens):
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.decode(tokens)
    
    def get_special_token_id(self, token):
        """Get ID of special token"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.token_to_id(token)
    
    def expand_vocabulary(self, new_tokens: list, model: SmartLLM):
        """
        Correctly adds new tokens to the vocabulary and resizes the model's embeddings.
        This is crucial when adding new tokens during fine-tuning.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded or trained first.")
        
        # Add new tokens to the tokenizer
        num_added = self.tokenizer.add_tokens(new_tokens)
        
        if num_added > 0:
            # IMPORTANT: Resize the model's token embedding layer to match
            model.resize_token_embeddings(self.tokenizer.get_vocab_size())
            logger.info(f"Vocabulary expanded by {num_added} tokens to {self.tokenizer.get_vocab_size()} total tokens.")
        else:
            logger.info("No new tokens were added (all tokens already in vocabulary).")
        
        return num_added
    
    def get_vocab_size(self):
        """Get current vocabulary size"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.get_vocab_size()
    
    def add_special_tokens(self, special_tokens_dict: dict, model: SmartLLM):
        """
        Add special tokens like pad_token, eos_token, etc.
        
        Args:
            special_tokens_dict: Dict with keys like 'pad_token', 'eos_token', etc.
            model: SmartLLM model to resize
        """
        new_tokens = []
        for token_type, token in special_tokens_dict.items():
            if token not in self.special_tokens:
                new_tokens.append(token)
                self.special_tokens.append(token)
        
        if new_tokens:
            return self.expand_vocabulary(new_tokens, model)
        return 0


# Fallback character tokenizer for when tokenizers library is not available
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import math
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Check for Lion optimizer availability
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

# Check for tokenizers availability
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
# A constant for the ignore index in the loss function
IGNORE_INDEX = -100

class FallbackCharTokenizer:
    """Fallback character-level tokenizer with save/load functionality and enhanced robustness"""
    def __init__(self, vocab_size=32000):  # vocab_size is mostly for API compatibility
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<user>': 4, '<assistant>': 5, '<system>': 6
        }
        self.token_to_char = {}
        self.char_to_token = {}
        self.build_vocab()
    
    def build_vocab(self):
        """Build vocabulary from ASCII characters with proper mapping"""
        # Invert special tokens for the token_to_char map
        self.token_to_char = {v: k for k, v in self.special_tokens.items()}
        self.char_to_token = {k: v for k, v in self.special_tokens.items()}
        
        current_idx = len(self.special_tokens)
        # Build vocab from the first 256 ASCII characters
        for i in range(256):
            char = chr(i)
            if char not in self.char_to_token:
                self.token_to_char[current_idx] = char
                self.char_to_token[char] = current_idx
                current_idx += 1
        self.vocab_size = len(self.token_to_char)
    
    def save(self, path: str):
        """Saves the tokenizer's vocabulary to a JSON file"""
        vocab_data = {
            'char_to_token': self.char_to_token,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Fallback tokenizer saved to {path}")

    def load(self, path: str):
        """Loads a tokenizer's vocabulary from a JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.char_to_token = vocab_data['char_to_token']
        self.special_tokens = vocab_data['special_tokens']
        # Rebuild the inverse mapping
        self.token_to_char = {int(v): k for k, v in self.char_to_token.items()}
        self.vocab_size = len(self.token_to_char)
        logger.info(f"Fallback tokenizer loaded from {path}")
        
    def encode(self, text):
        """Encode text to token IDs with proper unknown token handling"""
        unk_token_id = self.special_tokens['<unk>']
        return [self.char_to_token.get(str(char), unk_token_id) for char in text]
    
    def decode(self, tokens):
        """Decode token IDs to text"""
        return "".join([self.token_to_char.get(token, '') for token in tokens])
    
    def get_special_token_id(self, token):
        """Get ID of special token"""
        return self.special_tokens.get(token, self.special_tokens['<unk>'])
    
    def get_vocab_size(self):
        """Get current vocabulary size"""
        return self.vocab_size


class ConversationDataset(Dataset):
    """
    State-of-the-art dataset for conversational AI, featuring:
    1. Correct tokenization formatting
    2. Left-truncation to preserve recent context
    3. Loss masking to train only on assistant's responses
    """
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        input_ids = [self.tokenizer.get_special_token_id('<bos>')]
        labels = [IGNORE_INDEX]  # We never predict the first token

        for turn in conv:
            role = turn['role']
            content = turn['content']
            
            # Format the message with special tokens
            message = f"<{role}>{content}"
            message_ids = self.tokenizer.encode(message)
            
            input_ids.extend(message_ids)
            
            # Mask loss for user prompts, calculate loss for assistant responses
            if role == 'assistant':
                labels.extend(message_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(message_ids))

        input_ids.append(self.tokenizer.get_special_token_id('<eos>'))
        labels.append(self.tokenizer.get_special_token_id('<eos>'))  # The model must learn to predict EOS

        # Left-truncate if sequence is too long to preserve the most recent context
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        # The target for a token at position `i` is the token at position `i+1`
        # So we shift inputs and labels
        model_input = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_target = torch.tensor(labels[1:], dtype=torch.long)

        return {'input_ids': model_input, 'targets': model_target}


class CurriculumTrainer:
    """
    SOTA trainer with Gradient Accumulation, Adaptive Curriculum, Lion optimizer,
    Perplexity validation, and bfloat16 support
    """
    def __init__(self, model, tokenizer, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.train_data = None
        self.val_data = None
        
        # --- Enhanced Trainer Features ---
        self.accumulation_steps = 8  # Simulate a batch size 8x larger
        self.optimizer = self._setup_optimizer(optimizer_name="AdamW")  # "AdamW" or "Lion"
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=500, T_mult=2
        )
        
        # Advanced mixed precision with bfloat16 support
        self.use_amp = (self.device == 'cuda')
        # Use bfloat16 if available (on Ampere GPUs or newer), else float16
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.step = 0
        
        # Adaptive curriculum learning stages with perplexity thresholds
        self.curriculum_stages = [
            {"max_len": 256, "batch_size": 16, "threshold": 50.0},
            {"max_len": 512, "batch_size": 8, "threshold": 25.0},
            {"max_len": 1024, "batch_size": 4, "threshold": 15.0}
        ]

    def _setup_optimizer(self, optimizer_name: str = "AdamW"):
        """Setup optimizer with Lion support"""
        if optimizer_name == "Lion" and LION_AVAILABLE:
            logger.info("Using Lion optimizer")
            return Lion(self.model.parameters(), lr=3e-5, weight_decay=0.1)
        logger.info("Using AdamW optimizer")
        return optim.AdamW(
            self.model.parameters(), 
            lr=3e-4, 
            betas=(0.9, 0.95), 
            weight_decay=0.1
        )

    def _collate_fn(self, batch):
        """Enhanced collation with dynamic padding"""
        # Dynamically pad to the longest sequence in the batch
        max_len = max(len(item['input_ids']) for item in batch)
        input_ids, targets = [], []
        
        pad_token_id = self.tokenizer.get_special_token_id('<pad>')
        
        for item in batch:
            input_pad_len = max_len - len(item['input_ids'])
            target_pad_len = max_len - len(item['targets'])
            
            input_ids.append(torch.cat([
                item['input_ids'], 
                torch.full((input_pad_len,), pad_token_id, dtype=torch.long)
            ]))
            targets.append(torch.cat([
                item['targets'], 
                torch.full((target_pad_len,), IGNORE_INDEX, dtype=torch.long)
            ]))
            
        return {'input_ids': torch.stack(input_ids), 'targets': torch.stack(targets)}

    def _train_epoch(self, dataloader, epoch, stage):
        """Enhanced training epoch with gradient accumulation and mixed precision"""
        self.model.train()
        pbar = tqdm(dataloader, desc=f"Stage {stage}, Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            # Mixed precision forward pass
            with torch.autocast(
                device_type=self.device.split(':')[0], 
                dtype=self.amp_dtype, 
                enabled=self.use_amp
            ):
                _, loss, _ = self.model(
                    batch['input_ids'].to(self.device),
                    batch['targets'].to(self.device)
                )
                loss = loss / self.accumulation_steps  # Scale loss for gradient accumulation
            
            self.scaler.scale(loss).backward()
            
            # Gradient Accumulation Step
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                self.scheduler.step()
                self.step += 1
            
            pbar.set_postfix({
                'loss': loss.item() * self.accumulation_steps, 
                'lr': self.scheduler.get_last_lr()[0]
            })

    def _validate(self, val_data, max_length):
        """Enhanced validation with perplexity calculation"""
        self.model.eval()
        val_dataset = ConversationDataset(val_data, self.tokenizer, max_length)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=8, 
            shuffle=False, 
            collate_fn=self._collate_fn
        )
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(
                    device_type=self.device.split(':')[0], 
                    dtype=self.amp_dtype, 
                    enabled=self.use_amp
                ):
                    _, loss, _ = self.model(
                        batch['input_ids'].to(self.device),
                        batch['targets'].to(self.device)
                    )
                
                # Calculate loss for perplexity
                # Count only the non-ignored tokens
                num_tokens = (batch['targets'] != IGNORE_INDEX).sum().item()
                if num_tokens > 0:
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
        return perplexity

    def train_with_curriculum(self, train_data, val_data):
        """Train with an adaptive curriculum that progresses based on perplexity thresholds"""
        self.train_data = train_data
        self.val_data = val_data
        
        current_stage_idx = 0
        while current_stage_idx < len(self.curriculum_stages):
            stage_config = self.curriculum_stages[current_stage_idx]
            stage_num = current_stage_idx + 1
            
            logger.info(f"--- Starting Curriculum Stage {stage_num} ---")
            logger.info(f"Max length: {stage_config['max_len']}, Batch size: {stage_config['batch_size']}")
            
            dataset = ConversationDataset(train_data, self.tokenizer, stage_config["max_len"])
            dataloader = DataLoader(
                dataset, 
                batch_size=stage_config["batch_size"], 
                shuffle=True, 
                collate_fn=self._collate_fn, 
                num_workers=4, 
                pin_memory=True
            )
            
            # Run one epoch and validate
            self._train_epoch(dataloader, epoch=1, stage=stage_num)
            perplexity = self._validate(val_data, stage_config["max_len"])
            logger.info(f"Stage {stage_num} Validation Perplexity: {perplexity:.2f}")
            
            # Adaptive Curriculum Progression
            if perplexity <= stage_config['threshold']:
                logger.info(f"Threshold of {stage_config['threshold']} met. Progressing to next stage.")
                current_stage_idx += 1
            else:
                logger.info("Threshold not met. Repeating current stage.")

    def save_model(self, path):
        """Enhanced model saving with comprehensive state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'step': self.step,
            'curriculum_stages': self.curriculum_stages,
            'amp_dtype': str(self.amp_dtype),
            'accumulation_steps': self.accumulation_steps
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model with full state restoration"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.step = checkpoint.get('step', 0)
        if 'curriculum_stages' in checkpoint:
            self.curriculum_stages = checkpoint['curriculum_stages']
        if 'accumulation_steps' in checkpoint:
            self.accumulation_steps = checkpoint['accumulation_steps']
            
        logger.info(f"Model loaded from {path} at step {self.step}")
    
    def get_training_stats(self):
        """Get current training statistics"""
        return {
            'step': self.step,
            'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else 0,
            'accumulation_steps': self.accumulation_steps,
            'amp_enabled': self.use_amp,
            'amp_dtype': str(self.amp_dtype),
            'device': self.device
        }

class SmartChatBot:
    """
    State-of-the-art chatbot interface with advanced sampling, conversation history,
    and robust inference optimizations
    """
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Conversation history management
        self.conversation_history = []
        self.max_history_length = 2048  # Maximum tokens to keep in history
        
        # Advanced sampling parameters
        self.default_params = {
            'max_length': 200,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1
        }
        
        # Multiple stop tokens for cleaner generation
        self.stop_tokens = {
            self.tokenizer.get_special_token_id('<eos>') if TOKENIZERS_AVAILABLE and isinstance(self.tokenizer, SmartBPETokenizer) 
            else self.tokenizer.special_tokens['<eos>'],
            self.tokenizer.get_special_token_id('<user>') if TOKENIZERS_AVAILABLE and isinstance(self.tokenizer, SmartBPETokenizer) 
            else self.tokenizer.special_tokens['<user>']
        }

    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """Apply repetition penalty to discourage repeated tokens"""
        if penalty == 1.0:
            return logits
        
        score = torch.gather(logits, 1, input_ids)
        # If score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids, score)
        return logits

    def _apply_top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        if top_k <= 0:
            return logits
        
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits"""
        if top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

    def _build_context(self, user_input: str) -> str:
        """Build conversation context with history management"""
        # Add current user input to history
        current_turn = f"<user>{user_input}<assistant>"
        
        # Build full context from history
        if self.conversation_history:
            context = "".join(self.conversation_history) + current_turn
        else:
            context = current_turn
        
        # Tokenize to check length
        context_tokens = self.tokenizer.encode(context)
        
        # Truncate history if too long (keep recent context)
        while len(context_tokens) > self.max_history_length and self.conversation_history:
            # Remove oldest conversation turn
            self.conversation_history.pop(0)
            context = "".join(self.conversation_history) + current_turn
            context_tokens = self.tokenizer.encode(context)
        
        return context

    def generate_response(
        self, 
        user_input: str, 
        max_length: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None
    ) -> str:
        """
        Generate response with advanced sampling techniques and conversation history
        """
        # Use default parameters if not specified
        params = {
            'max_length': max_length or self.default_params['max_length'],
            'temperature': temperature or self.default_params['temperature'],
            'top_p': top_p or self.default_params['top_p'],
            'top_k': top_k or self.default_params['top_k'],
            'repetition_penalty': repetition_penalty or self.default_params['repetition_penalty']
        }
        
        # Build conversation context
        context = self._build_context(user_input)
        
        # Tokenize the context
        if TOKENIZERS_AVAILABLE and isinstance(self.tokenizer, SmartBPETokenizer):
            tokens = [self.tokenizer.get_special_token_id('<bos>')] + self.tokenizer.encode(context)
        else:
            tokens = [self.tokenizer.special_tokens['<bos>']] + self.tokenizer.encode(context)
            
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Initialize KV caches
        kv_caches = None
        generated_tokens = []
        
        # Generation loop with advanced sampling
        with torch.no_grad():
            for step in range(params['max_length']):
                # Forward pass
                logits, _, kv_caches = self.model(input_ids, kv_caches=kv_caches)
                next_token_logits = logits[:, -1, :] / params['temperature']
                
                # Apply repetition penalty
                if step > 0:  # Only apply after first token
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, params['repetition_penalty']
                    )
                
                # Apply top-k filtering
                next_token_logits = self._apply_top_k_filtering(next_token_logits, params['top_k'])
                
                # Apply top-p (nucleus) filtering
                next_token_logits = self._apply_top_p_filtering(next_token_logits, params['top_p'])
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for stop tokens
                if next_token.item() in self.stop_tokens:
                    break
                
                generated_tokens.append(next_token.item())
                
                # Prepare input for next iteration (only the new token)
                input_ids = next_token
        
        # Decode response
        if generated_tokens:
            response = self.tokenizer.decode(generated_tokens)
            # Clean up response
            response = response.split('<user>')[0].split('<eos>')[0].strip()
        else:
            response = "I'm sorry, I couldn't generate a response."
        
        # Update conversation history
        self.conversation_history.append(f"<user>{user_input}<assistant>{response}")
        
        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history_length(self) -> int:
        """Get current conversation history length in tokens"""
        if not self.conversation_history:
            return 0
        history_text = "".join(self.conversation_history)
        return len(self.tokenizer.encode(history_text))

    def set_default_params(self, **params):
        """Update default generation parameters"""
        for key, value in params.items():
            if key in self.default_params:
                self.default_params[key] = value
                logger.info(f"Updated {key} to {value}")


def format_dolly_dataset(dataset):
    """Formats the Dolly dataset into our conversational structure"""
    formatted_data = []
    for item in dataset:
        instruction, context, response = item['instruction'], item['context'], item['response']
        user_content = f"{context}\n\n{instruction}" if context else instruction
        formatted_data.append([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ])
    return formatted_data

def prepare_tokenizer_training_data(dataset):
    """Prepare text data for tokenizer training"""
    texts = []
    for item in dataset:
        instruction, context, response = item['instruction'], item['context'], item['response']
        user_content = f"{context}\n\n{instruction}" if context else instruction
        text = f"<user>{user_content}<assistant>{response}"
        texts.append(text)
    return texts

def main():
    """
    Main training and chat loop with state-of-the-art optimizations and robust error handling
    """
    print("🚀 Initializing Enhanced Smart LLM with SOTA Features...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Advanced model configuration
    config = ModelConfig(
        vocab_size=16000 if TOKENIZERS_AVAILABLE else 32000,
        d_model=512, 
        n_heads=8, 
        n_kv_heads=2, 
        n_layers=8,
        d_ff=1024, 
        max_seq_len=1024, 
        dropout=0.1,
        use_rotary_embeddings=True, 
        use_mixture_of_experts=True,
        num_experts=4, 
        num_experts_per_token=2
    )
    
    # Initialize tokenizer with enhanced error handling
    if TOKENIZERS_AVAILABLE:
        tokenizer = SmartBPETokenizer(config.vocab_size)
        tokenizer_path = "smart-bpe-tokenizer.json"
        print("✅ Using advanced BPE tokenizer")
    else:
        print("⚠️  Tokenizers library not found. Using fallback character tokenizer.")
        print("💡 Install with: pip install tokenizers")
        tokenizer = FallbackCharTokenizer(config.vocab_size)
        tokenizer_path = "fallback-tokenizer.json"
    
    # Initialize model and trainer
    model = SmartLLM(config)
    trainer = CurriculumTrainer(model, tokenizer, config)
    
    # Display comprehensive model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size (MB): {total_params * 4 / 1024 / 1024:.1f}")
    print(f"   Architecture: {config.n_layers} layers, {config.n_heads} heads, {config.d_model} dim")
    print(f"   MoE: {config.num_experts} experts, {config.num_experts_per_token} active")
    print(f"🔧 Device: {trainer.device}")
    print(f"⚡ Mixed precision: {trainer.use_amp} ({trainer.amp_dtype})")
    
    model_path = "enhanced_smart_llm_model.pth"
    
    # Enhanced model loading with comprehensive state restoration
    if os.path.exists(model_path):
        print("\n📂 Loading existing model...")
        try:
            trainer.load_model(model_path)
            print("✅ Model loaded successfully!")
            
            # Load tokenizer if it exists
            if os.path.exists(tokenizer_path):
                tokenizer.load(tokenizer_path)
                print("✅ Tokenizer loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print("🔄 Starting fresh training...")
    else:
        print("\n🎯 Training new model...")
        
        if not DATASETS_AVAILABLE:
            print("❌ Datasets library not found. Install with: pip install datasets")
            print("🤖 Starting chat with untrained model...")
        else:
            try:
                logger.info("📥 Loading databricks/dolly-15k dataset...")
                # IMPORTANT: Make sure you replace '~' with your actual username folder
                local_path = r"C:\Users\User\~\math-ai\databricks-dolly-15k.jsonl"
                dolly_dataset = load_dataset("json", data_files=local_path, split="train")
                
                # Train tokenizer if using BPE and doesn't exist
                if TOKENIZERS_AVAILABLE and not os.path.exists(tokenizer_path):
                    logger.info("🔤 Training BPE tokenizer...")
                    training_texts = prepare_tokenizer_training_data(dolly_dataset)
                    tokenizer.train_from_texts(training_texts, tokenizer_path)
                elif TOKENIZERS_AVAILABLE and os.path.exists(tokenizer_path):
                    tokenizer.load(tokenizer_path)
                elif not TOKENIZERS_AVAILABLE and os.path.exists(tokenizer_path):
                    tokenizer.load(tokenizer_path)
                
                logger.info("📊 Formatting dataset...")
                training_data = format_dolly_dataset(dolly_dataset)
                random.shuffle(training_data)
                
                # Split data with better proportions
                split_idx = int(len(training_data) * 0.95)
                train_split, val_split = training_data[:split_idx], training_data[split_idx:]
                
                print(f"📈 Training samples: {len(train_split)}")
                print(f"🔍 Validation samples: {len(val_split)}")
                
                logger.info("🚀 Starting adaptive curriculum training...")
                trainer.train_with_curriculum(train_split, val_split)
                
                trainer.save_model(model_path)
                print("💾 Training completed and model saved!")
                
            except Exception as e:
                logger.error(f"Error during training: {e}")
                print("🤖 Starting chat with untrained model...")
    
    # Initialize enhanced chatbot
    print("\n🤖 Enhanced Smart LLM is ready! Starting interactive chat...")
    print("✨ Features: Advanced Sampling, Conversation History, MoE, Curriculum Learning")
    print("📝 Commands: 'quit', 'help', 'stats', 'clear', 'params'")
    
    chatbot = SmartChatBot(model, tokenizer, trainer.device)
    
    # Interactive chat loop with enhanced commands
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                print("\n🔧 Available Commands:")
                print("  'quit' - Exit the chat")
                print("  'help' - Show this help message")
                print("  'stats' - Show detailed model statistics")
                print("  'clear' - Clear conversation history")
                print("  'params' - Show/modify generation parameters")
                print("  'history' - Show conversation history length")
                continue
                
            elif user_input.lower() == 'stats':
                stats = trainer.get_training_stats()
                print(f"\n📊 Detailed Model Statistics:")
                print(f"   Parameters: {total_params:,}")
                print(f"   Layers: {config.n_layers}")
                print(f"   Attention heads: {config.n_heads} (KV: {config.n_kv_heads})")
                print(f"   Model dimension: {config.d_model}")
                print(f"   Feed-forward dimension: {config.d_ff}")
                print(f"   Vocabulary size: {config.vocab_size}")
                print(f"   Max sequence length: {config.max_seq_len}")
                print(f"   Training step: {stats['step']}")
                print(f"   Learning rate: {stats['learning_rate']:.2e}")
                print(f"   Device: {stats['device']}")
                print(f"   Mixed precision: {stats['amp_enabled']}")
                continue
                
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🗑️ Conversation history cleared!")
                continue
                
            elif user_input.lower() == 'history':
                length = chatbot.get_history_length()
                print(f"📚 Conversation history: {length} tokens")
                continue
                
            elif user_input.lower() == 'params':
                print("\n⚙️ Current Generation Parameters:")
                for key, value in chatbot.default_params.items():
                    print(f"   {key}: {value}")
                print("\n💡 To modify, use: params <param>=<value> (e.g., 'params temperature=0.8')")
                continue
                
            elif user_input.lower().startswith('params '):
                try:
                    param_str = user_input[7:]  # Remove 'params '
                    key, value = param_str.split('=')
                    key, value = key.strip(), value.strip()
                    
                    # Convert value to appropriate type
                    if key in ['max_length', 'top_k']:
                        value = int(value)
                    else:
                        value = float(value)
                    
                    chatbot.set_default_params(**{key: value})
                    print(f"✅ Updated {key} to {value}")
                except Exception as e:
                    print(f"❌ Error updating parameter: {e}")
                continue
            
            if user_input:
                print("🤔 Generating response...")
                response = chatbot.generate_response(user_input, max_length=150)
                print(f"🤖 Assistant: {response}")
            else:
                print("❓ Please enter a message.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting gracefully...")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"❌ Error: {e}")
            print("💡 Try 'clear' to reset conversation or 'help' for commands")
            continue

if __name__ == "__main__":
    # Set optimal environment variables for performance
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    main()
