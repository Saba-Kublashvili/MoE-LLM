import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
import math
from config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class RotaryPositionalEmbedding(nn.Module):
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
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, Seq, 1, Dim]
    sin = sin.unsqueeze(0).unsqueeze(2) # [1, Seq, 1, Dim]
    q_out = (q * cos) + (torch.cat([-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]], dim=-1) * sin)
    k_out = (k * cos) + (torch.cat([-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]], dim=-1) * sin)
    return q_out, k_out

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.w3(self.dropout(gate * up))

class GroupedQueryAttention(nn.Module):
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
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        
        current_seq_len = k.shape[2]
        is_causal_needed = T > 1
        attn_mask = None
        if self.use_sliding_window and current_seq_len > self.sliding_window_size:
            is_causal_needed = False
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

        flat_selected_experts = selected_experts.view(-1)
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.num_experts_per_token)

        final_routed_output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.routed_experts):
            expert_mask = (flat_selected_experts == i)
            if expert_mask.any():
                tokens_for_expert = x_flat[flat_token_indices[expert_mask]]
                weights_for_expert = routing_weights.view(-1)[expert_mask].unsqueeze(1)
                expert_out = expert(tokens_for_expert)
                final_routed_output.index_add_(0, flat_token_indices[expert_mask], expert_out * weights_for_expert)
        
        final_output = shared_output + final_routed_output
        
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
             aux_loss = self._compute_auxiliary_losses(router_logits)
        
        return final_output.view(B, T, C), aux_loss

    def _compute_auxiliary_losses(self, router_logits: torch.Tensor):
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        expert_load = router_probs.sum(dim=0)
        expert_fraction = expert_load / expert_load.sum()
        load_balancing_loss = self.num_experts * torch.sum(expert_fraction * expert_fraction)
        
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return (self.load_balancing_weight * load_balancing_loss) + (self.router_z_loss_weight * z_loss)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_rezero = config.use_rezero
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        
        self.attention = GroupedQueryAttention(config)
        
        if config.use_mixture_of_experts:
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = SwiGLU(config.d_model, config.d_ff, config.dropout)
        
        if self.use_rezero:
            self.rezero_alpha_attn = nn.Parameter(torch.zeros(1))
            self.rezero_alpha_mlp = nn.Parameter(torch.zeros(1))
        
        self.layer_drop = nn.Dropout(config.layer_drop_prob)
    
    def _forward_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor], 
                          kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        return self.attention(self.ln1(x), mask, kv_cache)
    
    def _forward_mlp(self, x: torch.Tensor):
        if isinstance(self.mlp, MixtureOfExperts):
            return self.mlp(self.ln2(x))
        else:
            return self.mlp(self.ln2(x)), torch.tensor(0.0, device=x.device)
    
    def forward(self, x, mask=None, kv_cache=None):
        aux_loss = torch.tensor(0.0, device=x.device)
        residual = x
        
        use_checkpoint = self.training and self.use_gradient_checkpointing
        
        if use_checkpoint:
            attn_out, new_kv_cache = checkpoint(
                self._forward_attention, x, mask, kv_cache, 
                use_reentrant=False
            )
        else:
            attn_out, new_kv_cache = self._forward_attention(x, mask, kv_cache)
        
        if self.training:
            attn_out = self.layer_drop(attn_out)
        
        if self.use_rezero:
            x = residual + self.rezero_alpha_attn * attn_out
        else:
            x = residual + attn_out
        
        residual = x
        
        if use_checkpoint:
            mlp_out, mlp_aux_loss = checkpoint(self._forward_mlp, x, use_reentrant=False)
        else:
            if isinstance(self.mlp, MixtureOfExperts):
                mlp_out, mlp_aux_loss = self.mlp(self.ln2(x))
            else:
                mlp_out = self.mlp(self.ln2(x))
                mlp_aux_loss = torch.tensor(0.0, device=x.device)
        
        aux_loss = aux_loss + mlp_aux_loss
        
        if self.training:
            mlp_out = self.layer_drop(mlp_out)
        
        if self.use_rezero:
            x = residual + self.rezero_alpha_mlp * mlp_out
        else:
            x = residual + mlp_out
        
        return x, new_kv_cache, aux_loss

class SmartLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2.0 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, kv_caches=None):
        B, T = tokens.shape
        x = self.token_embedding(tokens)
        
        total_aux_loss = torch.tensor(0.0, device=tokens.device)
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers
            
        new_kv_caches = []
        
        for i, layer in enumerate(self.layers):
            x, new_kv, aux_loss = layer(x, kv_cache=kv_caches[i])
            total_aux_loss += aux_loss
            new_kv_caches.append(new_kv)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            if total_aux_loss.item() > 0:
                loss += total_aux_loss
        
        return logits, loss, new_kv_caches

    def resize_token_embeddings(self, new_vocab_size):
        old_vocab_size = self.config.vocab_size
        self.config.vocab_size = new_vocab_size
        
        new_token_embedding = nn.Embedding(new_vocab_size, self.config.d_model)
        new_lm_head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
        
        with torch.no_grad():
            new_token_embedding.weight[:old_vocab_size] = self.token_embedding.weight
            new_lm_head.weight[:old_vocab_size] = self.lm_head.weight
            
            if new_vocab_size > old_vocab_size:
                torch.nn.init.normal_(new_token_embedding.weight[old_vocab_size:], mean=0.0, std=0.02)
                torch.nn.init.normal_(new_lm_head.weight[old_vocab_size:], mean=0.0, std=0.02)
        
        self.token_embedding = new_token_embedding
        self.lm_head = new_lm_head
        self.lm_head.weight = self.token_embedding.weight
