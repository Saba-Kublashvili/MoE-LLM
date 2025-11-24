from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Hyperparameters for the SmartLLM.
    Default values represent a medium-sized architecture suitable for single-GPU study.
    """
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1

    # Advanced architecture toggles
    use_rotary_embeddings: bool = True
    use_mixture_of_experts: bool = True
    num_experts: int = 4
    num_experts_per_token: int = 2
    load_balancing_weight: float = 0.01

    # Attention and Loss settings
    sliding_window_size: int = 128
    router_z_loss_weight: float = 0.001
    use_flash_attention: bool = True
    use_sliding_window_attention: bool = True

    # Regularization and Training stability
    use_rezero: bool = True
    layer_drop_prob: float = 0.05
    use_gradient_checkpointing: bool = True
