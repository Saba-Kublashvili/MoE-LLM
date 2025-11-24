# Custom_LLM: Advanced Modular Transformer Architecture

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)

**Custom_LLM** is a comprehensive, ground-up implementation of a modern Large Language Model (LLM) architecture.

Unlike standard tutorials that implement the original 2017 Transformer, CustomLLM aggregates **state-of-the-art (SOTA) architectural improvements** used in modern foundation models (like Llama 3, Mixtral, and PaLM) into a single, cohesive codebase.

The primary goal of this project is **architectural transparency and modularity**. It serves as a practical "living library" where developers can study, extract, and implement advanced techniques like Mixture of Experts, Rotary Embeddings, and SwiGLU activations using pure PyTorch.

## Key Features & Technologies

This repository demonstrates how to implement the specific technologies that drive modern LLM performance:

### Advanced Neural Architecture

- **Mixture of Experts (MoE):** Sparse MoE implementation featuring top-k noisy gating, shared experts, and auxiliary load-balancing losses (Z-loss) for high parameter efficiency.
- **Rotary Positional Embeddings (RoPE):** Relative positional encoding for superior sequence length generalization.
- **Grouped Query Attention (GQA):** Inference-optimized attention that reduces KV-cache memory footprint while maintaining performance.
- **Sliding Window Attention (SWA):** Vectorized masking implementation for efficient processing of long sequences with local context windows.
- **SwiGLU Activation:** Replaces standard ReLU/GELU with the Swish-Gated Linear Unit for improved convergence.
- **RMSNorm:** Root Mean Square Normalization for training stability.

### Optimization & Training Stability

- **ReZero (Residual Zero):** Learnable residual scaling parameters to initialize deep networks effectively.
- **LayerDrop:** Structured dropout that skips entire layers during training, acting as a powerful regularizer.
- **Gradient Checkpointing:** Memory-efficient training allowing for deeper models on consumer hardware.
- **Flash Attention Ready:** Logic structured for easy integration with optimized attention kernels.

### Robust Training Loop

- **Adaptive Curriculum Learning:** Dynamic training stages that increase sequence length and batch size based on validation perplexity.
- **Mixed Precision (AMP):** Native `bfloat16` (Ampere+ GPUs) and `float16` support via `torch.cuda.amp`.
- **Lion & AdamW Optimizers:** Support for the SOTA Lion optimizer.
- **Gradient Accumulation:** Simulates massive batch sizes on limited hardware.

### Tokenization

- **Byte-Level BPE:** Robust subword tokenization with full Unicode support (via HuggingFace `tokenizers`).
- **Zero-Dependency Fallback:** Includes a character-level tokenizer ensuring the model runs purely on Python standard libraries if needed.

## Project Structure

The code is modularized to facilitate component extraction:

```bash
.
├── config.py       # Hyperparameters & Dataclasses
├── model.py        # Core Architecture (RoPE, MoE, TransformerBlock, SmartLLM)
├── tokenization.py # SmartBPETokenizer & FallbackCharTokenizer
├── trainer.py      # Curriculum Trainer, Mixed Precision, & Dataset Logic
├── inference.py    # Generation Logic (Top-k, Top-p, Repetition Penalty)
└── main.py         # Entry point & CLI
