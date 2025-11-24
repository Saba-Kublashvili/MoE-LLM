import os
import json
import logging
from typing import List, Dict, Optional

# Conditional import for robust execution without tokenizers lib
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SmartBPETokenizer:
    """Byte-Level BPE Tokenizer wrapper."""

    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<system>"]
        
    def train_from_texts(self, texts, save_path="smart-bpe-tokenizer.json"):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required for BPE tokenizer")
        
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(),
            pre_tokenizers.Digits(individual_digits=True)
        ])
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2
        )
        
        self.tokenizer.train_from_iterator(texts, trainer)
        self.tokenizer.save(save_path)
        logger.info(f"BPE tokenizer trained with {self.tokenizer.get_vocab_size()} tokens.")
        
    def load(self, path="smart-bpe-tokenizer.json"):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required.")
            
        if os.path.exists(path):
            self.tokenizer = Tokenizer.from_file(path)
            logger.info(f"Loaded BPE tokenizer from {path}")
        else:
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
    
    def encode(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.decode(tokens)
    
    def get_special_token_id(self, token):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.tokenizer.token_to_id(token)
    
    def expand_vocabulary(self, new_tokens: list, model):
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first.")
        
        num_added = self.tokenizer.add_tokens(new_tokens)
        if num_added > 0:
            model.resize_token_embeddings(self.tokenizer.get_vocab_size())
            logger.info(f"Vocabulary expanded by {num_added} tokens.")
        return num_added
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

class FallbackCharTokenizer:
    """Robust character-level tokenizer for environments without 'tokenizers' lib."""
    
    def __init__(self, vocab_size=32000):
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<user>': 4, '<assistant>': 5, '<system>': 6
        }
        self.token_to_char = {}
        self.char_to_token = {}
        self.build_vocab()
        # Keep consistent size attribute
        self.vocab_size = len(self.token_to_char)
    
    def build_vocab(self):
        self.token_to_char = {v: k for k, v in self.special_tokens.items()}
        self.char_to_token = {k: v for k, v in self.special_tokens.items()}
        
        current_idx = len(self.special_tokens)
        for i in range(256):
            char = chr(i)
            if char not in self.char_to_token:
                self.token_to_char[current_idx] = char
                self.char_to_token[char] = current_idx
                current_idx += 1
    
    def save(self, path: str):
        vocab_data = {'char_to_token': self.char_to_token, 'special_tokens': self.special_tokens}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.char_to_token = vocab_data['char_to_token']
        self.special_tokens = vocab_data['special_tokens']
        self.token_to_char = {int(v): k for k, v in self.char_to_token.items()}
        self.vocab_size = len(self.token_to_char)
        
    def encode(self, text):
        unk_token_id = self.special_tokens['<unk>']
        return [self.char_to_token.get(str(char), unk_token_id) for char in text]
    
    def decode(self, tokens):
        return "".join([self.token_to_char.get(token, '') for token in tokens])
    
    def get_special_token_id(self, token):
        return self.special_tokens.get(token, self.special_tokens['<unk>'])
    
    def get_vocab_size(self):
        return len(self.token_to_char)
