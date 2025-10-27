"""
Character-level tokenizer for URL processing.
"""
import string
from typing import List, Dict, Tuple
import torch
import numpy as np

class CharTokenizer:
    """Character-level tokenizer for URLs."""
    
    def __init__(self, vocab_size: int = 150):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character vocabulary."""
        # Common URL characters
        chars = []
        
        # Letters
        chars.extend(string.ascii_lowercase)
        chars.extend(string.ascii_uppercase)
        
        # Digits
        chars.extend(string.digits)
        
        # URL symbols
        url_symbols = ['/', ':', '.', '-', '_', '?', '=', '&', '%', '+', '#', '@', '!', '~', '*', "'", '(', ')', ';', ',']
        chars.extend(url_symbols)
        
        # Special tokens
        chars.extend([self.pad_token, self.unk_token])
        
        # Limit vocabulary size
        chars = chars[:self.vocab_size - 2]  # Reserve space for special tokens
        
        # Build mappings
        self.char_to_id = {char: idx for idx, char in enumerate(chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Add special tokens
        self.char_to_id[self.pad_token] = len(self.char_to_id)
        self.char_to_id[self.unk_token] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = self.pad_token
        self.id_to_char[len(self.id_to_char)] = self.unk_token
        
        self.pad_id = self.char_to_id[self.pad_token]
        self.unk_id = self.char_to_id[self.unk_token]
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        for char in text[:max_length]:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.pad_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.id_to_char.get(token_id, self.unk_token) for token_id in token_ids)
    
    def batch_encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        """Encode batch of texts."""
        encoded = [self.encode(text, max_length) for text in texts]
        return torch.tensor(encoded, dtype=torch.long)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char_to_id)
    
    def save(self, path: str):
        """Save tokenizer."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        tokenizer.pad_id = tokenizer.char_to_id[tokenizer.pad_token]
        tokenizer.unk_id = tokenizer.char_to_id[tokenizer.unk_token]
        
        return tokenizer
