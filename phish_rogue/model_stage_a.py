"""
Stage-A: CNN + Transformer URL classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class URLTiny(nn.Module):
    """Lightweight CNN + Transformer model for URL classification."""
    
    def __init__(
        self,
        vocab_size: int = 150,
        d_model: int = 192,
        nheads: int = 3,
        nlayers: int = 2,
        num_classes: int = 5,
        max_length: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # CNN layers for local pattern extraction
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # Pooling and classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            pooled_embedding: Pooled embeddings of shape (batch_size, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # CNN layers for local patterns
        x_conv = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = self.norm1(x_conv.transpose(1, 2))  # (batch_size, seq_len, d_model)
        
        x_conv = x_conv.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = self.norm2(x_conv.transpose(1, 2))  # (batch_size, seq_len, d_model)
        
        # Residual connection
        x = x + x_conv
        
        # Transformer encoder
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Global pooling
        x_pooled = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x_pooled = self.pool(x_pooled)  # (batch_size, d_model, 1)
        x_pooled = x_pooled.squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x_pooled)  # (batch_size, num_classes)
        
        return logits, x_pooled
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get pooled embeddings for Stage-B."""
        _, pooled_embedding = self.forward(input_ids)
        return pooled_embedding
