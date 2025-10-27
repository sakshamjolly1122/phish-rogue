"""
Stage-B: Fusion model combining URL embeddings with HTML features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FusionHead(nn.Module):
    """Fusion model combining URL embeddings with HTML content features."""
    
    def __init__(
        self,
        url_embedding_dim: int = 192,
        content_dim: int = 20,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.url_embedding_dim = url_embedding_dim
        self.content_dim = content_dim
        self.num_classes = num_classes
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(url_embedding_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
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
    
    def forward(self, url_embeddings: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            url_embeddings: URL embeddings from Stage-A of shape (batch_size, url_embedding_dim)
            content_features: HTML content features of shape (batch_size, content_dim)
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
        """
        # Concatenate URL embeddings and content features
        combined = torch.cat([url_embeddings, content_features], dim=-1)
        
        # Apply fusion layers
        logits = self.fusion(combined)
        
        return logits
    
    def get_embeddings(self, url_embeddings: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor:
        """Get fused embeddings before final classification layer."""
        combined = torch.cat([url_embeddings, content_features], dim=-1)
        
        # Get hidden representation
        hidden = F.relu(self.fusion[0](combined))
        hidden = self.fusion[2](hidden)  # Apply dropout
        
        return hidden
