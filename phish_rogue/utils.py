"""
General utilities for PHISH-ROGUE project.
"""
import os
import json
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save data as YAML file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: str) -> None:
    """Save data as JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup rich logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = RichHandler(console=console, show_time=True, show_path=False)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def normalize_labels(labels: list) -> tuple:
    """Convert string labels to integer mapping."""
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def validate_url(url: str) -> bool:
    """Basic URL validation."""
    return url.startswith(('http://', 'https://')) and len(url) > 10
