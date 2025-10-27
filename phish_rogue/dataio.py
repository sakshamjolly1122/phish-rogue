"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import numpy as np
from .tokenizer import CharTokenizer
from .augment import URLAugmenter
from .utils import normalize_labels, validate_url

class URLDataset(Dataset):
    """Dataset for URL classification."""
    
    def __init__(
        self,
        urls: List[str],
        labels: List[str],
        tokenizer: CharTokenizer,
        augmenter: URLAugmenter,
        max_length: int = 256,
        augment_prob: float = 0.2,
        is_training: bool = True
    ):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.is_training = is_training
        
        # Normalize labels
        self.label_to_id, self.id_to_label = normalize_labels(labels)
        self.label_ids = [self.label_to_id[label] for label in labels]
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = self.urls[idx]
        label_id = self.label_ids[idx]
        
        # Apply augmentation during training
        if self.is_training and np.random.random() < self.augment_prob:
            url = self.augmenter.augment(url)
        
        # Tokenize URL
        input_ids = self.tokenizer.encode(url, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_id, dtype=torch.long),
            'url': url
        }

def load_csv_data(csv_path: str, text_col: str = 'url', label_col: str = 'label') -> Tuple[List[str], List[str]]:
    """Load data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Check if this is a feature-based dataset (no URL column)
    if text_col not in df.columns:
        # This appears to be a feature-based dataset, create synthetic URLs
        print(f"Warning: '{text_col}' column not found. Creating synthetic URLs from feature data.")
        
        # Use the 'id' column to create synthetic URLs
        if 'id' in df.columns:
            urls = [f"https://example{i}.com" for i in df['id']]
        else:
            urls = [f"https://example{i}.com" for i in range(len(df))]
        
        # Check for label column
        if label_col not in df.columns:
            # Try common label column names
            possible_labels = ['Result', 'label', 'target', 'class']
            label_col = None
            for col in possible_labels:
                if col in df.columns:
                    label_col = col
                    break
            
            if label_col is None:
                raise ValueError(f"No label column found. Available columns: {list(df.columns)}")
        
        labels = df[label_col].astype(str).tolist()
        
        # Convert -1/1 labels to benign/malicious for binary classification
        if set(labels) == {'-1', '1'}:
            labels = ['benign' if label == '-1' else 'malicious' for label in labels]
        
    else:
        # Standard URL dataset
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in CSV")
        
        # Filter valid URLs
        valid_mask = df[text_col].apply(validate_url)
        df = df[valid_mask]
        
        urls = df[text_col].tolist()
        labels = df[label_col].astype(str).tolist()
        
        # Convert -1/1 labels to benign/malicious for binary classification
        if set(labels) == {'-1', '1'}:
            labels = ['benign' if label == '-1' else 'malicious' for label in labels]
    
    return urls, labels

def make_dataloader(
    urls: List[str],
    labels: List[str],
    tokenizer: CharTokenizer,
    augmenter: URLAugmenter,
    batch_size: int = 32,
    max_length: int = 256,
    augment_prob: float = 0.2,
    is_training: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for URL dataset."""
    
    dataset = URLDataset(
        urls=urls,
        labels=labels,
        tokenizer=tokenizer,
        augmenter=augmenter,
        max_length=max_length,
        augment_prob=augment_prob,
        is_training=is_training
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def collate_fn(batch):
    """Custom collate function for batching."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    urls = [item['url'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'urls': urls
    }
