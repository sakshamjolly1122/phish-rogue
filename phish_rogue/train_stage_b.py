"""
Training script for Stage-B fusion model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    set_seed, load_yaml, get_device, setup_logger, 
    AverageMeter, count_parameters, ensure_dir, load_json
)
from .model_stage_b import FusionHead
from .model_stage_a import URLTiny
from .tokenizer import CharTokenizer
from .features_content import HTMLFeatureExtractor

class StageBDataset(Dataset):
    """Dataset for Stage-B training."""
    
    def __init__(
        self,
        urls: list,
        labels: list,
        url_embeddings: np.ndarray,
        content_features: np.ndarray,
        label_to_id: dict
    ):
        self.urls = urls
        self.labels = labels
        self.url_embeddings = url_embeddings
        self.content_features = content_features
        self.label_to_id = label_to_id
        
        # Convert labels to IDs
        self.label_ids = [label_to_id[label] for label in labels]
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        return {
            'url_embedding': torch.tensor(self.url_embeddings[idx], dtype=torch.float32),
            'content_features': torch.tensor(self.content_features[idx], dtype=torch.float32),
            'label': torch.tensor(self.label_ids[idx], dtype=torch.long),
            'url': self.urls[idx]
        }

def train_epoch(model, dataloader, optimizer, criterion, device, logger):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc="Training Stage-B")
    for batch in pbar:
        url_embeddings = batch['url_embedding'].to(device)
        content_features = batch['content_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(url_embeddings, content_features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), url_embeddings.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg

def validate_epoch(model, dataloader, device, class_names):
    """Validate for one epoch."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Stage-B"):
            url_embeddings = batch['url_embedding'].to(device)
            content_features = batch['content_features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(url_embeddings, content_features)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute macro F1
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return f1_macro

def main():
    parser = argparse.ArgumentParser(description='Train Stage-B fusion model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--routed_csv', type=str, required=True,
                       help='Path to routed CSV with escalated samples')
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    set_seed(config['seed'])
    
    # Setup
    device = get_device()
    logger = setup_logger('train_stage_b')
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    stage_b_dir = config['paths']['stage_b_dir']
    ensure_dir(stage_b_dir)
    
    # Load routed data
    logger.info(f"Loading routed data from {args.routed_csv}")
    routed_df = pd.read_csv(args.routed_csv)
    
    # Filter escalated samples
    escalated_df = routed_df[routed_df['escalate'] == True].copy()
    logger.info(f"Training on {len(escalated_df)} escalated samples")
    
    if len(escalated_df) == 0:
        logger.warning("No escalated samples found! Cannot train Stage-B.")
        return
    
    # Load Stage-A model and tokenizer
    stage_a_dir = config['paths']['stage_a_dir']
    checkpoint_path = os.path.join(stage_a_dir, 'best.pt')
    tokenizer_path = os.path.join(stage_a_dir, 'tokenizer.json')
    label_mappings_path = os.path.join(stage_a_dir, 'label_mappings.json')
    
    # Load Stage-A checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    
    # Initialize Stage-A model
    stage_a_model = URLTiny(
        vocab_size=model_config['stage_a']['model']['vocab_size'],
        d_model=model_config['stage_a']['model']['d_model'],
        nheads=model_config['stage_a']['model']['nheads'],
        nlayers=model_config['stage_a']['model']['nlayers'],
        num_classes=len(model_config['class_names']),
        max_length=model_config['data']['max_url_len']
    ).to(device)
    
    stage_a_model.load_state_dict(checkpoint['model_state_dict'])
    stage_a_model.eval()
    
    # Load tokenizer and label mappings
    tokenizer = CharTokenizer.load(tokenizer_path)
    label_mappings = load_json(label_mappings_path)
    
    logger.info("Stage-A model loaded for embedding extraction")
    
    # Extract URL embeddings
    logger.info("Extracting URL embeddings...")
    urls = escalated_df['url'].tolist()
    labels = escalated_df['true_label'].tolist()
    
    # Tokenize URLs
    input_ids = tokenizer.batch_encode(urls, max_length=config['data']['max_url_len'])
    input_ids = input_ids.to(device)
    
    # Extract embeddings
    with torch.no_grad():
        url_embeddings = stage_a_model.get_embeddings(input_ids)
        url_embeddings = url_embeddings.cpu().numpy()
    
    # Extract HTML content features
    logger.info("Extracting HTML content features...")
    extractor = HTMLFeatureExtractor(
        timeout_ms=config['stage_b']['html_timeout_ms'],
        content_dim=config['stage_b']['content_dim']
    )
    content_features = extractor.batch_extract_features(urls)
    
    logger.info(f"URL embeddings shape: {url_embeddings.shape}")
    logger.info(f"Content features shape: {content_features.shape}")
    
    # Create dataset and dataloader
    dataset = StageBDataset(
        urls=urls,
        labels=labels,
        url_embeddings=url_embeddings,
        content_features=content_features,
        label_to_id=label_mappings['label_to_id']
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['stage_b']['train']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['stage_b']['train']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize Stage-B model
    model = FusionHead(
        url_embedding_dim=config['stage_a']['model']['d_model'],
        content_dim=config['stage_b']['content_dim'],
        hidden_dim=256,
        num_classes=len(config['class_names']),
        dropout=0.1
    ).to(device)
    
    logger.info(f"Stage-B model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['stage_b']['train']['lr'],
        weight_decay=config['stage_b']['train']['weight_decay']
    )
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    train_losses = []
    val_f1s = []
    
    for epoch in range(config['stage_b']['train']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['stage_b']['train']['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        train_losses.append(train_loss)
        
        # Validate
        val_f1 = validate_epoch(model, val_loader, device, config['class_names'])
        val_f1s.append(val_f1)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config,
                'feature_names': extractor.get_feature_names()
            }
            torch.save(checkpoint, os.path.join(stage_b_dir, 'best.pt'))
            
            logger.info(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 3:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save feature names
    from .utils import save_json
    save_json(
        {'feature_names': extractor.get_feature_names()},
        os.path.join(stage_b_dir, 'content_feat_names.json')
    )
    
    # Final evaluation
    logger.info("Final evaluation...")
    from .metrics import evaluate_model
    
    # Create a wrapper for Stage-B model to work with evaluate_model
    class StageBWrapper:
        def __init__(self, model):
            self.model = model
        
        def __call__(self, batch):
            url_embeddings = batch['url_embedding']
            content_features = batch['content_features']
            return self.model(url_embeddings, content_features)
    
    wrapper = StageBWrapper(model)
    metrics = evaluate_model(wrapper, val_loader, device, config['class_names'])
    
    # Save metrics
    metrics_to_save = {
        'best_f1': best_f1,
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_f1s': val_f1s
    }
    save_json(metrics_to_save, os.path.join(stage_b_dir, 'metrics_val.json'))
    
    logger.info(f"Stage-B training completed. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
