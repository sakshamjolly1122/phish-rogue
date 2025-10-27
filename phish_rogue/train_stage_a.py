"""
Training script for Stage-A model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .utils import (
    set_seed, load_yaml, get_device, setup_logger, 
    AverageMeter, count_parameters, ensure_dir
)
from .tokenizer import CharTokenizer
from .augment import URLAugmenter
from .dataio import load_csv_data, make_dataloader
from .model_stage_a import URLTiny
from .metrics import evaluate_model

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model, dataloader, optimizer, criterion, device, logger):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(input_ids)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), input_ids.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg

def validate_epoch(model, dataloader, device, class_names):
    """Validate for one epoch."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute macro F1
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    return f1_macro

def main():
    parser = argparse.ArgumentParser(description='Train Stage-A model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    set_seed(config['seed'])
    
    # Setup
    device = get_device()
    logger = setup_logger('train_stage_a')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {config}")
    
    # Create output directory
    stage_a_dir = config['paths']['stage_a_dir']
    ensure_dir(stage_a_dir)
    
    # Load data
    logger.info("Loading data...")
    train_urls, train_labels = load_csv_data(config['data']['train_csv'])
    val_urls, val_labels = load_csv_data(config['data']['val_csv'])
    
    logger.info(f"Train samples: {len(train_urls)}")
    logger.info(f"Val samples: {len(val_urls)}")
    
    # Initialize tokenizer and augmenter
    tokenizer = CharTokenizer(vocab_size=config['stage_a']['model']['vocab_size'])
    augmenter = URLAugmenter(augment_prob=config['data']['augment_prob'])
    
    # Create dataloaders
    train_loader = make_dataloader(
        train_urls, train_labels, tokenizer, augmenter,
        batch_size=config['stage_a']['train']['batch_size'],
        max_length=config['data']['max_url_len'],
        is_training=True,
        shuffle=True
    )
    
    val_loader = make_dataloader(
        val_urls, val_labels, tokenizer, augmenter,
        batch_size=config['stage_a']['train']['batch_size'],
        max_length=config['data']['max_url_len'],
        is_training=False,
        shuffle=False
    )
    
    # Initialize model
    model = URLTiny(
        vocab_size=config['stage_a']['model']['vocab_size'],
        d_model=config['stage_a']['model']['d_model'],
        nheads=config['stage_a']['model']['nheads'],
        nlayers=config['stage_a']['model']['nlayers'],
        num_classes=len(config['class_names']),
        max_length=config['data']['max_url_len']
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = FocalLoss(gamma=config['stage_a']['train']['focal_gamma'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['stage_a']['train']['lr'],
        weight_decay=config['stage_a']['train']['weight_decay']
    )
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    train_losses = []
    val_f1s = []
    
    for epoch in range(config['stage_a']['train']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['stage_a']['train']['epochs']}")
        
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
                'config': config
            }
            torch.save(checkpoint, os.path.join(stage_a_dir, 'best.pt'))
            
            logger.info(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['stage_a']['train']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save tokenizer and label mappings
    tokenizer.save(os.path.join(stage_a_dir, 'tokenizer.json'))
    
    # Get label mappings from first dataloader
    train_dataset = train_loader.dataset
    label_mappings = {
        'label_to_id': train_dataset.label_to_id,
        'id_to_label': train_dataset.id_to_label
    }
    
    from .utils import save_json
    save_json(label_mappings, os.path.join(stage_a_dir, 'label_mappings.json'))
    
    # Final evaluation
    logger.info("Final evaluation...")
    metrics = evaluate_model(model, val_loader, device, config['class_names'])
    
    # Save metrics
    metrics_to_save = {
        'best_f1': best_f1,
        'final_metrics': metrics,
        'train_losses': train_losses,
        'val_f1s': val_f1s
    }
    save_json(metrics_to_save, os.path.join(stage_a_dir, 'metrics.json'))
    
    logger.info(f"Training completed. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
