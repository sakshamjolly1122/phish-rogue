"""
Inference script for Stage-A model.
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import load_yaml, get_device, setup_logger, load_json
from .tokenizer import CharTokenizer
from .model_stage_a import URLTiny
from .dataio import load_csv_data, make_dataloader

def main():
    parser = argparse.ArgumentParser(description='Infer Stage-A model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                       required=True, help='Data split to process')
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    
    # Setup
    device = get_device()
    logger = setup_logger('infer_stage_a')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Processing split: {args.split}")
    
    # Load model and tokenizer
    stage_a_dir = config['paths']['stage_a_dir']
    checkpoint_path = os.path.join(stage_a_dir, 'best.pt')
    tokenizer_path = os.path.join(stage_a_dir, 'tokenizer.json')
    label_mappings_path = os.path.join(stage_a_dir, 'label_mappings.json')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    
    # Initialize model
    model = URLTiny(
        vocab_size=model_config['stage_a']['model']['vocab_size'],
        d_model=model_config['stage_a']['model']['d_model'],
        nheads=model_config['stage_a']['model']['nheads'],
        nlayers=model_config['stage_a']['model']['nlayers'],
        num_classes=len(model_config['class_names']),
        max_length=model_config['data']['max_url_len']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer and label mappings
    tokenizer = CharTokenizer.load(tokenizer_path)
    label_mappings = load_json(label_mappings_path)
    id_to_label = label_mappings['id_to_label']
    
    logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
    logger.info(f"Best F1: {checkpoint['best_f1']:.4f}")
    
    # Load data
    if args.split == 'train':
        csv_path = config['data']['train_csv']
    elif args.split == 'val':
        csv_path = config['data']['val_csv']
    else:  # test
        csv_path = config['data']['test_csv']
    
    urls, labels = load_csv_data(csv_path)
    logger.info(f"Processing {len(urls)} samples")
    
    # Create dataloader
    from .augment import URLAugmenter
    augmenter = URLAugmenter(augment_prob=0.0)  # No augmentation during inference
    
    dataloader = make_dataloader(
        urls, labels, tokenizer, augmenter,
        batch_size=config['stage_a']['train']['batch_size'],
        max_length=config['data']['max_url_len'],
        is_training=False,
        shuffle=False
    )
    
    # Inference
    all_predictions = []
    all_probabilities = []
    all_entropies = []
    all_max_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            
            logits, embeddings = model(input_ids)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            # Compute entropy and max probability
            entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_entropies.extend(entropies.cpu().numpy())
            all_max_probs.extend(max_probs.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_entropies = np.array(all_entropies)
    all_max_probs = np.array(all_max_probs)
    all_embeddings = np.array(all_embeddings)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'url': urls,
        'true_label': labels,
        'predicted_label': [id_to_label[str(pred)] for pred in all_predictions],
        'max_probability': all_max_probs,
        'entropy': all_entropies
    })
    
    # Add probability columns for each class
    for i, class_name in enumerate(model_config['class_names']):
        pred_df[f'prob_{class_name}'] = all_probabilities[:, i]
    
    # Save predictions
    pred_path = os.path.join(stage_a_dir, f'preds_{args.split}.csv')
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")
    
    # Save embeddings
    embeddings_path = os.path.join(stage_a_dir, f'embeddings_{args.split}.npy')
    np.save(embeddings_path, all_embeddings)
    logger.info(f"Embeddings saved to {embeddings_path}")
    
    # Print summary statistics
    logger.info(f"Mean max probability: {np.mean(all_max_probs):.4f}")
    logger.info(f"Mean entropy: {np.mean(all_entropies):.4f}")
    
    # Compute accuracy if we have labels
    if labels:
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels, [id_to_label[str(pred)] for pred in all_predictions])
        logger.info(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
