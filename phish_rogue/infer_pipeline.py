"""
End-to-end inference pipeline combining Stage-A and Stage-B.
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    load_yaml, get_device, setup_logger, load_json, save_json, ensure_dir
)
from .tokenizer import CharTokenizer
from .model_stage_a import URLTiny
from .model_stage_b import FusionHead
from .features_content import HTMLFeatureExtractor
from .metrics import evaluate_model, plot_confusion_matrix, plot_reliability_diagram

def main():
    parser = argparse.ArgumentParser(description='End-to-end inference pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                       required=True, help='Data split to process')
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    
    # Setup
    device = get_device()
    logger = setup_logger('infer_pipeline')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Processing split: {args.split}")
    
    # Load Stage-A model
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
    id_to_label = label_mappings['id_to_label']
    
    logger.info("Stage-A model loaded")
    
    # Load Stage-B model
    stage_b_dir = config['paths']['stage_b_dir']
    stage_b_checkpoint_path = os.path.join(stage_b_dir, 'best.pt')
    
    if os.path.exists(stage_b_checkpoint_path):
        stage_b_checkpoint = torch.load(stage_b_checkpoint_path, map_location=device)
        
        stage_b_model = FusionHead(
            url_embedding_dim=config['stage_a']['model']['d_model'],
            content_dim=config['stage_b']['content_dim'],
            hidden_dim=256,
            num_classes=len(config['class_names']),
            dropout=0.1
        ).to(device)
        
        stage_b_model.load_state_dict(stage_b_checkpoint['model_state_dict'])
        stage_b_model.eval()
        
        logger.info("Stage-B model loaded")
        stage_b_available = True
    else:
        logger.warning("Stage-B model not found, using Stage-A only")
        stage_b_available = False
    
    # Load data
    if args.split == 'train':
        csv_path = config['data']['train_csv']
    elif args.split == 'val':
        csv_path = config['data']['val_csv']
    else:  # test
        csv_path = config['data']['test_csv']
    
    from .dataio import load_csv_data
    urls, labels = load_csv_data(csv_path)
    logger.info(f"Processing {len(urls)} samples")
    
    # Stage-A inference
    logger.info("Running Stage-A inference...")
    input_ids = tokenizer.batch_encode(urls, max_length=config['data']['max_url_len'])
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        logits_a, url_embeddings = stage_a_model(input_ids)
        probs_a = torch.softmax(logits_a, dim=-1)
        preds_a = torch.argmax(logits_a, dim=-1)
        
        # Compute uncertainty metrics
        entropies = -torch.sum(probs_a * torch.log(probs_a + 1e-8), dim=-1)
        max_probs = torch.max(probs_a, dim=-1)[0]
    
    # Convert to numpy
    url_embeddings = url_embeddings.cpu().numpy()
    preds_a = preds_a.cpu().numpy()
    probs_a = probs_a.cpu().numpy()
    entropies = entropies.cpu().numpy()
    max_probs = max_probs.cpu().numpy()
    
    # Determine which samples need Stage-B
    logger.info("Determining samples for Stage-B...")
    escalate_mask = np.zeros(len(urls), dtype=bool)
    
    # High entropy samples
    high_entropy_mask = entropies > config['stage_a']['routing']['entropy_eps']
    escalate_mask |= high_entropy_mask
    
    # Low confidence samples
    for class_name, threshold in config['stage_a']['routing']['min_confidence'].items():
        class_idx = model_config['class_names'].index(class_name)
        low_conf_mask = (preds_a == class_idx) & (max_probs < threshold)
        escalate_mask |= low_conf_mask
    
    escalated_count = escalate_mask.sum()
    logger.info(f"Escalating {escalated_count} samples to Stage-B")
    
    # Stage-B inference for escalated samples
    final_preds = preds_a.copy()
    final_probs = probs_a.copy()
    
    if stage_b_available and escalated_count > 0:
        logger.info("Running Stage-B inference...")
        
        # Extract HTML features for escalated samples
        escalated_urls = [urls[i] for i in range(len(urls)) if escalate_mask[i]]
        escalated_indices = np.where(escalate_mask)[0]
        
        extractor = HTMLFeatureExtractor(
            timeout_ms=config['stage_b']['html_timeout_ms'],
            content_dim=config['stage_b']['content_dim']
        )
        content_features = extractor.batch_extract_features(escalated_urls)
        
        # Get URL embeddings for escalated samples
        escalated_embeddings = url_embeddings[escalate_mask]
        
        # Convert to tensors
        escalated_embeddings_tensor = torch.tensor(escalated_embeddings, dtype=torch.float32).to(device)
        content_features_tensor = torch.tensor(content_features, dtype=torch.float32).to(device)
        
        # Stage-B inference
        with torch.no_grad():
            logits_b = stage_b_model(escalated_embeddings_tensor, content_features_tensor)
            probs_b = torch.softmax(logits_b, dim=-1)
            preds_b = torch.argmax(logits_b, dim=-1)
        
        # Update final predictions
        final_preds[escalate_mask] = preds_b.cpu().numpy()
        final_probs[escalate_mask] = probs_b.cpu().numpy()
        
        logger.info("Stage-B inference completed")
    
    # Create final results DataFrame
    results_df = pd.DataFrame({
        'url': urls,
        'true_label': labels,
        'predicted_label': [id_to_label[str(pred)] for pred in final_preds],
        'stage_a_predicted': [id_to_label[str(pred)] for pred in preds_a],
        'escalated': escalate_mask,
        'max_probability': np.max(final_probs, axis=1),
        'entropy': -np.sum(final_probs * np.log(final_probs + 1e-8), axis=1)
    })
    
    # Add probability columns for each class
    for i, class_name in enumerate(config['class_names']):
        results_df[f'prob_{class_name}'] = final_probs[:, i]
    
    # Save final predictions
    outputs_dir = config['paths']['outputs_dir']
    ensure_dir(outputs_dir)
    
    final_preds_path = os.path.join(outputs_dir, f'final_preds_{args.split}.csv')
    results_df.to_csv(final_preds_path, index=False)
    logger.info(f"Final predictions saved to {final_preds_path}")
    
    # Compute and save metrics
    logger.info("Computing metrics...")
    
    # Convert string labels to integers for metrics computation
    label_to_id = label_mappings['label_to_id']
    true_labels_int = [label_to_id[label] for label in labels]
    pred_labels_int = final_preds
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(true_labels_int, pred_labels_int)
    f1_macro = f1_score(true_labels_int, pred_labels_int, average='macro')
    f1_per_class = f1_score(true_labels_int, pred_labels_int, average=None)
    cm = confusion_matrix(true_labels_int, pred_labels_int)
    
    # Create metrics report
    metrics_report = {
        'split': args.split,
        'total_samples': len(urls),
        'escalated_samples': escalated_count,
        'escalation_rate': escalated_count / len(urls),
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_per_class': {
            config['class_names'][i]: f1_per_class[i] 
            for i in range(len(config['class_names']))
        },
        'confusion_matrix': cm.tolist(),
        'stage_b_used': stage_b_available and escalated_count > 0
    }
    
    # Save metrics report
    reports_path = os.path.join(outputs_dir, f'reports_{args.split}.json')
    save_json(metrics_report, reports_path)
    logger.info(f"Metrics report saved to {reports_path}")
    
    # Create plots if matplotlib is enabled
    if config['logging']['enable_matplotlib']:
        plots_dir = os.path.join(outputs_dir, 'plots')
        ensure_dir(plots_dir)
        
        # Confusion matrix plot
        plot_confusion_matrix(
            cm,
            config['class_names'],
            save_path=os.path.join(plots_dir, f'confusion_matrix_{args.split}.png'),
            title=f"Confusion Matrix - {args.split.title()}"
        )
        
        # Reliability diagram
        plot_reliability_diagram(
            true_labels_int,
            np.max(final_probs, axis=1),
            save_path=os.path.join(plots_dir, f'reliability_diagram_{args.split}.png')
        )
        
        logger.info(f"Plots saved to {plots_dir}")
    
    # Print summary
    logger.info(f"Pipeline completed for {args.split} split:")
    logger.info(f"  Total samples: {len(urls)}")
    logger.info(f"  Escalated samples: {escalated_count}")
    logger.info(f"  Escalation rate: {escalated_count/len(urls):.2%}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Macro: {f1_macro:.4f}")
    logger.info(f"  Stage-B used: {stage_b_available and escalated_count > 0}")

if __name__ == '__main__':
    main()
