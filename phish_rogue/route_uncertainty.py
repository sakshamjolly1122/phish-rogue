"""
Uncertainty routing for Stage-A predictions.
"""
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List

from .utils import load_yaml, setup_logger, save_json

def compute_uncertainty_flags(
    df: pd.DataFrame,
    entropy_eps: float = 0.6,
    min_confidence: Dict[str, float] = None
) -> pd.DataFrame:
    """Compute uncertainty flags for routing to Stage-B."""
    
    if min_confidence is None:
        min_confidence = {
            'benign': 0.90,
            'phishing': 0.85,
            'malware': 0.85,
            'spam': 0.85,
            'defacement': 0.85
        }
    
    # Initialize escalate column
    df['escalate'] = False
    
    # High entropy samples
    high_entropy_mask = df['entropy'] > entropy_eps
    df.loc[high_entropy_mask, 'escalate'] = True
    
    # Low confidence samples (class-specific thresholds)
    for class_name, threshold in min_confidence.items():
        prob_col = f'prob_{class_name}'
        if prob_col in df.columns:
            low_conf_mask = (df['predicted_label'] == class_name) & (df['max_probability'] < threshold)
            df.loc[low_conf_mask, 'escalate'] = True
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Route uncertain samples to Stage-B')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to Stage-A predictions CSV')
    parser.add_argument('--out', type=str, required=True,
                       help='Path to output routed CSV')
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    logger = setup_logger('route_uncertainty')
    
    logger.info(f"Loading predictions from {args.input}")
    
    # Load predictions
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} predictions")
    
    # Compute uncertainty flags
    df = compute_uncertainty_flags(
        df,
        entropy_eps=config['stage_a']['routing']['entropy_eps'],
        min_confidence=config['stage_a']['routing']['min_confidence']
    )
    
    # Save routed predictions
    df.to_csv(args.out, index=False)
    logger.info(f"Routed predictions saved to {args.out}")
    
    # Print statistics
    total_samples = len(df)
    escalated_samples = df['escalate'].sum()
    escalation_rate = escalated_samples / total_samples
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Escalated samples: {escalated_samples}")
    logger.info(f"Escalation rate: {escalation_rate:.2%}")
    
    # Per-class escalation rates
    logger.info("Per-class escalation rates:")
    for class_name in config['class_names']:
        class_mask = df['predicted_label'] == class_name
        if class_mask.sum() > 0:
            class_escalation_rate = df[class_mask]['escalate'].mean()
            logger.info(f"  {class_name}: {class_escalation_rate:.2%}")
    
    # Save escalation statistics
    escalation_stats = {
        'total_samples': total_samples,
        'escalated_samples': escalated_samples,
        'escalation_rate': escalation_rate,
        'per_class_escalation': {}
    }
    
    for class_name in config['class_names']:
        class_mask = df['predicted_label'] == class_name
        if class_mask.sum() > 0:
            escalation_stats['per_class_escalation'][class_name] = df[class_mask]['escalate'].mean()
    
    stats_path = args.out.replace('.csv', '_stats.json')
    save_json(escalation_stats, stats_path)
    logger.info(f"Escalation statistics saved to {stats_path}")
    
    # Create escalated and non-escalated subsets
    escalated_df = df[df['escalate']].copy()
    non_escalated_df = df[~df['escalate']].copy()
    
    escalated_path = args.out.replace('.csv', '_escalated.csv')
    non_escalated_path = args.out.replace('.csv', '_non_escalated.csv')
    
    escalated_df.to_csv(escalated_path, index=False)
    non_escalated_df.to_csv(non_escalated_path, index=False)
    
    logger.info(f"Escalated subset saved to {escalated_path}")
    logger.info(f"Non-escalated subset saved to {non_escalated_path}")

if __name__ == '__main__':
    main()
