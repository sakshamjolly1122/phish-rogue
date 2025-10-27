"""
Evaluation metrics and visualization utilities.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from .utils import ensure_dir

def compute_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str]) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)

def compute_f1_scores(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, float]:
    """Compute per-class and macro F1 scores."""
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    f1_scores = {}
    for i, class_name in enumerate(class_names):
        f1_scores[f'f1_{class_name}'] = f1_per_class[i]
    f1_scores['f1_macro'] = f1_macro
    
    return f1_scores

def compute_expected_calibration_error(y_true: List[int], y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reliability_diagram(
    y_true: List[int],
    y_probs: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 10
):
    """Plot reliability diagram."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(y_true[in_bin].mean())
            bin_confidences.append(y_probs[in_bin].mean())
            bin_counts.append(prop_in_bin)
    
    plt.figure(figsize=(8, 8))
    plt.bar(bin_centers, bin_accuracies, width=0.1, alpha=0.7, label='Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

def compute_max_prob(probs: torch.Tensor) -> torch.Tensor:
    """Compute maximum probability."""
    return torch.max(probs, dim=-1)[0]

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_entropies = []
    all_max_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            entropies = compute_entropy(probs)
            max_probs = compute_max_prob(probs)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_entropies.extend(entropies.cpu().numpy())
            all_max_probs.extend(max_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_entropies = np.array(all_entropies)
    all_max_probs = np.array(all_max_probs)
    
    # Compute metrics
    cm = compute_confusion_matrix(all_labels, all_preds, class_names)
    f1_scores = compute_f1_scores(all_labels, all_preds, class_names)
    ece = compute_expected_calibration_error(all_labels, all_max_probs)
    
    # Get max probabilities for each sample
    max_probs_per_sample = np.max(all_probs, axis=1)
    
    metrics = {
        'confusion_matrix': cm.tolist(),
        'f1_scores': f1_scores,
        'ece': ece,
        'mean_entropy': np.mean(all_entropies),
        'mean_max_prob': np.mean(max_probs_per_sample),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'entropies': all_entropies.tolist(),
        'max_probs': max_probs_per_sample.tolist()
    }
    
    return metrics
