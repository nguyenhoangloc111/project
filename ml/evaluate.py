"""
Enhanced evaluation metrics for model visualization
Includes PR-AUC, ROC-AUC, and data for charting
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc, roc_auc_score
)

def calculate_metrics_with_viz(model, X_data, y_true):
    """
    Calculate comprehensive metrics including visualization data
    
    Args:
        model: trained Isolation Forest model
        X_data: features data
        y_true: true labels (1 = anomaly, 0 = normal)
        
    Returns:
        Dictionary with all metrics and visualization data
    """
    if y_true is None:
        return None
    
    # Get anomaly scores (lower = more anomalous)
    scores = model.score_samples(X_data)
    
    # Normalize scores to [0, 1] range for probability
    scores_min = scores.min()
    scores_max = scores.max()
    y_proba = 1 - (scores - scores_min) / (scores_max - scores_min) if scores_max > scores_min else np.ones_like(scores)
    
    # Optimize threshold (48% percentile)
    threshold_percentile = np.percentile(scores, 48)
    y_pred = (scores < threshold_percentile).astype(int)
    
    # Ensure y_true is binary
    y_true_binary = (y_true == 1).astype(int)
    
    # === BASIC METRICS ===
    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    # === AUC METRICS ===
    try:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true_binary, y_proba)
        pr_auc = auc(pr_recall, pr_precision)
    except:
        pr_precision, pr_recall = [0], [0]
        pr_auc = 0
    
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_proba)
        roc_auc = roc_auc_score(y_true_binary, y_proba)
    except:
        fpr, tpr = [0, 1], [0, 1]
        roc_auc = 0
    
    # === THRESHOLD ANALYSIS ===
    thresholds_to_test = np.linspace(0, 1, 15)  # Reduced from 21 to 15 points
    threshold_data = []
    
    for t in thresholds_to_test:
        y_pred_t = (y_proba >= t).astype(int)
        if (y_pred_t == 1).sum() > 0:
            prec_t = precision_score(y_true_binary, y_pred_t, zero_division=0)
            rec_t = recall_score(y_true_binary, y_pred_t, zero_division=0)
            f1_t = f1_score(y_true_binary, y_pred_t, zero_division=0)
        else:
            prec_t, rec_t, f1_t = 0, 0, 0
        
        threshold_data.append({
            'threshold': round(float(t), 3),
            'precision': round(float(prec_t), 4),
            'recall': round(float(rec_t), 4),
            'f1_score': round(float(f1_t), 4)
        })
    
    # === SCORE DISTRIBUTION ===
    sample_size = min(3000, len(X_data))  # Reduced from 5000 to 3000
    sample_indices = np.random.choice(len(X_data), sample_size, replace=False)
    
    scores_sample = y_proba[sample_indices]
    y_true_sample = y_true_binary[sample_indices]
    
    bins = np.linspace(0, 1, 16)  # Reduced from 21 to 16 bins
    normal_hist, _ = np.histogram(scores_sample[y_true_sample == 0], bins=bins)
    abnormal_hist, _ = np.histogram(scores_sample[y_true_sample == 1], bins=bins)
    bin_edges = (bins[:-1] + bins[1:]) / 2
    
    score_distribution = {
        'bins': [round(float(b), 3) for b in bin_edges],
        'normal': [int(h) for h in normal_hist],
        'abnormal': [int(h) for h in abnormal_hist]
    }
    
    # === PR CURVE DATA ===
    pr_sample_indices = np.linspace(0, len(pr_recall)-1, min(30, len(pr_recall)), dtype=int)  # Reduced from 50 to 30
    pr_curve = {
        'recall': [round(float(pr_recall[i]), 4) for i in pr_sample_indices],
        'precision': [round(float(pr_precision[i]), 4) for i in pr_sample_indices]
    }
    
    # === ROC CURVE DATA ===
    roc_sample_indices = np.linspace(0, len(fpr)-1, min(30, len(fpr)), dtype=int)  # Reduced from 50 to 30
    roc_curve_data = {
        'fpr': [round(float(fpr[i]), 4) for i in roc_sample_indices],
        'tpr': [round(float(tpr[i]), 4) for i in roc_sample_indices]
    }
    
    # === FIND OPTIMAL THRESHOLD ===
    optimal_idx = np.argmax([td['f1_score'] for td in threshold_data])
    optimal_threshold = threshold_data[optimal_idx]['threshold']
    
    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'pr_auc': round(float(pr_auc), 4),
        'roc_auc': round(float(roc_auc), 4),
        'threshold_analysis': threshold_data,
        'score_distribution': score_distribution,
        'pr_curve': pr_curve,
        'roc_curve': roc_curve_data,
        'optimal_threshold': optimal_threshold
    }
