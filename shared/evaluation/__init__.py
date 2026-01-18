"""
Metrics and evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from sklearn import metrics


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
    """Generate a comprehensive classification report."""
    report = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_proba is not None:
        try:
            report['roc_auc'] = metrics.roc_auc_score(y_true, y_proba)
            report['log_loss'] = metrics.log_loss(y_true, y_proba)
        except ValueError:
            pass
    
    return report


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Generate a comprehensive regression report."""
    return {
        'rmse': np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        'mae': metrics.mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100,
        'r2': metrics.r2_score(y_true, y_pred),
    }


def print_metrics(metrics_dict: Dict, title: str = "Evaluation Metrics") -> None:
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"📊 {title}")
    print(f"{'='*50}")
    for name, value in metrics_dict.items():
        print(f"  {name.upper():15s}: {value:.4f}")
    print(f"{'='*50}\n")


def oof_score(
    y_true: np.ndarray,
    oof_preds: np.ndarray,
    metric_fn: Callable,
    threshold: float = 0.5,
    is_classification: bool = True
) -> float:
    """Calculate OOF (Out-of-Fold) score."""
    if is_classification:
        preds = (oof_preds > threshold).astype(int)
    else:
        preds = oof_preds
    
    score = metric_fn(y_true, preds)
    print(f"📊 OOF Score: {score:.4f}")
    return score


def create_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    id_col: str = 'id',
    target_col: str = 'target',
    filename: str = 'submission.csv',
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """Create a submission file."""
    if threshold is not None:
        predictions = (predictions > threshold).astype(int)
    
    submission = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"💾 Submission saved to {filename}")
    print(f"📊 Shape: {submission.shape}")
    print(f"📊 Preview:\n{submission.head()}")
    
    return submission


METRIC_MAPPING = {
    # Classification
    'accuracy': metrics.accuracy_score,
    'precision': lambda y, p: metrics.precision_score(y, p, average='weighted', zero_division=0),
    'recall': lambda y, p: metrics.recall_score(y, p, average='weighted', zero_division=0),
    'f1': lambda y, p: metrics.f1_score(y, p, average='weighted', zero_division=0),
    'roc_auc': metrics.roc_auc_score,
    'log_loss': metrics.log_loss,
    
    # Regression
    'rmse': lambda y, p: np.sqrt(metrics.mean_squared_error(y, p)),
    'mae': metrics.mean_absolute_error,
    'mse': metrics.mean_squared_error,
    'r2': metrics.r2_score,
}


def get_metric(name: str) -> Callable:
    """Get a metric function by name."""
    if name not in METRIC_MAPPING:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_MAPPING.keys())}")
    return METRIC_MAPPING[name]
