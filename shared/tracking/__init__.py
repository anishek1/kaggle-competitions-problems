"""
MLflow experiment tracking utilities.
Inspired by AI Data Science Team patterns.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class ExperimentTracker:
    """Track experiments with or without MLflow."""
    
    def __init__(
        self, 
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        use_mlflow: bool = True
    ):
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.runs: List[Dict] = []
        self._mlflow = None
        self._run = None
        
        if use_mlflow:
            try:
                import mlflow
                self._mlflow = mlflow
                
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                
                mlflow.set_experiment(experiment_name)
                print(f"📊 MLflow experiment: {experiment_name}")
            except ImportError:
                print("⚠️ MLflow not installed. Using local tracking.")
                self.use_mlflow = False
    
    def start_run(self, run_name: Optional[str] = None) -> 'ExperimentTracker':
        """Start a new experiment run."""
        if self.use_mlflow and self._mlflow:
            self._run = self._mlflow.start_run(run_name=run_name)
        
        self._current_run = {
            'name': run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'params': {},
            'metrics': {},
            'artifacts': [],
            'start_time': datetime.now()
        }
        return self
    
    def log_params(self, params: Dict[str, Any]) -> 'ExperimentTracker':
        """Log parameters."""
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_params(params)
        
        self._current_run['params'].update(params)
        return self
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> 'ExperimentTracker':
        """Log metrics."""
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        
        for key, value in metrics.items():
            self._current_run['metrics'][key] = value
            print(f"  📈 {key}: {value:.4f}")
        
        return self
    
    def log_artifact(self, path: str) -> 'ExperimentTracker':
        """Log an artifact file."""
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_artifact(path)
        
        self._current_run['artifacts'].append(path)
        return self
    
    def log_model(self, model, model_name: str) -> 'ExperimentTracker':
        """Log a model."""
        if self.use_mlflow and self._mlflow:
            try:
                # Try sklearn flavor first
                self._mlflow.sklearn.log_model(model, model_name)
            except:
                pass
        return self
    
    def end_run(self) -> Dict:
        """End the current run."""
        if self.use_mlflow and self._mlflow:
            self._mlflow.end_run()
        
        self._current_run['end_time'] = datetime.now()
        self._current_run['duration'] = (
            self._current_run['end_time'] - self._current_run['start_time']
        ).total_seconds()
        
        self.runs.append(self._current_run)
        
        print(f"✅ Run '{self._current_run['name']}' completed ({self._current_run['duration']:.1f}s)")
        return self._current_run
    
    def get_best_run(self, metric: str, higher_is_better: bool = True) -> Optional[Dict]:
        """Get the best run by a specific metric."""
        if not self.runs:
            return None
        
        valid_runs = [r for r in self.runs if metric in r['metrics']]
        if not valid_runs:
            return None
        
        return max(valid_runs, key=lambda x: x['metrics'][metric] * (1 if higher_is_better else -1))
    
    def summary(self) -> None:
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"📊 Experiment Summary: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total runs: {len(self.runs)}")
        
        if self.runs:
            all_metrics = set()
            for run in self.runs:
                all_metrics.update(run['metrics'].keys())
            
            for metric in all_metrics:
                values = [r['metrics'].get(metric) for r in self.runs if metric in r['metrics']]
                if values:
                    print(f"  {metric}: best={max(values):.4f}, avg={sum(values)/len(values):.4f}")
        
        print(f"{'='*60}\n")


def quick_track(experiment_name: str):
    """Decorator for quick experiment tracking."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = ExperimentTracker(experiment_name, use_mlflow=False)
            tracker.start_run(func.__name__)
            
            try:
                result = func(*args, **kwargs)
                
                # If result is a dict with metrics, log them
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    if metrics:
                        tracker.log_metrics(metrics)
                
                return result
            finally:
                tracker.end_run()
        
        return wrapper
    return decorator
