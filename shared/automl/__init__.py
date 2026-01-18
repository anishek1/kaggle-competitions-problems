"""
AutoML utilities - H2O, Auto-sklearn, and ensemble approaches.
Inspired by AI Data Science Team patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path


class AutoMLRunner:
    """Unified AutoML interface supporting multiple backends."""
    
    def __init__(
        self,
        task: str = 'classification',
        time_budget: int = 300,  # seconds
        n_jobs: int = -1,
        seed: int = 42
    ):
        self.task = task
        self.time_budget = time_budget
        self.n_jobs = n_jobs
        self.seed = seed
        self.best_model = None
        self.leaderboard = None
        self._backend = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        backend: str = 'auto'
    ) -> 'AutoMLRunner':
        """
        Fit AutoML with specified backend.
        
        Args:
            X: Features
            y: Target
            backend: 'h2o', 'autosklearn', 'flaml', or 'auto' (tries in order)
        """
        if backend == 'auto':
            for b in ['flaml', 'h2o', 'autosklearn']:
                try:
                    return self._fit_backend(X, y, b)
                except ImportError:
                    continue
            raise ImportError("No AutoML backend available. Install flaml, h2o, or auto-sklearn.")
        
        return self._fit_backend(X, y, backend)
    
    def _fit_backend(self, X: pd.DataFrame, y: pd.Series, backend: str) -> 'AutoMLRunner':
        """Fit with specific backend."""
        
        if backend == 'h2o':
            return self._fit_h2o(X, y)
        elif backend == 'flaml':
            return self._fit_flaml(X, y)
        elif backend == 'autosklearn':
            return self._fit_autosklearn(X, y)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _fit_h2o(self, X: pd.DataFrame, y: pd.Series) -> 'AutoMLRunner':
        """Fit using H2O AutoML."""
        import h2o
        from h2o.automl import H2OAutoML
        
        h2o.init(nthreads=self.n_jobs)
        
        # Combine X and y for H2O
        train_df = X.copy()
        train_df['target'] = y.values
        h2o_train = h2o.H2OFrame(train_df)
        
        # Set target type
        if self.task == 'classification':
            h2o_train['target'] = h2o_train['target'].asfactor()
        
        # Run AutoML
        aml = H2OAutoML(
            max_runtime_secs=self.time_budget,
            seed=self.seed,
            sort_metric='AUC' if self.task == 'classification' else 'RMSE'
        )
        
        aml.train(y='target', training_frame=h2o_train)
        
        self.best_model = aml.leader
        self.leaderboard = aml.leaderboard.as_data_frame()
        self._backend = 'h2o'
        self._h2o = h2o
        
        print(f"🏆 Best H2O model: {self.leaderboard.iloc[0]['model_id']}")
        return self
    
    def _fit_flaml(self, X: pd.DataFrame, y: pd.Series) -> 'AutoMLRunner':
        """Fit using FLAML (Fast and Lightweight AutoML)."""
        from flaml import AutoML
        
        automl = AutoML()
        
        automl.fit(
            X, y,
            task=self.task if self.task == 'classification' else 'regression',
            time_budget=self.time_budget,
            n_jobs=self.n_jobs,
            seed=self.seed,
            verbose=1
        )
        
        self.best_model = automl
        self.leaderboard = pd.DataFrame({
            'model': [automl.best_estimator],
            'score': [automl.best_loss]
        })
        self._backend = 'flaml'
        
        print(f"🏆 Best FLAML model: {automl.best_estimator}")
        return self
    
    def _fit_autosklearn(self, X: pd.DataFrame, y: pd.Series) -> 'AutoMLRunner':
        """Fit using Auto-sklearn."""
        if self.task == 'classification':
            from autosklearn.classification import AutoSklearnClassifier
            automl = AutoSklearnClassifier(
                time_left_for_this_task=self.time_budget,
                n_jobs=self.n_jobs,
                seed=self.seed
            )
        else:
            from autosklearn.regression import AutoSklearnRegressor
            automl = AutoSklearnRegressor(
                time_left_for_this_task=self.time_budget,
                n_jobs=self.n_jobs,
                seed=self.seed
            )
        
        automl.fit(X, y)
        
        self.best_model = automl
        self.leaderboard = pd.DataFrame(automl.leaderboard())
        self._backend = 'autosklearn'
        
        print(f"🏆 Best Auto-sklearn model found")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self._backend == 'h2o':
            h2o_test = self._h2o.H2OFrame(X)
            preds = self.best_model.predict(h2o_test).as_data_frame()
            if self.task == 'classification':
                return preds['p1'].values if 'p1' in preds.columns else preds.iloc[:, -1].values
            return preds['predict'].values
        else:
            return self.best_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self._backend == 'h2o':
            h2o_test = self._h2o.H2OFrame(X)
            preds = self.best_model.predict(h2o_test).as_data_frame()
            return preds['p1'].values if 'p1' in preds.columns else preds.iloc[:, -1].values
        else:
            return self.best_model.predict_proba(X)[:, 1]
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get AutoML leaderboard."""
        return self.leaderboard


def quick_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    task: str = 'classification',
    time_budget: int = 300
) -> Tuple[np.ndarray, 'AutoMLRunner']:
    """Quick AutoML prediction."""
    runner = AutoMLRunner(task=task, time_budget=time_budget)
    runner.fit(X_train, y_train)
    
    if task == 'classification':
        preds = runner.predict_proba(X_test)
    else:
        preds = runner.predict(X_test)
    
    return preds, runner
