"""
Model training, prediction, and ensemble utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import joblib


class BaseModel:
    """Base class for model wrappers."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.model = None
        self.feature_names = None
    
    def fit(self, X, y, **kwargs):
        raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        model = joblib.load(path)
        print(f"📂 Model loaded from {path}")
        return model


class LGBMWrapper(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, params: Dict[str, Any] = None, task: str = 'classification'):
        super().__init__(params)
        self.task = task
        
        default_params = {
            'objective': 'binary' if task == 'classification' else 'regression',
            'metric': 'auc' if task == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        default_params.update(self.params)
        self.params = default_params
    
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        import lightgbm as lgb
        
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=kwargs.get('num_boost_round', 1000),
            callbacks=[
                lgb.early_stopping(stopping_rounds=kwargs.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(period=kwargs.get('verbose_eval', 100))
            ]
        )
        return self
    
    def predict(self, X) -> np.ndarray:
        preds = self.model.predict(X)
        if self.task == 'classification':
            return (preds > 0.5).astype(int)
        return preds
    
    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        importance = pd.DataFrame({
            'feature': self.feature_names or range(len(self.model.feature_importance())),
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        return importance


class XGBWrapper(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, params: Dict[str, Any] = None, task: str = 'classification'):
        super().__init__(params)
        self.task = task
        
        default_params = {
            'objective': 'binary:logistic' if task == 'classification' else 'reg:squarederror',
            'eval_metric': 'auc' if task == 'classification' else 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'verbosity': 0
        }
        default_params.update(self.params)
        self.params = default_params
    
    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        import xgboost as xgb
        
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'valid'))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=kwargs.get('num_boost_round', 1000),
            evals=evals,
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
            verbose_eval=kwargs.get('verbose_eval', 100)
        )
        return self
    
    def predict(self, X) -> np.ndarray:
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        preds = self.model.predict(dtest)
        if self.task == 'classification':
            return (preds > 0.5).astype(int)
        return preds
    
    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)


def cross_validate(
    model_class,
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    params: Dict[str, Any] = None,
    task: str = 'classification',
    **fit_kwargs
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Run cross-validation and return OOF predictions and test predictions."""
    
    oof_preds = np.zeros(len(X))
    models = []
    
    for fold in sorted(folds.unique()):
        if fold < 0:
            continue
            
        print(f"\n{'='*50}")
        print(f"📁 Fold {fold + 1}")
        print(f"{'='*50}")
        
        train_idx = folds != fold
        val_idx = folds == fold
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_class(params=params, task=task)
        model.fit(X_train, y_train, X_val, y_val, **fit_kwargs)
        
        if task == 'classification':
            oof_preds[val_idx] = model.predict_proba(X_val)
        else:
            oof_preds[val_idx] = model.predict(X_val)
        
        models.append(model)
    
    return oof_preds, models
