"""
Data Science Pipeline utilities.
Chain operations into reproducible, trackable workflows.
Inspired by AI Data Science Team patterns.
"""

import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PipelineStep:
    """A single step in the pipeline."""
    name: str
    func: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    executed: bool = False
    execution_time: float = 0.0
    input_shape: tuple = None
    output_shape: tuple = None


class Pipeline:
    """
    Reproducible data science pipeline.
    Chain transformations, track execution, and serialize for reproducibility.
    """
    
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.history: List[Dict] = []
        self._data = None
        self.created_at = datetime.now()
    
    def add_step(
        self,
        name: str,
        func: Callable,
        params: Dict[str, Any] = None,
        description: str = ""
    ) -> 'Pipeline':
        """Add a step to the pipeline."""
        step = PipelineStep(
            name=name,
            func=func,
            params=params or {},
            description=description
        )
        self.steps.append(step)
        print(f"  ➕ Added step: {name}")
        return self
    
    def run(self, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Execute the pipeline."""
        import time
        
        if verbose:
            print(f"\n🚀 Running Pipeline: {self.name}")
            print("="*50)
        
        self._data = data.copy()
        
        for i, step in enumerate(self.steps):
            if verbose:
                print(f"\n[{i+1}/{len(self.steps)}] {step.name}")
            
            input_shape = self._data.shape
            start_time = time.perf_counter()
            
            try:
                self._data = step.func(self._data, **step.params)
                
                step.executed = True
                step.execution_time = time.perf_counter() - start_time
                step.input_shape = input_shape
                step.output_shape = self._data.shape
                
                self.history.append({
                    'step': step.name,
                    'status': 'success',
                    'input_shape': input_shape,
                    'output_shape': self._data.shape,
                    'time': step.execution_time
                })
                
                if verbose:
                    print(f"   ✅ {input_shape} → {self._data.shape} ({step.execution_time:.2f}s)")
                    
            except Exception as e:
                self.history.append({
                    'step': step.name,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"   ❌ Error: {e}")
                raise
        
        if verbose:
            print("\n" + "="*50)
            print(f"✅ Pipeline complete! Final shape: {self._data.shape}")
        
        return self._data
    
    def summary(self) -> pd.DataFrame:
        """Get pipeline execution summary."""
        return pd.DataFrame([
            {
                'Step': s.name,
                'Executed': '✅' if s.executed else '❌',
                'Input': s.input_shape,
                'Output': s.output_shape,
                'Time (s)': f"{s.execution_time:.2f}" if s.executed else '-'
            }
            for s in self.steps
        ])
    
    def save(self, path: str) -> None:
        """Save pipeline config (not functions, just metadata)."""
        config = {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'steps': [
                {
                    'name': s.name,
                    'description': s.description,
                    'params': s.params
                }
                for s in self.steps
            ],
            'history': self.history
        }
        
        Path(path).write_text(json.dumps(config, indent=2, default=str))
        print(f"💾 Pipeline config saved to {path}")
    
    def __repr__(self) -> str:
        return f"Pipeline('{self.name}', steps={len(self.steps)})"


# ============== Common Pipeline Steps ==============

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop specified columns."""
    return df.drop(columns=[c for c in columns if c in df.columns], errors='ignore')


def fill_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Fill missing values."""
    df = df.copy()
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if strategy == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == 'zero':
        df[num_cols] = df[num_cols].fillna(0)
    
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    return df


def encode_categoricals(df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
    """Encode categorical columns."""
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df


def scale_features(df: pd.DataFrame, method: str = 'standard', exclude: List[str] = None) -> pd.DataFrame:
    """Scale numerical features."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    df = df.copy()
    exclude = exclude or []
    
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def remove_outliers(df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> pd.DataFrame:
    """Remove outliers using IQR method."""
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    return df


def log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log transformation to specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


# ============== Quick Pipeline Builder ==============

def create_preprocessing_pipeline(
    drop_cols: List[str] = None,
    fill_strategy: str = 'median',
    encode_method: str = 'label',
    scale: bool = False,
    target_col: str = None
) -> Pipeline:
    """Create a standard preprocessing pipeline."""
    
    pipeline = Pipeline("Preprocessing")
    
    if drop_cols:
        pipeline.add_step(
            "Drop Columns",
            drop_columns,
            {'columns': drop_cols},
            f"Drop {len(drop_cols)} columns"
        )
    
    pipeline.add_step(
        "Fill Missing",
        fill_missing,
        {'strategy': fill_strategy},
        f"Fill missing values using {fill_strategy}"
    )
    
    pipeline.add_step(
        "Encode Categoricals",
        encode_categoricals,
        {'method': encode_method},
        f"Encode categoricals using {encode_method} encoding"
    )
    
    if scale:
        exclude = [target_col] if target_col else []
        pipeline.add_step(
            "Scale Features",
            scale_features,
            {'method': 'standard', 'exclude': exclude},
            "Standardize numerical features"
        )
    
    return pipeline
