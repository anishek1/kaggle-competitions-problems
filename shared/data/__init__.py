"""
Data loading, preprocessing, and feature engineering utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union


def load_competition_data(
    competition_path: Union[str, Path],
    train_file: str = "train.csv",
    test_file: str = "test.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test data from competition folder."""
    path = Path(competition_path)
    raw_path = path / "data" / "raw"
    
    train = pd.read_csv(raw_path / train_file)
    test = pd.read_csv(raw_path / test_file)
    
    print(f"📊 Train shape: {train.shape}")
    print(f"📊 Test shape: {test.shape}")
    
    return train, test


def create_folds(
    df: pd.DataFrame,
    target_col: str,
    n_folds: int = 5,
    stratified: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """Create cross-validation folds."""
    from sklearn.model_selection import KFold, StratifiedKFold
    
    df = df.copy()
    df['fold'] = -1
    
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(kf.split(df, df[target_col])):
            df.loc[val_idx, 'fold'] = fold
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold
    
    print(f"✅ Created {n_folds} folds")
    return df


def get_feature_types(df: pd.DataFrame) -> dict:
    """Identify numerical, categorical, and datetime columns."""
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numerical': numerical,
        'categorical': categorical,
        'datetime': datetime
    }


def basic_preprocessing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply basic preprocessing to train and test."""
    
    # Combine for consistent preprocessing
    if target_col and target_col in train.columns:
        train_target = train[target_col]
        train = train.drop(columns=[target_col])
    else:
        train_target = None
    
    train['_is_train'] = 1
    test['_is_train'] = 0
    combined = pd.concat([train, test], ignore_index=True)
    
    # Handle missing values info
    missing = combined.isnull().sum()
    if missing.any():
        print(f"⚠️ Columns with missing values:")
        print(missing[missing > 0])
    
    # Split back
    train = combined[combined['_is_train'] == 1].drop(columns=['_is_train'])
    test = combined[combined['_is_train'] == 0].drop(columns=['_is_train'])
    
    if train_target is not None:
        train[target_col] = train_target.values
    
    return train, test


def save_processed_data(
    df: pd.DataFrame,
    competition_path: Union[str, Path],
    filename: str
) -> None:
    """Save processed data to the processed folder."""
    path = Path(competition_path) / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    
    filepath = path / filename
    df.to_csv(filepath, index=False)
    print(f"💾 Saved to {filepath}")
