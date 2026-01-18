"""
Visualization utilities for EDA and model analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def setup_plotting(figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
    """Setup matplotlib/seaborn defaults."""
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    sns.set_style(style)


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    ax: Optional[plt.Axes] = None,
    title: str = "Target Distribution"
) -> plt.Axes:
    """Plot target variable distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[target_col].nunique() <= 20:  # Categorical
        df[target_col].value_counts().plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_ylabel('Count')
    else:  # Continuous
        df[target_col].hist(bins=50, ax=ax, color='steelblue', edgecolor='black')
        ax.set_ylabel('Frequency')
    
    ax.set_title(title)
    ax.set_xlabel(target_col)
    plt.tight_layout()
    return ax


def plot_missing_values(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    threshold: float = 0.0
) -> plt.Axes:
    """Plot missing values per column."""
    missing = df.isnull().sum() / len(df) * 100
    missing = missing[missing > threshold].sort_values(ascending=True)
    
    if len(missing) == 0:
        print("✅ No missing values above threshold!")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing) * 0.3)))
    
    missing.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Missing %')
    ax.set_title('Missing Values by Column')
    
    for i, v in enumerate(missing):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    return ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True
) -> plt.Figure:
    """Plot correlation heatmap."""
    if cols:
        corr = df[cols].corr()
    else:
        corr = df.select_dtypes(include=[np.number]).corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=annot, 
        fmt='.2f', 
        cmap='RdBu_r',
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5
    )
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    n_features: int = 20,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot feature importance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.3)))
    
    top_features = importance_df.head(n_features).sort_values('importance')
    
    ax.barh(top_features['feature'], top_features['importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {n_features} Feature Importance')
    plt.tight_layout()
    return ax


def plot_numerical_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    ncols: int = 3,
    figsize_per_plot: Tuple[int, int] = (4, 3)
) -> plt.Figure:
    """Plot distributions of numerical columns."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for i, col in enumerate(cols):
        df[col].hist(bins=30, ax=axes[i], color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel('')
    
    # Hide empty subplots
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    ncols: int = 3,
    figsize_per_plot: Tuple[int, int] = (4, 3)
) -> plt.Figure:
    """Plot value counts of categorical columns."""
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    for i, col in enumerate(cols):
        df[col].value_counts().head(10).plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig
