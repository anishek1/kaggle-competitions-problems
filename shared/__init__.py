"""
Shared utilities for Kaggle competitions and dataset analysis.
Enhanced with AI-powered features inspired by business-science/ai-data-science-team.
"""

from . import data
from . import models
from . import visualization
from . import evaluation
from . import utils
from . import tracking
from . import automl
from . import auto_eda
from . import pipeline

__all__ = [
    'data', 
    'models', 
    'visualization', 
    'evaluation', 
    'utils',
    'tracking',      # MLflow experiment tracking
    'automl',        # H2O, FLAML, Auto-sklearn
    'auto_eda',      # Automated EDA with recommendations
    'pipeline',      # Reproducible data pipelines
]
