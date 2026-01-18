# 🏆 Kaggle Competitions Hub

A modular, reusable project structure for solving multiple Kaggle competitions and dataset problems.

## 📁 Project Structure

```
kaggle/
├── competitions/           # Individual competition folders
│   └── _template/          # Template for new competitions
├── datasets/               # Standalone dataset analysis projects
│   └── _template/          # Template for dataset projects
├── shared/                 # Shared utilities and modules
│   ├── data/               # Common data processing utilities
│   ├── models/             # Reusable model architectures
│   ├── visualization/      # Plotting and visualization helpers
│   ├── evaluation/         # Metrics and evaluation utilities
│   └── utils/              # General utility functions
├── notebooks/              # Experimental/scratch notebooks
├── configs/                # Global configuration files
├── scripts/                # Automation scripts
└── artifacts/              # Saved models, submissions, etc.
```

## 🚀 Quick Start

### 1. Create a New Competition Project
```bash
python scripts/new_competition.py "titanic" --type classification
```

### 2. Create a New Dataset Project
```bash
python scripts/new_dataset.py "housing-prices"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🔧 Features

- **Modular Structure**: Reusable code across competitions
- **Template System**: Quick-start templates for new projects
- **Shared Utilities**: Common preprocessing, visualization, and modeling code
- **Experiment Tracking**: Built-in logging and experiment management
- **Submission Pipeline**: Automated submission generation

## 📋 Competition Workflow

1. **Setup**: Copy template, download data
2. **EDA**: Explore data, understand distributions
3. **Preprocessing**: Clean data, feature engineering
4. **Modeling**: Train, validate, ensemble
5. **Submission**: Generate and submit predictions

## 🛠️ Shared Modules

| Module | Description |
|--------|-------------|
| `shared.data` | Data loading, preprocessing, feature engineering |
| `shared.models` | Model wrappers, ensembles, AutoML integrations |
| `shared.visualization` | EDA plots, model diagnostics, submission analysis |
| `shared.evaluation` | Metrics, cross-validation, leaderboard tracking |
| `shared.utils` | Seeds, logging, file management, timing |
| **`shared.auto_eda`** | 🤖 **Automated EDA** - One-line data profiling with recommendations |
| **`shared.automl`** | 🚀 **AutoML** - Unified interface for FLAML, H2O, Auto-sklearn |
| **`shared.tracking`** | 📊 **Experiment Tracking** - MLflow integration with local fallback |
| **`shared.pipeline`** | 🔗 **Pipelines** - Reproducible, chainable data transformations |

## 🤖 AI-Powered Features

### Quick Automated EDA
```python
from shared.auto_eda import quick_eda
report = quick_eda(df, target_col='target')  # Full analysis in one line!
```

### AutoML Training
```python
from shared.automl import quick_automl
preds, runner = quick_automl(X_train, y_train, X_test, task='classification')
print(runner.get_leaderboard())
```

### Experiment Tracking
```python
from shared.tracking import ExperimentTracker
tracker = ExperimentTracker("my-competition")
tracker.start_run("lgb-v1").log_params(params).log_metrics({'auc': 0.85}).end_run()
```

### Reproducible Pipelines
```python
from shared.pipeline import create_preprocessing_pipeline
pipeline = create_preprocessing_pipeline(drop_cols=['id'], fill_strategy='median')
processed_df = pipeline.run(df)
pipeline.save('pipeline_config.json')
```

## 📊 Supported Problem Types

- Binary/Multi-class Classification
- Regression
- Time Series Forecasting
- NLP/Text Classification
- Computer Vision
- Tabular Data

---
*Happy Kaggling! 🎯*
