# {{ COMPETITION_NAME }}

> Kaggle Competition: [{{ COMPETITION_NAME }}](https://www.kaggle.com/c/{{ competition_slug }})

## 📋 Competition Info

- **Type**: {{ competition_type }} (classification/regression/etc.)
- **Metric**: {{ evaluation_metric }}
- **Deadline**: {{ deadline }}

## 🎯 Goal

{{ description }}

## 📁 Structure

```
{{ competition_name }}/
├── data/
│   ├── raw/           # Original competition data
│   └── processed/     # Cleaned/engineered features
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb      # Model training & validation
│   └── 04_submission.ipynb    # Final submission generation
├── src/
│   ├── features.py    # Feature engineering functions
│   ├── models.py      # Competition-specific model code
│   └── config.py      # Competition configuration
├── submissions/       # Submission files
└── README.md          # This file
```

## 🚀 Quick Start

```bash
# 1. Download data (requires Kaggle API setup)
kaggle competitions download -c {{ competition_slug }} -p data/raw

# 2. Unzip data
unzip data/raw/*.zip -d data/raw/

# 3. Run notebooks in order
```

## 📊 Current Results

| Model | CV Score | LB Score | Notes |
|-------|----------|----------|-------|
| Baseline | 0.XXX | 0.XXX | Simple model |

## 📝 Notes

- ...
