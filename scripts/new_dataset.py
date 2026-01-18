#!/usr/bin/env python
"""
Create a new dataset analysis project from template.

Usage:
    python scripts/new_dataset.py "housing-prices"
"""

import argparse
import shutil
from pathlib import Path
import re


def slugify(name: str) -> str:
    """Convert dataset name to slug."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def create_dataset(name: str) -> None:
    """Create a new dataset analysis project."""
    
    project_root = Path(__file__).parent.parent
    template_path = project_root / "datasets" / "_template"
    
    slug = slugify(name)
    target_path = project_root / "datasets" / slug
    
    if target_path.exists():
        print(f"❌ Dataset project '{slug}' already exists!")
        return
    
    # Create from template if exists, otherwise create basic structure
    if template_path.exists():
        shutil.copytree(template_path, target_path)
    else:
        target_path.mkdir(parents=True)
    
    # Create directories
    (target_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (target_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (target_path / "notebooks").mkdir(exist_ok=True)
    (target_path / "outputs").mkdir(exist_ok=True)
    
    # Create README
    readme_content = f"""# {name}

> Dataset Analysis Project

## 📁 Structure

```
{slug}/
├── data/
│   ├── raw/           # Original data
│   └── processed/     # Cleaned data
├── notebooks/
│   ├── 01_eda.ipynb   # Exploratory Data Analysis
│   └── 02_analysis.ipynb
└── outputs/           # Figures, reports
```

## 📊 Data Source

- Source: [Add source here]
- Downloaded: [Add date]

## 💡 Key Insights

1. ...
2. ...
3. ...
"""
    (target_path / "README.md").write_text(readme_content)
    
    print(f"""
🎉 Dataset project '{name}' created successfully!

📁 Location: {target_path}

📋 Next steps:
   1. Add data to datasets/{slug}/data/raw/
   2. Create analysis notebooks in datasets/{slug}/notebooks/
""")


def main():
    parser = argparse.ArgumentParser(description="Create a new dataset analysis project")
    parser.add_argument("name", help="Dataset name")
    
    args = parser.parse_args()
    create_dataset(args.name)


if __name__ == "__main__":
    main()
