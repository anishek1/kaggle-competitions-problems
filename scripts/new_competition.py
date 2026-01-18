#!/usr/bin/env python
"""
Create a new Kaggle competition project from template.

Usage:
    python scripts/new_competition.py "titanic" --type classification
"""

import argparse
import shutil
from pathlib import Path
import re


def slugify(name: str) -> str:
    """Convert competition name to slug."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def create_competition(name: str, task_type: str = "classification") -> None:
    """Create a new competition project from template."""
    
    project_root = Path(__file__).parent.parent
    template_path = project_root / "competitions" / "_template"
    
    slug = slugify(name)
    target_path = project_root / "competitions" / slug
    
    if target_path.exists():
        print(f"❌ Competition '{slug}' already exists!")
        return
    
    # Copy template
    shutil.copytree(template_path, target_path)
    print(f"✅ Created competition folder: {target_path}")
    
    # Create data directories
    (target_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (target_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (target_path / "models").mkdir(exist_ok=True)
    (target_path / "submissions").mkdir(exist_ok=True)
    
    # Update README
    readme_path = target_path / "README.md"
    readme_content = readme_path.read_text()
    readme_content = readme_content.replace("{{ COMPETITION_NAME }}", name)
    readme_content = readme_content.replace("{{ competition_slug }}", slug)
    readme_content = readme_content.replace("{{ competition_name }}", slug)
    readme_content = readme_content.replace("{{ competition_type }}", task_type)
    readme_path.write_text(readme_content)
    
    # Update config
    config_path = target_path / "src" / "config.py"
    if config_path.exists():
        config_content = config_path.read_text()
        config_content = config_content.replace('COMPETITION_SLUG = "competition-name"', f'COMPETITION_SLUG = "{slug}"')
        config_content = config_content.replace('TASK = "classification"', f'TASK = "{task_type}"')
        config_path.write_text(config_content)
    
    print(f"""
🎉 Competition '{name}' created successfully!

📁 Location: {target_path}

📋 Next steps:
   1. Download data: kaggle competitions download -c {slug} -p competitions/{slug}/data/raw
   2. Unzip data: unzip competitions/{slug}/data/raw/*.zip -d competitions/{slug}/data/raw/
   3. Start with notebooks/01_eda.ipynb
""")


def main():
    parser = argparse.ArgumentParser(description="Create a new Kaggle competition project")
    parser.add_argument("name", help="Competition name")
    parser.add_argument("--type", dest="task_type", default="classification",
                        choices=["classification", "regression", "timeseries", "nlp", "cv"],
                        help="Task type (default: classification)")
    
    args = parser.parse_args()
    create_competition(args.name, args.task_type)


if __name__ == "__main__":
    main()
