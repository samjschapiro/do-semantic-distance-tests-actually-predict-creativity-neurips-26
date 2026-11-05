"""Small shared helpers: repository paths, config loading, output directories."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Embedding files are not shipped (see README). Put them in resources/ or point
# EMBEDDINGS_DIR at the folder that holds them.
EMBEDDINGS_DIR = Path(os.environ.get("EMBEDDINGS_DIR", ROOT / "resources"))
GLOVE_PATH = EMBEDDINGS_DIR / "glove.840B.300d.txt"
FASTTEXT_PATH = EMBEDDINGS_DIR / "crawl-300d-2M.vec"


def init_directory(output_dir: str | Path) -> Path:
    """Create a fresh output directory. Refuses to reuse an existing one, so a
    re-run never deletes earlier results: pick a new output_dir instead."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Output directory {output_dir} already exists. "
                         "Choose a new output_dir in the config.")
    output_dir.mkdir(parents=True)
    return output_dir


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "output_dir" not in config:
        raise ValueError("'output_dir' is required in the config")
    return config


def save_config(config: dict, output_dir: str | Path, filename: str = "config.yaml") -> Path:
    path = Path(output_dir) / filename
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return path
