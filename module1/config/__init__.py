"""
Configuration utilities for Module 1.
"""

import os
import yaml
from typing import Dict, Any

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "preprocessing_config.yaml")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
