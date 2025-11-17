"""
Utility functions for the healthcare fraud detection project
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def ensure_dir(path: str) -> None:
    """Ensure a directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[Any, Any], filepath: str) -> None:
    """Save data to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[Any, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_data_path(filename: str, subfolder: str = 'raw') -> str:
    """Get path to data file"""
    root = get_project_root()
    return str(root / 'data' / subfolder / filename)


def get_output_path(filename: str, subfolder: str = 'reports') -> str:
    """Get path to output file"""
    root = get_project_root()
    return str(root / 'outputs' / subfolder / filename)


def get_model_path(filename: str) -> str:
    """Get path to model file"""
    root = get_project_root()
    return str(root / 'models' / filename)


def load_env_variables(env_file: str = '.env') -> Dict[str, str]:
    """Load environment variables from .env file"""
    from dotenv import load_dotenv
    load_dotenv(env_file)
    return dict(os.environ)


def print_separator(char: str = '=', length: int = 60) -> None:
    """Print a separator line"""
    print(char * length)


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate basic statistics for a column"""
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q25': df[column].quantile(0.25),
        'q75': df[column].quantile(0.75),
        'q99': df[column].quantile(0.99)
    }

