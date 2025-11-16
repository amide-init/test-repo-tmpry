"""
Helper utility functions for AFN comparison experiments
"""

import numpy as np
import random
from typing import List, Any


def convert_to_json(obj: Any) -> Any:
    """
    Recursively convert numpy types to JSON-serializable Python types
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json(v) for v in obj]
    return obj


def parse_list_arg(arg_str: str) -> List[int]:
    """
    Parse comma-separated string or ranges into list of integers
    
    Args:
        arg_str: Comma-separated string with optional ranges (e.g., "1,2,3" or "1-24")
        
    Returns:
        Sorted list of unique integers
        
    Examples:
        >>> parse_list_arg("1,2,3")
        [1, 2, 3]
        >>> parse_list_arg("1-5")
        [1, 2, 3, 4, 5]
        >>> parse_list_arg("1,3,5-7,10")
        [1, 3, 5, 6, 7, 10]
    """
    result = []
    for part in arg_str.split(','):
        part = part.strip()
        if '-' in part:
            # Handle range (e.g., "1-24")
            start, end = part.split('-', 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(list(set(result)))


def setup_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def extract_history(result: dict) -> list:
    """
    Extract optimization history from result dictionary
    
    Args:
        result: Result dictionary from optimizer
        
    Returns:
        List of function values over iterations
    """
    h = []
    if isinstance(result, dict):
        if 'history' in result and result['history'] is not None:
            h = result['history']
        elif 'y_history' in result and result['y_history'] is not None:
            h = result['y_history']
    
    # Convert to list if needed
    if isinstance(h, np.ndarray):
        h = h.tolist()
    elif not isinstance(h, list):
        h = list(h) if h is not None else []
    
    return h


def generate_seed(func_id: int, dimension: int, run_idx: int, algorithm_name: str, base_seed: int = 42) -> int:
    """
    Generate deterministic seed for reproducibility
    
    Args:
        func_id: Function ID
        dimension: Problem dimension
        run_idx: Run index
        algorithm_name: Name of algorithm
        base_seed: Base random seed
        
    Returns:
        Generated seed value
    """
    return base_seed + func_id * 1000 + dimension * 100 + run_idx + hash(algorithm_name) % 1000

