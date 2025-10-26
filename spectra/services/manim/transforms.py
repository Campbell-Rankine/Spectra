import numpy as np
from functools import wraps

def exp_transform(_a: np.ndarray, base: int) -> np.ndarray:
    return _a ** base

def sqrt_transform(_a: np.number, eps=1e-9) -> np.ndarray:
    if np.isnan(_a):
        _a = eps
    return np.sqrt(_a)

def safe_log(*args, eps=1e-9, replace_nan=None):
    """
    Decorator to sanitize array inputs by replacing NaNs and zeros
    before calling the wrapped function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(_a: np.ndarray, *args, **kwargs):
            # Replace NaNs with mean or custom value
            fill_value = np.mean(_a) if replace_nan is None else replace_nan
            _a = np.nan_to_num(_a, nan=fill_value)
            
            for idx, x in enumerate(_a):
                if x == 0:
                    _a[idx] = eps
            return func(_a, *args, **kwargs)
        return wrapper
    return decorator