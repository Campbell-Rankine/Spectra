import numpy as np

def exp_transform(_a: np.ndarray, base: int) -> np.ndarray:
    return _a ** base

def sqrt_transform(_a: np.ndarray, **kw) -> np.ndarray:
    return np.sqrt(_a, **kw)