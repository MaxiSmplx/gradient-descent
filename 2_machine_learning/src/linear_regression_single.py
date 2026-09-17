import numpy as np
from typing import Literal
import pandas as pd

def dmse_single(X: pd.Series, y: pd.Series, w: float, b: float, param: Literal["w", "b"]) -> float:
    if param == "w":
        return -2 * np.mean((y - w * X - b) * X)
    elif param == "b":
        return -2 * np.mean(y - w * X - b)

def gradient_descent_single(X: pd.Series, y: pd.Series, learning_rate: float, max_iters: int, tol: float) -> list:
    b_vals, w_vals = [0], [0]

    for _ in range(max_iters):
        b_gradient = dmse_single(X, y, w_vals[-1], b_vals[-1], "b")
        w_gradient = dmse_single(X, y, w_vals[-1], b_vals[-1], "w")

        b_vals.append(b_vals[-1] - learning_rate * b_gradient)
        w_vals.append(w_vals[-1] - learning_rate * w_gradient)

        if np.linalg.norm(b_gradient) < tol and np.linalg.norm(w_gradient) < tol:
            break
    
    return [w_vals, b_vals]
    

def linear_regression_single(X: pd.Series, 
                             y: pd.Series, 
                             learning_rate: float = 0.01, 
                             max_iters: int = 500, 
                             tol: float = 1e-6, 
                             history: bool = False
) -> tuple:
    assert type(X) == pd.Series
    assert type(y) == pd.Series

    assert np.issubdtype(X.dtype, np.number)
    assert np.issubdtype(y.dtype, np.number)

    w, b = gradient_descent_single(X, y, learning_rate, max_iters, tol)

    return (w[-1], b[-1]) if not history else (w, b)

 