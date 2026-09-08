import numpy as np
import pandas as pd

def gradient_descent(X, y, learning_rate: float, max_iters: int, tol: float):
    X = X.to_numpy()
    y = y.to_numpy()
    n_samples, n_features = X.shape

    w, b = np.zeros(n_features), 0

    for _ in range(max_iters):
        y_pred = X @ w + b

        residual = y - y_pred

        gradient_b = -2 * np.mean(residual)
        gradient_w = -2 * (X.T @ residual) / n_samples

        b -= learning_rate * gradient_b
        w -= learning_rate * gradient_w

        if np.abs(gradient_b) < tol and np.linalg.norm(gradient_w) < tol:
            break
    
    return [b, w]

def linear_regression(X: pd.DataFrame, 
                      y: pd.Series, 
                      learning_rate: float = 0.01, 
                      max_iters: int = 1_000, 
                      tol: float = 1e-6
) -> dict[str: float]:
    assert type(X) == pd.DataFrame
    assert type(y) == pd.Series

    b, w = gradient_descent(X, y, learning_rate=learning_rate, max_iters=max_iters, tol=tol)

    return {"b": b, "w_i": w}
