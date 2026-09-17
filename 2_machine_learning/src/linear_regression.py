import numpy as np
import pandas as pd

def max_learning_rate(X: pd.DataFrame) -> float:
    n = X.shape[0]
    X = X.to_numpy()

    lambda_max = np.max(np.linalg.eig(X.transpose() @ X).eigenvalues)

    return n/lambda_max

def gradient_descent(X, y, learning_rate: float, max_iters: int, tol: float):
    n_samples, m_features = X.shape

    X, y  = X.to_numpy(), y.to_numpy()
    w, b = np.zeros(m_features), np.float64(0.0)

    one_vector = np.ones(n_samples)

    for _ in range(max_iters):
        y_hat = X @ w + b

        e = y - y_hat

        w_gradient = (-2/n_samples) * X.transpose() @ e
        b_gradient = (-2/n_samples) * one_vector.transpose() @ e

        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

        if np.linalg.norm(w_gradient, ord=2) < tol and np.linalg.norm(np.array([b_gradient]), ord=2) < tol:
            break

    return w, b

    
def linear_regression(X: pd.DataFrame, 
                      y: pd.Series, 
                      learning_rate: float = 0.01, 
                      max_iters: int = 1_000, 
                      tol: float = 1e-6,
                      divergance_check: bool = False
) -> dict[str: float]:
    assert type(X) == pd.DataFrame
    assert type(y) == pd.Series

    assert X.select_dtypes(include=np.number).shape[1] == X.shape[1]
    assert np.issubdtype(y.dtype, np.number)

    if divergance_check:
        max_lr = max_learning_rate(X)
        assert learning_rate < max_lr, f"Gradient Descent will diverge with learning rate {learning_rate} as it can not be greater or equal to {max_lr:.6f}"

    w, b = gradient_descent(X, y, learning_rate=learning_rate, max_iters=max_iters, tol=tol)

    return b, w
