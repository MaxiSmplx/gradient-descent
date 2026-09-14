import numpy as np
import pandas as pd

        
class LogisticRegressionGD():
    def __init__(self):
        self.w = None
        self.b = None
        self.X = None
        self.y = None

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
    
    def log_loss(self, y, y_pred_proba) -> float:
        boundary = np.finfo(float).eps
        y_pred_proba = np.clip(y_pred_proba, a_min=boundary, a_max=1-boundary)
        return -np.mean(y*np.log(y_pred_proba) + (1-y)*np.log(1-y_pred_proba))

    def gradient_descent(self, X, y, learning_rate, max_iter, tol, verbose):
        n_samples, m_features = X.shape

        w, b = np.zeros(m_features), np.float64(0.0)
        X, y = X.to_numpy(), y.to_numpy()

        self.X, self.y = X, y

        for i in range(max_iter):
            if verbose and (i+1) % (max_iter//10) == 0:
                print(f"Iteration {i+1}")
            y_hat = self.sigmoid(X @ w + b)

            # transform w_gradient from normal array into vector of form R^(m x 1)
            w_gradient = np.mean(X * (y_hat - y)[:, np.newaxis], axis=0)
            b_gradient = np.mean(y_hat - y)

            w -= learning_rate * w_gradient
            b -= learning_rate * b_gradient

            if np.linalg.norm(w_gradient, ord=2) < tol and np.linalg.norm(np.array([b_gradient]), ord=2) < tol:
                break

        return w, b

    def fit(self, X: pd.DataFrame, y: pd.Series, learning_rate: float = 0.001, max_iter: int = 10_000, tol: float = 1e-6, verbose: bool = False) -> None:
        assert type(X) == pd.DataFrame
        assert type(y) == pd.Series

        assert X.select_dtypes(include=np.number).shape[1] == X.shape[1]
        assert np.issubdtype(y.dtype, np.number)

        w, b = self.gradient_descent(X, y, learning_rate, max_iter, tol, verbose)

        self.w = w
        self.b = b
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X: pd.DataFrame, thres: float = 0.5) -> np.array:
        assert thres <= 1 and thres >= 0

        y_pred = self.predict_proba(X)
        y_pred_thresholded = (y_pred >= thres).astype(int)

        return y_pred_thresholded
    
    @property
    def coefs_(self) -> np.array:
        return self.w

    @property
    def intercept_(self) -> float:
        return self.b
    
    @property
    def loss_(self) -> float:
        return self.log_loss(self.y, self.predict_proba(self.X))

    @property
    def decision_boundary(self):
        if len(self.w) == 1:
            return -self.b / self.w
        elif len(self.w) > 1:
            return self.w, self.b