import numpy as np
import pandas as pd
from typing import Literal

class GradientDescent():
    def __init__(self):
        self.X = None
        self.y = None
        self.n = None
        self.m = None
        self.w = None
        self.b = None
        self.loss = []
        self.w_hist = []
        self.b_hist = []
        self.epochs_early_stop = 5
    
    def _loss(self, w, b):
        raise NotImplementedError

    def _w_gradient(self, X_batch, e):
        raise NotImplementedError

    def _b_gradient(self, e):
        raise NotImplementedError

    def _predict(self, X_batch, w, b):
        raise NotImplementedError

    def _optimize(self, 
                  X: pd.DataFrame, 
                  y: pd.Series, 
                  learning_rate: float, 
                  epochs: int, 
                  batch_size: int,
                  tol: float
        ) -> tuple[np.ndarray, float]:
        self.n, self.m = X.shape
        self.loss = []
        self.w_hist, self.b_hist = [], []

        self.X, self.y = X.to_numpy(), y.to_numpy()
        w, b = np.zeros(self.m), np.float64(0.0)

        batch_iters = np.ceil(self.n / batch_size).astype(int)

        for epoch in range(epochs):
            indices = np.random.permutation(self.n)

            for i in range(batch_iters):
                batch_indices = indices[i*batch_size:i*batch_size+batch_size]
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]

                y_hat = self._predict(X_batch, w, b)
                e = y_hat - y_batch

                w_gradient = self._w_gradient(X_batch, e)
                b_gradient = self._b_gradient(e)

                w -= learning_rate * w_gradient
                b -= learning_rate * b_gradient

                self._loss(w, b)
                self.b_hist.append(b)
                self.w_hist.append(w.copy())

            if epoch > self.epochs_early_stop:
                if np.abs(self._loss[-self.epochs_early_stop] - self._loss[-1]) < tol:
                    break
            
        return w, b

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            batch_method: Literal["batch", "stochastic", "mini_batch"] = "batch", 
            learning_rate: float = 0.001, 
            epochs: int = 1_000, 
            batch_size: int = 32,
            tol: float = 1e-4,
        ) -> None:
        assert type(X) == pd.DataFrame
        assert type(y) == pd.Series

        if batch_method == "batch":
            self.w, self.b = self._optimize(X, y, learning_rate=learning_rate, epochs=epochs, batch_size=X.shape[0], tol=tol)
        
        elif batch_method == "stochastic":
            self.w, self.b = self._optimize(X, y, learning_rate=learning_rate, epochs=epochs, batch_size=1, tol=tol)

        elif batch_method == "mini_batch":
            self.w, self.b = self._optimize(X, y, learning_rate=learning_rate, epochs=epochs, batch_size=min(X.shape[0], batch_size), tol=tol)
        
        else:
            raise ValueError(f"Batch method {batch_method} does not exist. Choose: 'batch', 'stochastic', 'mini_batch'.")
    
    @property
    def coefs_(self) -> np.ndarray:
        return self.w
    
    @property
    def intercept_(self) -> float:
        return self.b
    
    @property
    def losses_(self) -> list[float]:
        return self.loss
    

class LinearRegression(GradientDescent):
    def _loss(self, w, b) -> None:
        y_hat = self.X @ w + b
        self.loss.append(np.mean(np.power(self.y - y_hat, 2)))

    def _w_gradient(self, X_batch, e) -> np.ndarray:
        return (2/len(X_batch)) * X_batch.transpose() @ e
    
    def _b_gradient(self, e) -> float:
        return 2 * np.mean(e)

    def _predict(self, X_batch: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        return X_batch @ w + b
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert type(X) == pd.DataFrame

        return X @ self.w + self.b


class LogisticRegression(GradientDescent):
    def _loss(self, w, b) -> None:
        boundary = np.finfo(float).eps
        y_hat = np.clip(self._sigmoid(self.X @ w + b), a_min=boundary, a_max=1-boundary)
        self.loss.append(-np.mean(self.y * np.log(y_hat) + (1-self.y) * np.log(1-y_hat)))

    def _sigmoid(self, t: np.ndarray) -> float:
        sigmoid_res = np.empty_like(t, dtype=float)

        positive = t >= 0
        sigmoid_res[positive] = 1 / (1 + np.exp(-t[positive]))

        negative = t < 0
        sigmoid_res[negative] = np.exp(t[negative]) / (1 + np.exp(t[negative]))

        return sigmoid_res 
    
    def _w_gradient(self, X_batch, e: np.ndarray) -> np.ndarray:
        return np.mean(X_batch * e[:, np.newaxis], axis=0)
    
    def _b_gradient(self, e: np.ndarray) -> float:
        return np.mean(e)
    
    def _predict_proba(self, X_batch, w: np.ndarray, b: float) -> np.ndarray:
        return self._sigmoid(X_batch @ w + b)
    
    def _predict(self, X_batch, w: np.ndarray, b: float) -> np.ndarray:
        return self._predict_proba(X_batch, w, b)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: pd.DataFrame, thres: float) -> np.ndarray:
        assert type(X) == pd.DataFrame

        return (self.predict_proba(X) >= thres).astype(int)