import numpy as np
import pandas as pd
from typing import Literal

class Optimizer():
    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        raise NotImplementedError
    

class Vanilla(Optimizer):
    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

        return w, b

class Momentum(Optimizer):
    def __init__(self, beta: float = 0.9):
        self.v_b = 0.0
        self.v_w = None
        self.beta = beta

    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        if self.v_w is None:
            self.v_w = np.zeros(len(w))

        self.v_w = self.beta * self.v_w + (1-self.beta) * w_gradient
        self.v_b = self.beta * self.v_b + (1-self.beta) * b_gradient

        w -= learning_rate * self.v_w
        b -= learning_rate * self.v_b

        return w, b

class AdaGrad(Optimizer):
    def __init__(self):
        self.eps = 1e-8
        self.G_w = None
        self.G_b = 0.0

    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        if self.G_w is None:
            self.G_w = np.zeros(len(w))
        
        self.G_w += np.power(w_gradient, 2)
        self.G_b += np.power(b_gradient, 2)

        lr_w = learning_rate / (np.sqrt(self.G_w) + self.eps)
        lr_b = learning_rate / (np.sqrt(self.G_b) + self.eps)

        w -= lr_w * w_gradient
        b -= lr_b * b_gradient

        return w, b
    
class RMSProp(Optimizer):
    def __init__(self, beta: float = 0.9):
        self.eps = 1e-8
        self.v_b = 0.0
        self.v_w = None
        self.beta = beta
    
    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        if self.v_w is None:
            self.v_w = np.zeros(len(w))
        
        self.v_w = self.beta * self.v_w + (1-self.beta) * np.power(w_gradient, 2)
        self.v_b = self.beta * self.v_b + (1-self.beta) * np.power(b_gradient, 2)

        lr_w = learning_rate / (np.sqrt(self.v_w) + self.eps)
        lr_b = learning_rate / (np.sqrt(self.v_b) + self.eps)

        w -= lr_w * w_gradient
        b -= lr_b * b_gradient

        return w, b


class Adam(Optimizer):
    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.999):
        self.eps = 1e-8
        self.m_b = 0.0
        self.m_w = None
        self.v_b = 0.0
        self.v_w = None
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 1
    
    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        if self.m_w is None:
            self.m_w = np.zeros(len(w))
        
        self.m_w = self.beta_1 * self.m_w + (1 - self.beta_1) * w_gradient
        self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * b_gradient

        if self.v_w is None:
            self.v_w = np.zeros(len(w))
        
        self.v_w = self.beta_2 * self.v_w + (1 - self.beta_2) * np.power(w_gradient, 2)
        self.v_b = self.beta_2 * self.v_b + (1 - self.beta_2) * np.power(b_gradient, 2)

        m_w_hat = self.m_w / (1 - np.power(self.beta_1, self.t))
        m_b_hat = self.m_b / (1 - np.power(self.beta_1, self.t))

        v_w_hat = self.v_w / (1 - np.power(self.beta_2, self.t))
        v_b_hat = self.v_b / (1 - np.power(self.beta_2, self.t))

        w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
        b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        self.t += 1

        return w, b


class AdamW(Optimizer):
    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.999, weight_decay: float = 0.01):
        self.eps = 1e-8
        self.m_b = 0.0
        self.m_w = None
        self.v_b = 0.0
        self.v_w = None
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.t = 1
    
    def _update_parameters(self, w, b, w_gradient, b_gradient, learning_rate):
        if self.m_w is None:
            self.m_w = np.zeros(len(w))
        
        self.m_w = self.beta_1 * self.m_w + (1 - self.beta_1) * w_gradient
        self.m_b = self.beta_1 * self.m_b + (1 - self.beta_1) * b_gradient

        if self.v_w is None:
            self.v_w = np.zeros(len(w))
        
        self.v_w = self.beta_2 * self.v_w + (1 - self.beta_2) * np.power(w_gradient, 2)
        self.v_b = self.beta_2 * self.v_b + (1 - self.beta_2) * np.power(b_gradient, 2)

        m_w_hat = self.m_w / (1 - np.power(self.beta_1, self.t))
        m_b_hat = self.m_b / (1 - np.power(self.beta_1, self.t))

        v_w_hat = self.v_w / (1 - np.power(self.beta_2, self.t))
        v_b_hat = self.v_b / (1 - np.power(self.beta_2, self.t))

        w = w - learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.eps) - learning_rate * self.weight_decay * w
        b = b - learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.eps) - learning_rate * self.weight_decay * b

        self.t += 1

        return w, b



class GradientDescent():
    def __init__(self):
        self.X = None
        self.y = None
        self.n = None
        self.m = None
        self.w = None
        self.b = None
        self.loss = []
        self.epochs_early_stop = 5
    
    def _loss(self):
        raise NotImplementedError

    def _w_gradient(self, X_batch, e):
        raise NotImplementedError

    def _b_gradient(self, e):
        raise NotImplementedError

    def _predict(self, X_batch):
        raise NotImplementedError

    def _optimize(self, 
                  X: pd.DataFrame, 
                  y: pd.Series, 
                  optimizer: Optimizer,
                  learning_rate: float, 
                  epochs: int, 
                  batch_size: int,
                  tol: float
        ) -> tuple[np.ndarray, float]:
        self.n, self.m = X.shape
        self.loss = []

        self.X, self.y = X.to_numpy(), y.to_numpy()
        self.w, self.b = np.zeros(self.m), np.float64(0.0)

        batch_iters = np.ceil(self.n / batch_size).astype(int)

        for epoch in range(epochs):
            indices = np.random.permutation(self.n)

            for i in range(batch_iters):
                batch_indices = indices[i*batch_size:i*batch_size+batch_size]
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]

                y_hat = self._predict(X_batch)
                e = y_hat - y_batch

                w_gradient = self._w_gradient(X_batch, e)
                b_gradient = self._b_gradient(e)

                self.w, self.b = optimizer._update_parameters(self.w, self.b, w_gradient, b_gradient, learning_rate)

            self._loss()

            if epoch > self.epochs_early_stop:
                if np.abs(self._loss[-self.epochs_early_stop] - self._loss[-1]) < tol:
                    break

        return self.w, self.b

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            optimizer: Literal["vanilla", "momentum", "adagrad", "rmsprop", "adam", "adamw"] = "adamw", 
            learning_rate: float = 0.001, 
            epochs: int = 1_000, 
            batch_size: int = 32,
            tol: float = 1e-4,
        ) -> None:
        assert type(X) == pd.DataFrame
        assert type(y) == pd.Series

        batch_size = min(X.shape[0], batch_size)

        if optimizer == "vanilla":
            self.w, self.b = self._optimize(X, y, optimizer=Vanilla(), learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tol=tol)
        
        elif optimizer == "momentum":
            self.w, self.b = self._optimize(X, y, optimizer=Momentum(beta=0.9), learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tol=tol)
        
        elif optimizer == "adagrad":
            self.w, self.b = self._optimize(X, y, optimizer=AdaGrad(), learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tol=tol)
        
        elif optimizer == "rmsprop":
            self.w, self.b = self._optimize(X, y, optimizer=RMSProp(beta=0.9), learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tol=tol)
        
        elif optimizer == "adam":
            self.w, self.b = self._optimize(X, y, optimizer=Adam(beta_1=0.9, beta_2=0.999), learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tol=tol)

        elif optimizer == "adamw":
            self.w, self.b = self._optimize(X, y, optimizer=AdamW(beta_1=0.9, beta_2=0.999, weight_decay=0.01), learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, tol=tol)
        
        else:
            raise ValueError(f"Optimizer {optimizer} does not exist. Choose: 'vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam' or 'adamw'.")
    
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
    def _loss(self):
        y_hat = self.X @ self.w + self.b
        self.loss.append(np.mean(np.power(self.y - y_hat, 2)))

    def _w_gradient(self, X_batch, e) -> np.ndarray:
        return (2/len(X_batch)) * X_batch.transpose() @ e
    
    def _b_gradient(self, e) -> float:
        return 2 * np.mean(e)

    def _predict(self, X_batch: np.ndarray) -> np.ndarray:
        return X_batch @ self.w + self.b
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert type(X) == pd.DataFrame

        return X @ self.w + self.b


class LogisticRegression(GradientDescent):
    def _loss(self):
        boundary = np.finfo(float).eps
        y_hat = np.clip(self._sigmoid(self.X @ self.w + self.b), a_min=boundary, a_max=1-boundary)
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
    
    def _predict_proba(self, X_batch) -> np.ndarray:
        return self._sigmoid(X_batch @ self.w + self.b)
    
    def _predict(self, X_batch) -> np.ndarray:
        return self._predict_proba(X_batch)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: pd.DataFrame, thres: float) -> np.ndarray:
        assert type(X) == pd.DataFrame

        return (self.predict_proba(X) >= thres).astype(int)