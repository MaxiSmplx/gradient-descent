import numpy as np
import matplotlib.pyplot as plt
from typing import Literal


def gradient_descent(df: callable, x_init: float = 0, learning_rate: float = 0.01, max_iter: float = 30) -> float:
    x = x_init

    for _ in range(max_iter):
        gradient = df(x)
        x -= learning_rate * gradient
    
    return x


def gradient_descent_history(df: callable, x_init: float = 0, learning_rate: float = 0.01, max_iter: float = 30) -> list[float]:
    x_history = [x_init]

    for _ in range(max_iter):
        gradient = df(x_history[-1])
        x_history.append(x_history[-1] - (learning_rate * gradient))
    
    return x_history


def gradient_descent_history_sc(df: callable, 
                                x_init: float = 0, 
                                learning_rate: float = 0.01, 
                                max_iter: float = 100, 
                                tol: float = 1e-6, 
                                sc: Literal["parameter", "gradient"] = "parameter"
) -> list[float]:
    x_history = [x_init]

    for _ in range(max_iter):
        gradient = df(x_history[-1])
        x_history.append(x_history[-1] - (learning_rate * gradient))

        if sc == "parameter" and abs(x_history[-1] - x_history[-2]) < tol:
            break
        elif sc == "gradient" and np.sqrt(np.abs(gradient)**2) < tol:
            break
        
    
    return x_history

