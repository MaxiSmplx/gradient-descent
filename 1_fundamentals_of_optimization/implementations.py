from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

def plot_function(f: callable, x_max: int):
    x_values = np.linspace(-x_max, x_max, 200)
    y_values = f(x_values)

    plt.plot(x_values, y_values, label=f"{f.__doc__}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_function_and_gd(f: callable, x_history: list[float], x_max: int):
    x_values = np.linspace(-x_max, x_max, 200)
    y_values = f(x_values)

    y_updates = [f(x) for x in x_history]

    # function
    plt.plot(x_values, y_values, label=f"{f.__doc__}")

    # gradient descent points
    plt.scatter(x_history, y_updates, label=f"Gradient Descent \n >iters={len(x_history)-1} \n >x̂={x_history[-1]:.3f}", color="red")

    # connect points
    plt.plot(x_history, y_updates, linestyle="--")


    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


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


def gradient_descent_history_sc(df: callable, x_init: float = 0, learning_rate: float = 0.01, max_iter: float = 100, tol: float = 1e-6, sc: Literal["parameter", "gradient"] = "parameter"):
    x_history = [x_init]

    for _ in range(max_iter):
        gradient = df(x_history[-1])
        x_history.append(x_history[-1] - (learning_rate * gradient))

        if sc == "parameter":
            if abs(x_history[-1] - x_history[-2]) < tol:
                break
        else:
            if np.abs(gradient) < tol:
                break
        
    
    return x_history

