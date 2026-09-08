import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.scatter(X, y, label="Data", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



def plot_optimal_parameters(w_vals, b_vals, optimal_w: float = None, optimal_b: float = None):
    plt.figure(figsize=(8, 5))

    plt.plot(w_vals, label="w", color="blue")
    plt.plot(b_vals, label="b", color="orange")

    if optimal_w is not None:
        plt.axhline(optimal_w, linestyle="--", label=f"optimal w = {optimal_w}", color="blue")
    if optimal_b is not None:
        plt.axhline(optimal_b, linestyle="--", label=f"optimal b = {optimal_b}", color="orange")

    plt.xlabel("Iteration")
    plt.ylabel("Parameter value")
    plt.title("Gradient Descent Convergence")
    plt.legend()
    plt.show()


def plot_gradient_progression_lr(X, y, w_vals, b_vals):
    n_iter = len(w_vals)

    n_early = max(4, int(n_iter * 0.01))
    early_iters = np.random.choice(n_early, size=4, replace=False)

    step = max(1, n_iter // 4)
    regular_iters = range(n_early, n_iter, step)
    plot_iters = sorted(set(early_iters).union(set(regular_iters)))

    plt.scatter(X, y, label="Data", color="red")

    x_line = np.linspace(X.min(), X.max(), 200)

    for i in plot_iters:
        w, b = w_vals[i], b_vals[i]

        lr = w * x_line + b
        plt.plot(x_line, lr, label=f"y = {w:.2f}x + {b:.2f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()