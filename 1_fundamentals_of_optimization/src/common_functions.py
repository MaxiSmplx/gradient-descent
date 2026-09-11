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


def plot_parameter_effects(histories: dict[float: float], 
                           parameter_name:str = "Parameter", 
                           title: str = "Effects of parameter tuning of GD", 
                           global_min: float = None, 
                           exempt: set[float] = []
) -> None:
    for param, x_history in histories.items():
        if param in exempt:
            continue
        plt.plot(
            x_history,
            label=f"{parameter_name} = {param}"
        )

    if global_min:
        plt.axhline(global_min, label=f"minimum x = {global_min}", linewidth=2.5)

    plt.xlabel("Iteration")
    plt.ylabel("x")
    plt.title(f"{title}")
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.grid()
    plt.show()