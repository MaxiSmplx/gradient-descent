# Gradient Descent from First Principles

## Exploration of the fundamentals, mathematical derivations and variants of Gradient Descent

## Summary

Gradient Descent is one of the fundamental optimization algorithms underlying modern machine learning. This project explores Gradient Descent from first principles, progressing from mathematical optimization to its application in machine learning and finally to commonly used optimization variants.

Gradient Descent is formally expressed as

$$w_{n+1} = w_n - \eta \nabla L_w$$

where $w_n$ is the current parameter value, $\eta$ is a constant learning rate and $\nabla L_w$ is the derived loss function with respect to $w$: 

$$\nabla L_w = \frac{\partial L}{\partial w}$$ 

This update rule states that the parameter $w$ is iteratively updated by subtracting the gradient of the loss function scaled by a learning rate $\eta$. 

Since the gradient points in the direction of steepest increase, moving in the opposite direction reduces the loss and guides the optimization process toward a local minimum. The parameter values at this minimum correspond to those that minimize the discrepancy between the observed target values and the model predictions.

<img width="562" height="432" alt="gd1" src="https://github.com/user-attachments/assets/22aea089-2304-49e5-bfc8-501d07e3162f" />


The project covers:

* The mathematical foundations of Gradient Descent
* Optimization of simple functions and analysis of convergence
* Linear and Logistic Regression implemented from first principles
* Batch Gradient Descent, Stochastic Gradient Descent and Mini-Batch Gradient Descent
* Momentum, AdaGrad, RMSProp, Adam and AdamW
* The mathematical motivation behind each optimization method
* Learning rates, convergence, divergence and stopping criteria
* Practical experiments and visualizations of optimization behavior

The goal is not only to implement Gradient Descent, but to understand the mathematics behind it and how its different variants address limitations of the basic algorithm.

---

## Fundamentals of Optimization

The project begins with Gradient Descent itself before introducing machine learning models.

Simple quadratic functions are used to visualize the optimization process and investigate how the algorithm approaches an optimum. This provides an intuitive foundation for understanding gradients, parameter updates and convergence.

<img width="755" height="455" alt="gd_init" src="https://github.com/user-attachments/assets/86153bc4-e9f6-4bfa-bed3-d47db8322b25" />


The analysis includes:

* Gradient Descent on simple quadratic functions
* Local and global minima in non-convex functions
* The influence of the initial parameter values $x_{init}$
* Learning rate $\eta$
* Oscillation and divergence for excessively large learning rates
* Convergence and stopping criteria
* Parameter-change tolerance
* Loss-change tolerance
* Gradient norm using the Euclidean ($L_2$) norm

These experiments establish the relationship between the mathematical properties of an optimization problem and the behavior of Gradient Descent.

---

## Machine Learning

After establishing the fundamentals of optimization, Gradient Descent is applied to supervised machine learning models.

### Linear Regression

Linear Regression is implemented using Batch Gradient Descent from first principles with NumPy.

The implementation progresses from the one-dimensional case to the general matrix formulation, introducing feature matrices, parameter vectors and vectorized predictions.

<img width="563" height="432" alt="linear_regression" src="https://github.com/user-attachments/assets/3c48ab47-b5b1-4c5f-bbd4-9096f49ad98f" />

The project covers:

* Linear regression from one-dimensional vectors to general matrices $X \in \R^{n \times m}$
* Parameter vectors and vectorized predictions
* Step-by-step derivation of the MSE gradient
* Parameter optimization and convergence
* Learning-rate-dependent divergence
* Hessian matrices and eigenvalues as a theoretical basis for learning-rate constraints
* Z-standardization to improve optimization behavior

The implementation is deliberately built without relying on a machine learning framework in order to make the relationship between the mathematical derivation and the resulting code explicit.

### Logistic Regression

Logistic Regression is subsequently implemented using Batch Gradient Descent from first principles.

The project introduces binary classification through the sigmoid function and linear combinations of input features before deriving the corresponding loss function.

<img width="556" height="432" alt="logistic_regression_1" src="https://github.com/user-attachments/assets/96e9df58-5826-4ee5-8c32-d38b9f14acb0" />


The analysis covers:

* Supervised binary classification
* The sigmoid function
* Binary Cross-Entropy (log-loss)
* Step-by-step derivation of the gradient
* The relationship between the decision boundary, slope and model coefficients
* Model predictions on a real dataset
* Evaluation using accuracy, precision, recall and log-loss
* Interpretation of learned model coefficients

<img width="567" height="433" alt="logistic_regression_2" src="https://github.com/user-attachments/assets/d43af205-9fb6-4b6d-b7e3-51b81b332771" />


The Titanic dataset is used as a practical example for binary survival prediction.

---

## Variants of Gradient Descent

The final part of the project examines different ways of estimating gradients and updating model parameters.

The first step is the transition from Batch Gradient Descent to Stochastic and Mini-Batch Gradient Descent. Different batch sizes are introduced:

* Batch Gradient Descent: $k = n$
* Stochastic Gradient Descent: $k = 1$
* Mini-Batch Gradient Descent: $1 < k < n$

<img width="572" height="432" alt="Batch" src="https://github.com/user-attachments/assets/beacbab2-e093-47e7-8e16-a5b3acb592ee" />


Their respective effects on gradient estimates, computational cost, memory usage, update frequency and optimization noise are investigated.

The project then introduces increasingly sophisticated optimization algorithms:

* Momentum
* AdaGrad
* RMSProp
* Adam
* AdamW

Each optimizer is implemented from first principles and integrated into the existing Gradient Descent framework.

Rather than treating these algorithms as independent methods, the project explores how they build upon each other to address limitations of previous approaches. The mathematical update rules are derived and their effects on optimization are examined experimentally.

This includes:

* Momentum and exponentially weighted gradient combinations
* Adaptive learning rates
* Accumulation and decay of squared gradients
* First- and second-moment estimates
* Bias correction
* Decoupled weight decay in AdamW
* Visualization of momentum and parameter updates
* Comparison of loss trajectories across optimizers

<img width="576" height="432" alt="optimizers" src="https://github.com/user-attachments/assets/7d894d87-7de0-4b92-9fcc-b68cb083d8c0" />

---

## Project Goal

The overall goal of this project is to develop a complete understanding of Gradient Descent, from its mathematical formulation to its application in machine learning and its modern optimization variants.

The project therefore progresses through three levels:

```text
Mathematical Optimization
        ↓
Machine Learning
        ↓
Gradient Descent Variants
```

Starting with simple functions makes the underlying optimization process directly observable. Linear and Logistic Regression then demonstrate how the same mathematical principles are used to train machine learning models. Finally, Momentum and adaptive optimizers show how the basic algorithm can be extended to improve its optimization behavior.

By implementing the methods from first principles and deriving the relevant mathematics alongside the code, the project connects the theoretical formulation of Gradient Descent with its practical use in machine learning.
