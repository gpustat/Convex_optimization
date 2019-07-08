import numpy as np


def min_gd(fun, x0, grad, args):
    alpha = 0.3
    beta = 0.8
    a = args[0]
    b = args[1]
    while np.linalg.norm(grad(x0, a, b)) > 1e-6:
        t = 1
        gradient = grad(x0, a, b)
        while fun(x0 - t * gradient, a, b) > fun(x0, a, b) - alpha * t * np.power(np.linalg.norm(gradient), 2):
            t *= beta
        x0 -= t * gradient
    return x0
