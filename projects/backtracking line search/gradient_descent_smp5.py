import numpy as np

def min_gd( fun, x0, grad, args=() ):
    
    # initializing
    alpha, beta = 0.3, 0.8
    epsilon = 0.00001
    x = x0

    # 1. descent direction
    delta_x = -grad(x, args[0], args[1])
    
    # 2. line search
    t = 1
    while fun(x + t * delta_x, args[0], args[1]) > fun(x, args[0], args[1]) + alpha * t * grad(x, args[0], args[1]).T @ delta_x:
        t = beta * t
    
    # 3. update
    x = x + t * delta_x

    # stopping criterion
    if np.linalg.norm( grad(x, args[0], args[1]) ) < epsilon:
        return x
    else:
        return min_gd( fun, x, grad, args=(args[0], args[1]) )