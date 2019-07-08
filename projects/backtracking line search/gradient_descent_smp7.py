import numpy as np

def min_gd(fun, x0, grad, args=()):
    """
    Parameters
    fun : fun(x, *args) (n,)-> 1 (objective function)
    x0 : (n,)
    grad : grad(x, *args) (n,)->(n,) (gradient)
    args : A, b
    """
    alpha = 0.3
    beta = 0.8
    A,b = args
    t = 1
    x = x0
    g = grad(x,A,b)
    while(np.linalg.norm(g) >= 0.001):
        t = 1
        while(fun(x-t*g, A, b)>= fun(x,A,b)-alpha*t*np.dot(g.T,g)):
            t = beta*t
        x = x-t*g
        g = grad(x,A,b)
    
    return x
