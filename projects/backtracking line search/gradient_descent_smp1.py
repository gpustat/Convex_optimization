import numpy as np

def min_gd(fun, x0, grad, args=()):
    alpha = 0.3
    beta = 0.8
    epsilon = 0.00001
    A = args[0]
    b = args[1]
    
    while True:
        get_grad= - grad(x0, A, b)
        t = 1
        while True:
            t *= beta
            if fun(x0, A, b) + (alpha * t * np.dot(-get_grad, get_grad)) >  fun(x0 + (t * get_grad), A, b):
                break
         
        
        x0 = x0 + t * get_grad
        
        if np.linalg.norm(grad(x0, A, b), 2) < epsilon:
            break
            
    return x0