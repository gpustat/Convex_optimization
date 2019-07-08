import numpy as np

def min_gd(fun, x0, grad, args=()):
    alpha = 0.3
    beta = 0.8
    epsil = 0.00001
    x = x0
    norm_grad = np.linalg.norm(grad(x, *args))
    dx = -grad(x, *args)
    while norm_grad > epsil:
        t = 1
        dx = -grad(x, *args)
        while (fun(x+t*dx,*args) >= fun(x,*args) + alpha*t*(grad(x,*args).T)@dx):
            t=beta*t
        x=x+t*dx
        norm_grad = np.linalg.norm(grad(x, *args))
    
    return x
