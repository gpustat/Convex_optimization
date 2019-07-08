import numpy as np
#parameter setting.
alpha=0.3
beta=0.8
epsilon = 1.0e-3

# Least Squares function
def LeastSquares(x,A,b):
    return np.linalg.norm(A@x-b)**2

# gradient  
def grad_LeastSquares(x,A,b):
    return 2*((A.T@A)@x-A.T@b)

def min_gd(fun,x0,grad,args=()):
    dx0= -grad(x0,args[0],args[1])
    t=1
    while np.linalg.norm(dx0)>epsilon:
        while fun(x0 + t* dx0,args[0],args[1])- fun(x0,args[0],args[1]) - alpha*t*np.dot(grad(x0,args[0],args[1]).T,dx0) >0:
            t *= beta
        x0= x0+t*dx0
        dx0= -grad(x0,args[0],args[1])
        t=1
    return x0

