import numpy as np
    
def min_gd(fun,x0,grad,args=()):
    alpha=0.3
    beta=0.8
    x=x0
    #cnt=1
    while(np.linalg.norm(grad(x,*args))>0.00001):
        dir=(-1)*grad(x,*args)
        t=1
        while(fun(x+t*dir,*args)>=(fun(x,*args)+(-1)*alpha*t*(np.linalg.norm(grad(x,*args))**2))):
            t=t*beta
        #print("iteration",cnt,"t=",t)
        x=x+t*dir
        #cnt=cnt+1
    return x