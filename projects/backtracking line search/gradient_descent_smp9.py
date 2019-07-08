import numpy as np
def min_gd(fun,x0,grad,args=()):
    alpha=0.3
    beta=0.8
    epsilon=0.00001
    x=x0
    while 1:
        delta_x=(-1)*grad(x,args[0],args[1])
        t=1
        
        while 1:
            left=fun(x+t*delta_x,args[0],args[1])
            right=fun(x,args[0],args[1])+alpha*t*grad(x,args[0],args[1]).dot(delta_x)
            if (left<right).all():
                break
            t=beta*t
            
        x=x+t*delta_x
        
        a=np.linalg.norm(grad(x,args[0],args[1]))
        
        if (a<=epsilon).all():
            break
    return x