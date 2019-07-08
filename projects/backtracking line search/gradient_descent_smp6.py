import numpy as np

def min_gd(fun, x0, grad, args = ()):
    # your code goes here
    newX = x0
    upsilon = 0.0001
    gradSum = 0
    for elm in grad(newX, args[0], args[1]):
        gradSum += elm ** 2
    while gradSum > upsilon ** 2:
        #1. Descent direction
        deltaX = -grad(newX, args[0], args[1]);
        #2. backtracking line search(p.7)
        alpha, beta = 0.3, 0.8
        t = 1
        while True:
            if fun(newX + t * deltaX, args[0], args[1]) < fun(newX, args[0], args[1]) + alpha * t * grad(newX, args[0], args[1]).T@deltaX:
                break
            t *= beta;
        #3. update
        newX = newX + t * deltaX
        #4. until upsilon(0.0001)
        gradSum = 0
        for elm in grad(newX, args[0], args[1]):
            gradSum += elm ** 2
    #5. converge
    return newX


