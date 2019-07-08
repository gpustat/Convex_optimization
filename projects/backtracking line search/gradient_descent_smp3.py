import numpy as np


#backtracking line search
def back_line_search(fun,x0,grad,args):
 	t=1
 	alpha=0.3
 	beta=0.8
 	delta_x = -(grad(x0,args[0],args[1]))
 	while fun((x0+t*delta_x),args[0],args[1])>=fun(x0,args[0],args[1])+alpha*t*(grad(x0,args[0],args[1])).T@delta_x :
 		t=t*beta
 	return t

#gradient descent algorithm
def min_gd(fun,x0,grad,args):
	epsilon = 10**-10
	error = np.linalg.norm(grad(x0,args[0],args[1]))**2
	while error>epsilon :
		t=back_line_search(fun,x0,grad,args)
		x0 = x0 - t*(grad(x0,args[0],args[1]))	
		error = np.linalg.norm(grad(x0,args[0],args[1]))**2    
	return x0 