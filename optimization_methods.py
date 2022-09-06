import numpy as np
from scipy import linalg as sl, optimize as op
import matplotlib.pyplot as plt


# least squiare optimization
def least_square(M, b):
    return sl.inv(M.T@M)@M.T*b

# gradient descend method, takes numerous iterations to reach the result
# need to compute the gradient of j maunally
def gradient_descent(grad_j, x0, step_size, atol=1e-6, *args):
    x = x0
    itera = 0
    while sl.norm(grad_j(x)) > atol:
        x = x - grad_j(x)*step_size
        itera += 1 
    return x, itera

# Newton's method
def newton_method(grad_j, hess_j, x0, step_size, atol=1e-6):
    x = x0
    itera = 0
    while sl.norm(grad_j(x)) > atol:
        x = x - step_size*(sl.inv(hess_j(x)) @ grad_j(x))
        itera += 1
    return x, itera



def Hessian(x, *args):
    return np.array([[1./((1.-x[0]-x[1])**2)+(1./x[0]**2), 1./((1.-x[0]-x[1])**2)], 
                    [1./((1.-x[0]-x[1])**2), 1./((1.-x[0]-x[1])**2)+(1./x[1]**2)]])



def grad(x):
    return np.array([(1./(1.-x[0]-x[1]))-(1./x[0]), (1./(1.-x[0]-x[1]))-(1./x[1])])

def j(x):
    return -np.log(1-x[0]-x[1])-np.log(x[0])-np.log(x[1])


x0 = np.array([0.85, 0.05])
print(gradient_descent(grad, x0, 0.01))
print(newton_method(grad, Hessian, x0, .1))


# Scipy optimize function
result = op.minimize(j, x0, method='BFGS',                    # methods are 'BFGS', 'nelder-mead', 'CG', Powell'
                     options={'xatol': 1e-8, 'disp': True}).x
print(result)


result_newton = op.minimize(j, x0, method='Newton-CG', jac=grad, hess=Hessian,
                     options={'disp': True}).x

print(result_newton)


# Global minimization
minimizer_kwargs = {"method": "nelder-mead"}
ret = op.basinhopping(j, x0, minimizer_kwargs=minimizer_kwargs,
                   niter=200)
print(ret.x)              # value of the point reaches minimize
print(ret.fun)            # value of the minimized function value