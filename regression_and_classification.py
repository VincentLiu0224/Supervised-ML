from scipy.optimize import minimize as min
import scipy.linalg as sl
import numpy as np


# a = np.array([[2., 3., 1.], [1., 9., 1.]])   # assume row_num = M, the data point number; column_num = N 
# print(sl.det(a.T@a))                         # If M<N i.e. underdetermined, matrix is not invertible i.e. det=0, we need regularization
# print(sl.det(a@a.T))


# least square regression
def poly_least_sqr(y, z, degree):            # y--input, z--output
    coeff = np.polyfit(y, z, degree)
    func = np.poly1d(coeff)
    print('function with degree of {first} is: \n {sec}'.format(first=degree, sec=func))
    return func


# least square regression generalized, with predefined function set
# lambda and regularization method should be predefined if needed,
def least_square(func_set, x, y, c_ini, *args):
    X = np.ndarray(shape=(len(x), len(c_ini)))
    for i in range(len(x)):
        X[i, :] = func_set(x[i])
    if len(c_ini) <= len(x):
        lam = 0
        print('Coefficient processed without regularization')
    else:
        lam = args[0]
        print('{fir} regularization method used to compute coefficients'.format(fir=args[1]))
    def f(c):
        c_tot = 0
        if len(args) == 2:
            if args[1] == 'lasso':
                for i in c:
                    c_tot += abs(i)
            elif args[1] == 'tikhonov':
                for i in c:
                    c_tot += i**2
        return sl.norm(y-np.matmul(X, c.T))+lam*c_tot
    coeff = min(f, c_ini, method='Nelder-mead', options={'disp': False}).x    # choose 'CG', 'Powell', 'nelder-mead', 'BFGS'
    return coeff
# predefined function set
def func_set1(x):
    return np.array([0.5*x[0], 10*x[1]+1, x[2]-10, 9.5*x[3]-1., x[4]**2])
def func_set2(x):
    return np.array([0.5*x[0], 10*x[1]+1, x[2]-10, 9.5*x[3]-1., x[4]*2, 9*x[5], x[6]**0.5, x[7]+5, x[8]/8,
                    0.4*x[9], x[10]+8, x[11]+1, x[12]-1, x[13], 2*x[14]])



def logistic(x, y, c_ini):                    # x---attribute; y---labels either 0 or 1
    x_0 = []
    x_1 = []
    for i in range(len(y)):
        if y[i] == 0:
            x_0.append(x[i])
        elif y[i] == 1:
            x_1.append(x[i])
    def f0(c):
        s_new = 1.
        for i in x_0:
            s = np.matmul(np.array(c), i)
            s_new = s_new*(1. - (np.exp(s)/(1.+np.exp(s))))
        return  s_new
    def f1(c):
        s_new = 1.
        for i in x_1:
            s = np.matmul(np.array(c).T, i)
            s_new = s_new*np.exp(s)/(1.+np.exp(s))
        return  s_new
    def f_tot(c):
        return f0(c)*f1(c)
    coeff = min(f_tot, c_ini, method='Nelder-mead',                  # have to use nelder-mead method to work  
                options={'xatol':1e-8,'disp': True}).x               
    return coeff
        

# empirical risk minimization
# choose a set of data from the whole dataset
# compute optimization by using loss funtion, compute the gradient analytically, or using BFGS:
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#id23
# calculate new coeff using: Cx+1 = Cx - sum(alpha*grad(Cx))
# randomly select a mini-batch to compute grad
# grad need anallytical form, which is annoying, and scipy does't have a 
# sampling option for computing gradient in methods like BFGS




# Cross validation
def loocv(func_set, x, y, c0, *args):
    c = c0
    for i in range(4):
        x_test = x[i]
        y_test = y[i]
        x_learn = np.delete(x, i, 0)
        y_learn = np.array(list(y).pop(i))
        c = least_square(func_set, x_learn, y_learn, c, 0.01, 'lasso')
        yn = c*func_set(x_test)
        error = sum(yn-y_test)
        print('iteration {fir} completed, error obtained {sec}'.format(fir=i, sec=error))
    return c
        





x = np.random.rand(10, 5)    
y = np.array([5., 4., 8., 1.5, 2.5, 130., 9.5, 0.1, 8., 4.6])
y1 = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
c0 = np.random.rand(5)


# least square regression, if underdetermined problem, 
# one must define lamda value and style of regularization method
print(least_square(func_set1, x, y, c0, 0.01, 'lasso'))    
print(least_square(func_set1, x, y, c0, 0.01, 'tikhonov'))


# logistic regression classifying importance of characteristics given in x
print(logistic(x, y1, c0))

print(loocv(func_set1, x, y, c0))