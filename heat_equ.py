# from Yr2_NM import Session_2
# import ML_2 as m2
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl


def dudx(u, dx, opt):
    du = np.zeros(len(u)-2)
    if opt == 'forward':
        for i in range(1, len(u)-1):
            du[i-1] = (u[i+1] - u[i])/dx
    elif opt == 'backward':
        for i in range(1, len(u)-1):
            du[i-1] = (u[i] - u[i-1])/dx
    else:
        for i in range(1, len(u)-1):
            du[i-1] = (u[i+1] - u[i-1])/(2*dx)
    return du


def d2udx2(u, dx):
    d2u = np.zeros(len(u)-2)
    for i in range(0, len(d2u)):
        d2u[i] = (u[i+2] - 2*u[i+1] + u[i])/(dx**2.)
    return d2u


# Euler's forward method
def euler_diff(u_diff, u0, t0, tmax, dt, h):                         # forward euler's method; u0, t0 start values, with tmax
    print('start evaluating at tmax', tmax)
    u = u0                                                      # udiff is a given diff function of u
    t = t0                                                      # first order method(Global truncation error nearly linear to step size)
    while t < tmax:                                           # trade-off between step size and accuracy
        u[1:-1] = u[1:-1] + u_diff(u, t, h)*dt                 # Ut+1 = Ut + dt*f(t) 
        t = t + dt                                        # stability of explicit method: error increases catastrophically if step size goes over a threshold
        print('evaluated point at', t)
    return u, t


def simple_back_euler_diff(u_diff, u0, t0, tmax, dt, h):
    def R(u1, u0, t1, delta_t, f, h):
        return u1[1: -1] - u0[1: -1] - delta_t*f(u1, t1, h)
    print('start evaluation at tmax =', tmax)
    u = u0                                                              # backward euler's method from ML_02, Ut+1 = Ut + dt*f(t+1)
    t = t0                                                              # computed to solve an equation:
    while t < tmax:                                                   # R(Ut+1)= Ut+1 - Ut - dt*f(Ut+1, t+1)
        u = quasi_newton_for_backeu(R, u, t, dt, u_diff, h)        # unconditional stability--if true solution is bounded for all time, the obtained solution is bounded for all time                                                  
        t = t + dt                                                  # usually used analysing vast range of timescale
        print('evaluated point at', t)
    return u, t



def quasi_newton_for_backeu(f, x0, t0, dt, diff, dx, atol=1.0e-9):
    x = x0
    x_new = x0
    while True:
        dfdx = (f(x + dx, x0, t0+dt, dt, diff, dx) - f(x, x0, t0+dt, dt, diff, dx))/(dx)
        x_new[1: -1] =  x[1: -1] - (f(x, x0, t0+dt, dt, diff, dx)/dfdx)
        if sl.norm(x_new[1: -1]-x[1: -1]) < atol:
            return x_new
        else:
            x = x_new


# Using heat equation

def heat_equ(u, t, dx, k=130):
    return k*d2udx2(u, dx)

def exact(x, tmax, k=130):
    u = 0
    for n in range(1, 20):
        w = 2*n-1
        u += (4./(w*np.pi))*np.exp(-((w*np.pi/2.)**2)*k*tmax)*np.sin(w*x*np.pi/2.)
    return u, x








# Initial boundary value problem (IBVP)
# State vector 'u' has two dimensions, one in space and the other in time
# i.e. u(x, t)
# This module focuses on solving heat equation with a pair of homogeneous drichlet conditions and a initial condition


# Computing u(x, t), we start from computing initial state u0
# Defining range of spactial dimension
x_l = 2

# mesh spacing 'h', which is the spacing of adjacent element in space dimensions to model
# h = 0.5
# or a conversely and to garantee an integer for the array size
nSpace = 120
h = x_l/nSpace                                # mesh size
u0 = np.zeros(nSpace) + 1.                      # vector u at t=0
u0[0] = 0.
u0[-1] = 0.

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7))


x = np.linspace(0, x_l, nSpace)

# All codes are only designed for a pair of homogeneous boundary conditions and a initial condition
for i in [0.00001, 0.0001, 0.001, 0.01]:
# for i in [0.01, 0.1, 1., 3.]:
    # choose either forward euler method or backward euler method
    uo = u0.copy() 
    u, t =  simple_back_euler_diff(heat_equ, uo, 0., i, 0.000001, h)   # step size is very crucial for it to work
    ax1.plot(x, u)
    ax1.set_title('Heat equation using backward euler diff')
    #u, t =  euler_diff(heat_equ, u0, 0., i, 0.0000001, h)   # step size is very crucial for it to work
    #ax2.plot(x, u)
    ax2.set_title('Heat equation using euler diff')
    ue, te = exact(x, i)
    ax3.plot(x, ue)
    ax3.set_title('Exact solution')
# plt.plot(x, ue)


# print(rk.op(heat_equ, u0, 0.0, 0.5, 0.01)[1][-1])
plt.show()