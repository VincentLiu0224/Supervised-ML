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
        d2u[i] = (u[i+2] - 2*u[i+1] + u[i])/(dx**2)
    return d2u


def euler_diff(u_diff, u0, t0, tmax, dt, h):                         # forward euler's method; u0, t0 start values, with tmax                                                          
    u = u0                                                      # udiff is a given diff function of u
    t = t0                                                      # first order method(Global truncation error nearly linear to step size)
    print('start evaluation at tmax =', tmax)
    while t < tmax:                                           # trade-off between step size and accuracy
        u[1: -1] = u[1: -1] + u_diff(u, t, h)*dt                 # Ut+1 = Ut + dt*f(t) 
        t = t + dt                                        # stability of explicit method: error increases catastrophically if step size goes over a threshold
        print('evaluating point at', t)
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
        print('evaluted point at', t)
    return u, t



def quasi_newton_for_backeu(f, x0, t0, dt, diff, dx, atol=1.0E-6):
    x = x0
    x_new = x0
    while True:
        dfdx = (f(x + dx, x0, t0+dt, dt, diff, dx) - f(x, x0, t0+dt, dt, diff, dx))/(dx)
        x_new[1: -1] =  x[1: -1] - (f(x, x0, t0+dt, dt, diff, dx)/dfdx)
        if sl.norm(x_new[1: -1]-x[1: -1]) < atol:
            return x_new
        else:
            x = x_new



def conv_diff(u, t, dx, k=5e-4, c=0.5):
    return k*d2udx2(u, dx)-c*dudx(u, dx, '')   # select central diff





def plot_raw_data(xi, yi, ax):
    ax.plot(xi, yi, label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)


x_l = 1
nSpace = 1000
h = x_l/nSpace                                # mesh size


x = np.linspace(0, x_l, nSpace)
u0b1 = np.zeros(len(x))
u0b2 = np.zeros(len(x))
ite = 0
for i in x:
    if i <= 0.2+0.08:
        u0b1[ite] = 1
    if i >= 0.2-0.08:
        u0b2[ite] = 1
    ite += 1
u0 = 0.1*np.exp(-((x-0.2)**2)/(2*0.02**2))*u0b1*u0b2
fig, (ax1) = plt.subplots(1, 1, figsize=(14, 7))

er_tot = []
for i in [1.]:
    uo = u0.copy()
    u, t =  simple_back_euler_diff(conv_diff, uo, 0., i, 0.0001, h)
    plot_raw_data(x, u, ax1)
plt.show()