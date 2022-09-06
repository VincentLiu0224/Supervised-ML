from re import I
from telnetlib import SE
#from Numerical_methods import runge_kutta
#from Yr2_NM import Session_2
#from Yr2_NM import newton_raphson_new as nrn
import numpy as np
import matplotlib.pyplot as plt
import time 




# drag force, Cd and den should be inplicity when using external funcs defined to avoid error
def drag_f(v, t, Cd, den):
    return -0.5*Cd*den*(v**2)

def f(u, t):
    return np.cos(t)



def f1(u, t):
    return -1*u


def f2(u, t):
    return -100*(u - np.cos(t) - np.sin(t))


def f_hw(u, t):
    beta = 0.5
    kappa = 1/3
    i_new = beta*u[1]*u[0]-kappa*u[0]
    s_new = -beta*u[0]*u[1]
    r_new = kappa*u[0]
    return np.array([i_new, s_new, r_new])


def plot_raw_data(xi, yi, ax):
    ax.plot(xi, yi, label='raw data')
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.grid(True)



# State vector, which is fundimental to the nonlinear ODE to lower order
# basic function: du/dt=f(u, t),
# in which u can be represented by any related variables with respect to t
# i.e. both v and x can be listed, but more interesting one is v
v0 = 0
x0 = 350
u = np.array([v0, x0])

# Euler's forward method
def euler_diff(u_diff, u0, t0, tmax, dt):                         # forward euler's method; u0, t0 start values, with tmax                                                          
    u = [u0]                                                      # udiff is a given diff function of u
    t = [t0]                                                      # first order method(Global truncation error nearly linear to step size)
    while t[-1] < tmax:
        print(t[-1])                                           # trade-off between step size and accuracy
        u.append(u[-1] + u_diff(u[-1], t[-1])*dt)                 # Ut+1 = Ut + dt*f(t) 
        t.append(t[-1]+dt)                                        # stability of explicit method: error increases catastrophically if step size goes over a threshold
    return np.array(u), np.array(t)

# backward Euler's method
# requires 'Jacobi', the differnetial of R
# which will be resolved by either: given function of jackbi=dR
# or intrinsically calculate using quasi-newton method
def simple_back_euler_diff(u_diff, u0, t0, tmax, dt):
    def R(u1, u0, t1, delta_t, f):
        return u1 - u0 - delta_t*f(u1, t1)
    u = [u0]                                                              # backward euler's method from ML_02, Ut+1 = Ut + dt*f(t+1)
    t = [t0]                                                              # computed to solve an equation:
    while t[-1] < tmax:                                                   # R(Ut+1)= Ut+1 - Ut - dt*f(Ut+1, t+1)
        u_new = nrn.quasi_newton_for_backeu(R, u[-1], t[-1], dt, u_diff)  # unconditional stability--if true solution is bounded for all time, the obtained solution is bounded for all time
        u.append(u_new)                                                   # usually used analysing vast range of timescale
        t_new = t[-1] + dt
        t.append(t_new)
    return np.array(u), np.array(t)



i =0.01
s = 1-i
r = 0.
u0 = np.array([i, s, r])
start = time.time()
yf, xf = euler_diff(f_hw, u0, 0., 1000., 0.1)
#ys, xs = simple_back_euler_diff(f_hw, u0, 0., 100., 1.)
#xr, yr = runge_kutta.op(f_hw, u0, 0., 100, 1)  
end = time.time()
print(yf)
print(end - start)
                        # Runge Kutta method of diff-integration
# print(yr)
# Plot figures of backward euler and runge kutta methods

#fig = plt.figure(figsize=(8, 6))
#ax1 = fig.add_subplot(111)
#for i in range(3):
    #plot_raw_data(xf, yf[:, i], ax1)
#plt.show()
