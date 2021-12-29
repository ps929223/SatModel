'Listing3. Python function for Lorenz Dynamics'

import numpy as np
def Lorez63(state, *args): # Lorez 96 model
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x,y,z = state # Unpack the state vector
    f = np.zeros(3) # Derivatives
    f[0] = sigma * (y-x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    return f


'Listing 4. Python function for the time integration using the 1st Euler and the 4th Runge-Kutta schemes'
import numpy as np

def euler(rhs, state, dt, *args):
    k1 = rhs(state, *args)
    new_state = state + dt * k1
    return new_state

def RK4(rhs,state, dt, *args):
    k1 = rhs(state, *args)
    k2 = rhs(state+k1*dt/2, *args)
    k3 = rhs(state+k2*dt/2, *args)
    k4 = rhs(state+k3*dt, *args)
    new_state = state + (dt/6) * (k1+2*k2+2*k3+k4)
    return new_state

'Listing 5. Implementation of the 3DVAR for Lorez 63 system'
import numpy as np
import matplotlib.pyplot as plt

# Application: Lorenz 63
# parameters
sigma = 10.0
beta = 8.0/3.0
rho = 28.0
dt = 0.01
tm = 10
nt = int(tm/dt) # 1000 # No. of time stamp
t = np.linspace(0,tm,nt+1) # [ 0.  ,  0.01,  0.02, ...,  9.98,  9.99, 10.  ]

u0True = np.array([1,1,1]) # True initial condition

#################### Twin experiment ####################

np.random.seed(seed=1)
sig_m = 0.15 # standard deviation for measurement noise
R = sig_m**2*np.eye(3) # covariance matrix for measurement noise
H = np.eye(3) # linear observation operator

dt_m = 0.2 # time period between observation
tm_m = 2 # maximum time for observations
nt_m = int(tm_m/dt_m) # number of observation instants # 10

# t_m = np.linspace(dt_m,tm_m,nt_m) # np.where( (t<=2) & (t%0.1==0) )[0]
ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)
t_m = t[ind_m]

# time integration
uTrue = np.zeros([3,nt+1])
uTrue[:,0] = u0True  # assign initial
km = 0
w = np.zeros([3,nt_m])

for k in range(nt):
    uTrue[k+1] = RK4(Lorez63, uTrue[:,k],dt,sigma,beta,rho)
    if (km < nt_m) and (k+1==ind_m[km]):
        w[:,km] = H@uTrue[:,k+1] + np.random.normal(0,sig_m,[3,])
    km=km+1

plt.plot(t,uTrue[0,:])







