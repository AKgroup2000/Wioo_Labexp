** * * *  * EXP 5.1 
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t):
    k = 2
    dydt = -k * y
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()

* * * * * * EXP 5.2 
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(y,t):
    dydt = 5*(np.exp(-t/3))
    return dydt

# initial condition
y0 = 5

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.title('5e( −t/RC) Where RC = 3')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()



* * * * * *  *    EXP 5.3

import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
inv = lambda x: np.exp(-x)
def f(u,x):
  return (u[1],-2*u[1]-2*u[0]+np.exp(-x))


y0 = [0,0]
xs = np.linspace(1,10,200)
us = odeint(f, y0, xs)
ys = us[:,0]
plt.plot(xs, ys,'-')
plt.plot(xs, ys, 'r*')
plt.xlabel('t values')
plt.ylabel('x values')
plt.title('(D**2+2*D+2)x = e**(-t)')

* * * ** * * * EXP 5.4 *** * * 
from scipy import integrate
from pylab import * # for plotting commands

def rlc(A,t):
    Vc,x=A
    V = 5.0 #voltageSource
    R = 1.0 # 1 Ohm
    L=1.0e-6 #1mH
    C = 1.0e-6 #1microF
    res=array([x,(V-Vc-(x*R*C))/(L*C)])
    return res

time = linspace(0.0,0.6e-6,1001)
vc,x = integrate.odeint(rlc,[0.0,0.0],time).T
i = 5*(np.exp(-x))
figure()
plot(time,vc)
plt.title('RLC circuit with R = 1Ω,L = 1 mH and C = 1 μF')
xlabel('t')
ylabel('Vc')
show()
