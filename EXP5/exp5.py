


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
