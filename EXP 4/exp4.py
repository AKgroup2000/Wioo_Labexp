#EXP 4.2

import sympy as sp

x = sp.Symbol('x')
print("f(x) = x^3 \n")
print(" \n f'(x) = ", sp.diff(x**3))
print("\n f''(x) = ", sp.diff(x**3,x,x))
print("\n f(x) = sin(x)*((2x^2)+2)")

print(" \n f'(x) = ", sp.diff(sp.sin(x)*(2*x**2+2)))
print("\n f(x) = sin(x))/x \n f'(x) = ",sp.diff((sp.sin(x))/x))
print("\n f(x) = (x**2+1)**7 \n f'(x) = ",sp.diff((x**2+1)**7)
print("\n f(x) = exp(3*x) \n f'(x) = ", sp.diff(sp.exp(3*x)))

x, y = sp.symbols('x y')

f = x**4 * y
print(sp.diff(f, x))
print(sp.diff(f, y))
x, y, z = sp.symbols('x y z')

f = x**3 * y * z**2
print(sp.diff(f, z))
f = 2*x**3+4*x
f_prime = sp.diff(f)
print(f_prime)

      
## Exp 4.3
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative

# f(x) define the function.
def f(x):
  return np.sin(x)
derivative(f,1,dx=0.0001)
#xs = np.arange(0, 5, 0.1)
xs = np.linspace(-np.pi, np.pi, 100)
plt.plot(xs,f(xs))
print("F(x) = ", f(xs))
print("Derivative = ", derivative(f, xs, dx = 0.001))
plt.plot(xs,derivative(f,xs, dx=0.001))
ax = plt.gca()
ax.set_ylim(-5,5)
plt.savefig('temp.png')

      
# ~ ~ ~    EXP 4.4 * * * * 
"""
Numerical integration

Scientific Python provides a number of integration routines. A general purpose tool to solve integrals I of the kind
I=∫(b to a) f(x)dx

is provided by the quad() function of the scipy.integrate module.

It takes as input arguments the function f(x) to be integrated (the “integrand”), and the lower and upper limits a and b. It returns two values (in a tuple): the first one is the computed results and the second one is an estimation of the numerical error of that result.
"""

from math import cos, exp, pi
from scipy.integrate import quad

# function we want to integrate
def f(x):
    return exp(cos(-2 * x * pi)) + 3.2

# call quad to integrate f from -2 to 2
res, err = quad(f, -2, 2)

print("The numerical result is {:f} (+-{:g})"
    .format(res, err))

"""
Solving ordinary differential equations

To solve an ordinary differential equation of the type
dy/dt(t)=f(y,t)

with a given y(t0) = y0

, we can use scipy’s odeint function. Here is a (self explaining) example program (useodeint.py) to find
y(t) for t∈[0,2]

given this differential equation:
dy/dt(t)=−2yt with y(0)=1.
"""

%matplotlib inline
from scipy.integrate import odeint
import numpy as N

def f(y, t):
    """this is the RHS of the ODE to integrate, i.e. dy/dt=f(y,t)"""
    return -2 * y * t

y0 = 1             # initial value
a = 0              # integration limits for t
b = 2

t = N.arange(a, b, 0.01)  # values of t for
                          # which we require
                          # the solution y(t)
y = odeint(f, y0, t)  # actual computation of y(t)

import pylab          # plotting of results
pylab.plot(t, y)
pylab.xlabel('t'); pylab.ylabel('y(t)')
      
      
