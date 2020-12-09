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
