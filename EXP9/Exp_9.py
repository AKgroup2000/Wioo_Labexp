* * * * * *   EXP 9.1 * * * * * * * 

import numpy as np
from random import randint

def Toss():
  while True:
    toss = input("Press T for toss Q for quite : ")
    T = ['T','t']
    S=[1]
    if toss in T:
      p = randint(0,2)
      if p in S: print("Head")
      else: print("Tail")
    else:
      print("Exit \n")
      return 0

Toss()
print("Successfully completed")

* * * * * * *  *  EXP 9.2  * * * * * * 

import numpy as np
from random import randint
Sum =0
N = [10,500,1000, 5000, 50000]
for i in N:
  for j in range(i):
    p = randint(0,2)
    if p == 1:
      Sum+=1
  prob = Sum/i
  print("Probability of head occured in",i," toss = ",prob)

# * * * * * * * EXP 9.3 * * * * 
from numpy import random 
import matplotlib.pyplot as plt
%matplotlib inline

N = [100, 500, 1000, 5000, 500000]
Sum = 0
Prob = []
Abs_Err = []
for i in N:
  for j in range(i):
    if random.randint(0,2) == 1:
      Sum+=1
  print("Sum = ",Sum)
  Prob.append((Sum/i))
  print("Probability = ", Sum/i)
for i in Prob:
  Abs_Err.append(abs(0.5-i))

fig, ax = plt.subplots()
ax.plot(N, Abs_Err)
ax.set(xlabel='No of toss',
       ylabel='Abs Error',
       title='ABS Error calculation')

plt.show()
