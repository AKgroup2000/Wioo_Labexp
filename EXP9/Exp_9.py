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

 * * * ** * * * * *  * *  *EXP 9.4
  
import random 
%matplotlib inline

import matplotlib.pyplot as plt


a =[]
b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
c = []
sum =0
for i in range(10000):
  p = random.uniform(0,10)
  a.append(int(p))

for i in b:
  sum =0
  for j in a:
    if j is i:
      sum+=1
  c.append(sum)
print(c)

fig, ax = plt.subplots()
ax.plot(b, c)
ax.set(xlabel='Sample',
       ylabel='Total',
       title='Uniform Random Vector ')

plt.show()

* * * * *  * * * * * * EXP 9.5
import random 
%matplotlib inline

import matplotlib.pyplot as plt

Thre =2
a =[]
b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
c = []
sum =0
for i in range(10000):
  p = random.uniform(0,10)
  a.append(int(p))
sum =0 
for j in a:
   if j is Thre:
     sum+=1

print("Function Reached Threshold 2 = ",sum)

* * * *  * * * * * * EXP 9.6

import random 
%matplotlib inline

import matplotlib.pyplot as plt

Thre =2
a =[]
b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
c = []
sum =0
for i in range(10000):
  p = random.uniform(0,10)
  a.append(int(p))
sum =0 
for j in a:
   if j is Thre:
     sum+=1

print("Function Reached Threshold 2 = ",sum)
Sum1 = 0
for j in a:
   if j > Thre:
     Sum1+=1

print("Function Reached Threshold 2 = ",Sum1)
print("Function crossed Threshold 2 = ",sum + Sum1)

