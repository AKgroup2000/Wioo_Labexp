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
