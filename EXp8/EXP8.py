#  * * * ** * * * ** EXP 8.1 

# Just an introduction based on forier series

# ****** EXP 8.2 

%pylab inline
import matplotlib.pyplot as plot
from matplotlib.pyplot import figure
figure(num=None, figsize=(100, 20), dpi=90, facecolor='w', edgecolor='k')

import numpy as np
forier = lambda i,t,T: ((-1)**(i+1))*((4/(np.pi*((2*i)-1)))*np.cos((2*np.pi*((2*i)-1)*t)/T))
T = 20
time = []
Sum =[]
for t in np.arange(0,100,0.01):
  F =[]
  time.append(t)
  for n in range(1,3,1):
    F.append(forier(n,t,T))
  Sum.append(sum(F))

#plot(time, Sum, 'r-')

plot.plot(time, Sum)
# Give a title for the sine wave plot

plot.title('Sine wave')
# Give x axis label for the sine wave plot

plot.xlabel('Time')
# Give y axis label for the sine wave plot

plot.ylabel('Amplitude = Forier value')

plot.grid(True, which='both')

plot.axhline(y=0, color='k')

plot.show()
# Display the sine wave

plot.show()


