import numpy as np
import os
import matplotlib.pyplot as plt
from parameters import *

x=np.arange(0,x_max+dx,dx)
u0=10*np.sin(np.pi*x)

plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.axis([0,x_max,0,1.1*u_max])
plt.text(x_max/2,1.05*u_max,'$u_max=$%d'u_max)
fig=plt.plot(x,u0)
#plt.show()

if os.path.isfile("./figs/u0.png"):
   os.remove("./figs/u0.png")   

plt.savefig("./figs/u0.png")