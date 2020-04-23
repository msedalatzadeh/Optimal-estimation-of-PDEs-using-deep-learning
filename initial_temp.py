import numpy as np
import os
import matplotlib.pyplot as plt
from parameters import *

x=np.arange(0,x_max+dx,dx)
u=10*np.sin(np.pi*x)

plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.axis([0,x_max,0,1.1*u0])
plt.text(x_max/2, 1.05*u0,'$u_0=$%d'u0)
fig=plt.plot(x,u)
#plt.show()

if os.path.isfile("./figs/u0.png"):
   os.remove("./figs/u0.png")   

plt.savefig("./figs/u0.png")