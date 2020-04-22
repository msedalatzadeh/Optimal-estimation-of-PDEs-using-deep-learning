import numpy as np
import os
import matplotlib.pyplot as plt

x=np.arange(0,1,0.01)
u=10*np.sin(np.pi*x)

plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.axis([0,1,0,11])
plt.text(0.45, 10.5,'$u_0=10$')
fig=plt.plot(x,u)
#plt.show()

if os.path.isfile("./figs/u0.png"):
   os.remove("./figs/u0.png")   

plt.savefig("./figs/u0.png")