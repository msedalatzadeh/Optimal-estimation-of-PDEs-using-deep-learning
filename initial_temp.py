import numpy as np
import os
import matplotlib.pyplot as plt

x=np.arange(0,1,0.01)
u=8*np.sin(np.pi*x)

plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.axis([0,1,0,9])
plt.text(0.45, 8.5,'$u_0=8$')
plt.plot(x,u)
plt.show()
os.mkdir('figs/')
plt.savefig('figs/u0.png')