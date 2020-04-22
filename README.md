# Optimal Estimation of Temperature Change

Consider a one-dimensional steal bar over the interval <img src="/tex/acf5ce819219b95070be2dbeb8a671e9.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/>.


<p align="center"><img src="/tex/19b1a58487f907022d41cb15d7b4b6cd.svg?invert_in_darkmode&sanitize=true" align=middle width=446.20697055pt height=49.315569599999996pt/></p>



```python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt = 0.0005
dx = 0.0005
k = 10**(-4)
x_max = 0.04
t_max = 1
T0 = 100

def FTCS(dt,dx,t_max,x_max,k,T0):
    s = k*dt/dx**2
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    T = np.zeros([r,c])
    T[0,:] = T0*np.sin(np.pi/x_max*x)
    for n in range(0,r-1):
        for j in range(1,c-1):
            T[n+1,j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j+1]) 
        j = c-1 
        T[n+1, j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j-1])
        j = 0
        T[n+1, j] = T[n,j] + s*(T[n,j+1] - 2*T[n,j] + T[n,j+1])
    return x,T,r,s
    

x,T,r,s = FTCS(dt,dx,t_max,x_max,k,T0)

#plot_times = np.arange(0.01,1.0,0.01)
#for t in plot_times:
#    plt.plot(y,T[int(t/dt),:])


fig = plt.figure()
ax = plt.axes(xlim=(0, 0.04), ylim=(0,100))
line, = ax.plot([], [], lw=2)

plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('Change in Temperature')

def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, T[i,:])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,frames=2001, interval=1, blit=True)

```
