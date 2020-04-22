import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


dt = 0.1
dx = 0.005
k = 10**(-4)
x_max = 1
t_max = 100
u0 = 100

def FTCS(dt,dx,t_max,x_max,k,T0):
    s = k*dt/dx**2
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    u = np.zeros([r,c])
    u[0,:] = u0*np.sin(np.pi/x_max*x)
    for n in range(0,r-1):
        for j in range(1,c-1):
            u[n+1,j] = u[n,j] + s*(u[n,j-1] - 2*u[n,j] + u[n,j+1]) 
        j = c-1 
        u[n+1, j] = u[n,j] + s*(u[n,j-1] - 2*u[n,j] + u[n,j-1])
        j = 0
        u[n+1, j] = u[n,j] + s*(u[n,j+1] - 2*u[n,j] + u[n,j+1])
    return x,u,r,s
    

x,u,r,s = FTCS(dt,dx,t_max,x_max,k,u0)


fig = plt.figure()
ax = plt.axes(xlim=(0,x_max), ylim=(0,u0))
line, = ax.plot([], [], lw=2)

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Change in Temperature')

def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, u[i,:])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,frames=10*int(t_max/dt), interval=1, blit=True)
anim.save("./gifs/temp.gif", writer='imagemagick', fps=30)