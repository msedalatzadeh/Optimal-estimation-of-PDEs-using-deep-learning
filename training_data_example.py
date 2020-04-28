import numpy as np 
import random
from parameters import *
import matplotlib.pyplot as plt
from matplotlib import animation
from FTCS import *
import os



x=np.arange(0,x_max+dx,dx)
c = len(x)

u0=random.choices(range(-u_max, u_max), k=c)

plt.xlabel('$x$')
plt.ylabel('$u(x,0)$')
plt.axis([0,x_max,-u_max,u_max])
fig=plt.plot(x,u0)
#plt.show()
if os.path.isfile("./figs/u0_train.png"):
   os.remove("./figs/u0_train.png")   
plt.savefig("./figs/u0_train.png")

x,u,r,s = FTCS(dt,dx,t_max,x_max,k,u0)

fig = plt.figure()
ax = plt.axes(xlim=(0,x_max), ylim=(-u_max,u_max))
time=ax.annotate('$time=$0',xy=(0.1, 4.5))
line, = ax.plot([], [], lw=2)

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Change in Temperature')

def init():
    line.set_data([], [])
    return line,


def animate(i,dt):
    line.set_data(x, u[2*i,:])
    s=2*i*dt
    time.set_text('$time=$%2.1f'%s)
    return line,


anim = animation.FuncAnimation(fig,lambda i: animate(i,dt), init_func=init,frames=int(t_max/dt/2), interval=0.1, blit=False)
anim.save("./gifs/temp_train.gif", writer='imagemagick', fps=30)
