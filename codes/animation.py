import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import load, array, zeros, reshape, cos, pi, arange, append
from parameters import *
from FTCS import *
import ffmpy
from scipy.optimize import fminbound
from tensorflow.keras.models import load_model

######## Choose a test initial condition from below ##

# Polynomial initial condition
f = lambda s: -s**2*(s-1)**2*(s-1/4)**2
x0 = fminbound(f, 0, 1)
f_max = -f(x0)
u0 = -u_max/f_max*f(x)

## Sinusodial initial condition
# u0 = u_max*(1-cos(2*pi*x))/2

######## Choose sensor shape below ###################

## sensor over l 
# l = range(int((c-1)/4), int(3*(c-1)/4))
# c1 = len(l)
# input = zeros((r,c1+1))
# for s in range(0,r):
#     input[s, 0:c1] = u0[l]
#     input[s, c1] = t[s]

## sensor over [0,1]
input = zeros((r,c+1))
for s in range(0,r):
    input[s, 0:c] = u0
    input[s, c] = t[s]

######## load your model below ##############################################

model = load_model('model/model-selu-Adam-mean_squared_error-typ1-15modes.h5')

#############################################################################

u_real = FTCS(dt, dx, t_max, x_max, k, u0)

u_pred = model.predict(input)


fig = plt.figure()
ax = plt.axes(xlim=(0,x_max), ylim=(-u_max,u_max))
time=ax.annotate('$time=$0',xy=(0.1, -4))
line, = ax.plot([], [], lw=2)

plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Change in Temperature (Neumann BCs)')

plotcols = ["blue" , "red"]
plotlabels = ["real" , "prediction"]
lines = []
for index in range(2):
    lobj = ax.plot([],[], lw=2, color=plotcols[index], label=plotlabels[index])[0]
    ax.legend()
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines




def animate(i,dt):
    xlist = [x, x]
    ylist = [u_real[4*i,:], u_pred[4*i,:]]
    s=4*i*dt
    time.set_text('$time=$%2.1f'%s)
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 
    return lines

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, bitrate=1800)
# writer = animation.FFMpegWriter(fps=60) 

anim = animation.FuncAnimation(fig,lambda i: animate(i,dt), init_func=init,frames=int(t_max/dt/4), interval=0.1, blit=False)
#anim.save('./gifs/real-solution.gif', writer='imagemagick', fps=30)
# anim.save('./mp4s/real-solution.mp4', writer = PillowWriter, fps=30)


anim.save('./mp4s/IC3-typ1-15modes.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
ff = ffmpy.FFmpeg(inputs = {'./mp4s/IC3-typ1-15modes.mp4':None} , 
    outputs = {'./gifs/IC3-typ1-15modes.gif': None})
ff.run()