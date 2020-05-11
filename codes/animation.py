import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import load, array, zeros, reshape, cos, pi, arange, append
from parameters import *
from FTCS import *
from tensorflow.keras.models import load_model

x = arange(0,x_max+dx,dx)
t = arange(0,t_max+dt,dt)
c = len(x)
r = len(t)
u0 = u_max*(1-cos(2*pi*x))/2
u_real = FTCS(dt, dx, t_max, x_max, k, u0)


model = load_model('model\model-selu-Adam-mean_squared_error.h5')

input = zeros((r,c+1))
for s in range(0,r):
    input[s,0:c] = u0
    input[s, c] = t[s]

u_pred = model.predict(input)


fig = plt.figure()
ax = plt.axes(xlim=(0,x_max), ylim=(-1,u_max))
time=ax.annotate('$time=$0',xy=(0.1, 4.5))
line, = ax.plot([], [], lw=2)

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Change in Temperature (Dirichlet BCs)')

plotcols = ["blue" , "red"]
lines = []
for index in range(2):
    lobj = ax.plot([],[], lw=2, color=plotcols[index])[0]
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
anim.save('./mp4s/real-prediction-selu-Adam-mean_squared_error.mp4', fps=30, extra_args=['-vcodec', 'libx264'])