import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import load, array, zeros, reshape
from parameters import *
from FTCS import *


input = load('train_data\input.npy')
output = load('train_data\output.npy')


m, c = input.shape
m, r, c = output.shape

model = Sequential()
model.add(Dense(32, input_dim=c, activation='relu'))
model.add(Dense(r*c, activation='relu'))
model.add(Reshape((r,c)))

plot_model(model, to_file='figs\model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(input, output, epochs=1, batch_size=m)

u0 = random.choices(range(-u_max, u_max), k=c)
_,u_real,_,_ = FTCS(dt,dx,t_max,x_max,k,u0)

u0 = np.asarray(u0).reshape((1,c))
u_pred = model.predict(u0, batch_size=1)
eval_result = model.evaluate(u0, u_pred, batch_size=1)
print('evaluation result' , eval_result)

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


x=np.arange(0,x_max+dx,dx)

def animate_pred(i,dt):
    line.set_data(x, u_pred[0,2*i,:])
    s=2*i*dt
    time.set_text('$time=$%2.1f'%s)
    return line,

def animate_real(i,dt):
    line.set_data(x, u_real[2*i,:])
    s=2*i*dt
    time.set_text('$time=$%2.1f'%s)
    return line,

def animate(i,dt):
    line.set_data([x,x], [u_pred[0,2*i,:],u_real[2*i,:]])
    s=2*i*dt
    time.set_text('$time=$%2.1f'%s)
    return line,

anim = animation.FuncAnimation(fig,lambda i: animate_pred(i,dt), init_func=init,frames=int(t_max/dt/2), interval=0.1, blit=False)
anim.save("./gifs/temp_pred.gif", writer='imagemagick', fps=30)

anim = animation.FuncAnimation(fig,lambda i: animate_real(i,dt), init_func=init,frames=int(t_max/dt/2), interval=0.1, blit=False)
anim.save("./gifs/temp_real.gif", writer='imagemagick', fps=30)

anim = animation.FuncAnimation(fig,lambda i: animate(i,dt), init_func=init,frames=int(t_max/dt/2), interval=0.1, blit=False)
anim.save("./gifs/temp_real_pred.gif", writer='imagemagick', fps=30)