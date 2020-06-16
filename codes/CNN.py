# Module for convolutional neural network (CNN) predictor

import numpy as np 
import random
from parameters import *
from FTCS import *
import matplotlib
matplotlib.use("Agg")
from numpy import save, savez_compressed, sin, cos, pi, zeros, array, vstack

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv1D, Conv2D, Flatten, GRU, TimeDistributed, SimpleRNN, Dropout, MaxPooling1D
from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
from matplotlib import animation
from random import sample, choices
from itertools import chain
import os
from scipy.fft import fft, ifft, fft2, ifft2, dct, idct, dst, idst
from numpy.random import rand
import ffmpy
from scipy.optimize import fminbound


x = np.arange(0,x_max+dx,dx) 
t = np.arange(0,t_max+dt,dt)
r = len(t)
c = len(x)
u_max = int(u_max)


#################### To Generate Training Data ##########################################################
r1 = r-1

l1 = list(range(int(c/10),int(4*c/10)))       # sensor component 1
l2 = list(range(int(3*c/5),int(4*c/5)))       # sensor component 2
l = l1 + l2                                   # sensor shape
l = list(range(0,c))
cl = len(l)

m = N*(r-1)
input = zeros((m,1,cl))
output = zeros((m,1,c))

u0 = zeros(c)
def IC(x,omega,u_max):
    u = u_max*cos(omega*pi*x)
    return u

n=0
for omega in range(0,N):
    u0 = IC(x,omega,u_max)
    plt.plot(x, u0)
    u = FTCS(dt, dx, t_max, x_max, k, u0)
    for s in range(0,r-1):
        input[n,0,:] = u[s,l]
        output[n,0,:] = u[s+1,:]
        n = n+1

plt.xlabel('x')
plt.ylabel('u0(x)')
plt.xlim(0,1)
plt.title('Training Initial Conditions')

plt.savefig('./figs/CNN-training_ICs-%dmodes.png'%N)
save('./training-data/CNN-training_input-%dmodes.npy'%N, input)
save('./training-data/CNN-training_output-%dmodes.npy'%N, output)

############################# To Build and Train a Model ######################################
# model.add(layers.Lambda(lambda v: ...)))  ## use this line to add custom layer

c1 = c+1

model = Sequential([
    Conv1D(filters=c, activation=act_function, kernel_size=cl,
     strides=1, padding="same", input_shape=(1, cl))
     #SimpleRNN(c, return_sequences=True, activation=act_function),
     #TimeDistributed(Dense(c, activation=act_function))
     ])

model.compile(optimizer=opt_method, loss=loss_function, metrics=['accuracy'])

model.fit(input, output, batch_size=250, epochs=5)

########################### To Make Prediction ###############################################

## Polynomial initial condition
f = lambda t: -t**2*(t-1)**2*(t-1/2)**2
x0 = fminbound(f, 0, 1)
f_max = -f(x0)
u0 = -u_max/f_max*f(x)

u_real = FTCS(dt, dx, t_max, x_max, k, u0)

l_ = [x for x in range(0,c) if x not in l]

u_pred = zeros((r,c))
# for s in range(0,r-3,4):
#     u_pred[s,:] = model.predict(u_real[s,l].reshape((1,1,cl)))
#     u_pred[s+1,:] = u_pred[s,:]
#     u_pred[s+2,:] = u_pred[s,:]
#     u_pred[s+3,:] = u_pred[s,:]

u_pred[0,:] = u0
for s in range(0,r-1):
    u_pred[s+1,:] = model.predict(u_pred[s,l].reshape((1,1,cl)))

####################### To Create Animation ##################################################
fig = plt.figure()
y_min, y_max = [-u_max/2, 1.5*u_max]
ax = plt.axes(xlim=(0,x_max), ylim=(y_min,y_max))
time=ax.annotate('$time=$0',xy=(0.05, 6.5))
line, = ax.plot([], [], lw=2)

plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('Change in Temperature (Mixed BCs)')

plotcols = ["blue" , "red", "green"]
plotlabels = ["Actual" , "Prediction", "Sensor"]
plotlws = [2, 2, 3] 
plotmarkers = ['.','.','s']
lines = []
for index in range(3):
    lobj = ax.plot([],[], 's', lw=plotlws[index], marker=plotmarkers[index], color=plotcols[index], label=plotlabels[index])[0]
    ax.legend()
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines



x_sensor = x[l]
y_sensor = np.array([y_min]*cl)

def animate(i,dt):
    xlist = [x, x, x_sensor]
    ylist = [u_real[4*i,:], u_pred[4*i,:], y_sensor]
    s=4*i*dt
    time.set_text('$time=$%2.1f'%s)
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 
    return lines



###### Save Data Below ##################################
plot_model(model, to_file='./figs/CNN-model_plot.png', show_shapes=True, show_layer_names=True)
model.save('./model/CNN-%s-%s-%s-%dmodes.h5'%(act_function,opt_method,loss_function,N))
anim = animation.FuncAnimation(fig,lambda i: animate(i,dt), init_func=init,frames=int(t_max/dt/4), interval=0.1, blit=False)
anim.save('./mp4s/CNN-%dmodes.mp4'%N, fps=30, extra_args=['-vcodec', 'libx264'])
ff = ffmpy.FFmpeg(inputs = {'./mp4s/CNN-%dmodes.mp4'%N:None} , 
    outputs = {'./gifs/CNN-%dmodes.gif'%N: None})
ff.run()

