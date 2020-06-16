from numpy import load, array, zeros, reshape, cos, pi, arange, append
from tensorflow.keras.models import load_model
from scipy.optimize import fminbound
from keras import backend as k
from keras.losses import mean_absolute_error
from parameters import *
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from FTCS import *

model = load_model('model\model-random_data-selu-Adam-mean_squared_error.h5')


x = arange(0,x_max+dx,dx)
t = arange(0,t_max+dt,dt)
c = len(x)
r = len(t)

f = lambda s: -s**2*(s-1)**2*(s-1/2)**2
x0 = fminbound(f, 0, 1)
f_max = -f(x0)
# u0 = u_max*(1-cos(2*pi*x))/2

u0 = -u_max/f_max*f(x)
u_real = FTCS(dt, dx, t_max, x_max, k, u0)
input_test = zeros((r,c+1))
for s in range(0,r):
    input_test[s,0:c] = u0
    input_test[s, c] = t[s]



grads = k.gradients(model.input, model.output)

gradient = k.function([model.input], grads)