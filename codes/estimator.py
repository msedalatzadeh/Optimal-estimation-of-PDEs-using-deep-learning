from parameters import *
from FTCS import *
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import load, array, zeros, reshape



input = load('training-data/input.npy')
output = load('training-data/output.npy')

t = np.arange(0,t_max+dt,dt)
r = len(t)
m, c1 = input.shape
m, c = output.shape

ac_fun = 'selu'
model = Sequential()
model.add(Dense(100, input_dim = c1, activation = ac_fun))
model.add(Dense(1000, activation = ac_fun))
model.add(Dense(500, activation = ac_fun))
model.add(Dense(c, activation = ac_fun))
# model.add(Reshape((r,c1)))

# plot_model(model, to_file='figs\model-plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='Nadam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(input, output, batch_size=100, epochs=5)

#print('evaluation result: [loss, accuracy]=' , eval_result)

model.save('./model/model-selu-Nadam.h5')