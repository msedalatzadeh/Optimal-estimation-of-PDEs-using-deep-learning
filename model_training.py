import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.utils.vis_utils import plot_model
from parameters import *


x = np.arange(0,x_max+dx,dx) 
t = np.arange(0,t_max+dt,dt)
r = len(t)
c = len(x)


# load the dataset
input = np.loadtxt('train_data/input.csv', delimiter=',')
output = np.loadtxt('train_data/output.csv', delimiter=',')


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=c, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(r*c, activation='relu'))
model.add(Reshape((r,c)))

plot_model(model, to_file='figs/model_plot.png', show_shapes=True, show_layer_names=True)

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset

model.fit(input, output)
# evaluate the keras model
_, accuracy = model.evaluate(input, output)
print('Accuracy: %.2f' % (accuracy*100))
