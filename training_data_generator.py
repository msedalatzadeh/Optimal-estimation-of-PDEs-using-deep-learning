import numpy as np 
import random
from parameters import *
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import savez_compressed
from FTCS import *
import os


m=11  #number of training samples

x = np.arange(0,x_max+dx,dx) 
t = np.arange(0,t_max+dt,dt)
r = len(t)
c = len(x)

input=np.zeros((m,c))
output=np.zeros((m,r,c))

for i in range(1,11):
    u0=random.choices(range(-u_max, u_max), k=c)
    x,u,r,s = FTCS(dt,dx,t_max,x_max,k,u0)

    input[i,:]=u0
    output[i,:,:]=u
  

np.savez_compressed('train_data/input.npz',input)
np.savez_compressed('train_data/output.npz',output)

np.save('train_data/input.npy',input)
np.save('train_data/output.npy',output)