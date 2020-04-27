import numpy as np 
import random
from parameters import *
import matplotlib.pyplot as plt
from matplotlib import animation
from FTCS import *
import os


m=11  #number of training samples

input=np.zeros((m,c))
output=np.zeros((m,r,c))

for i in range(1,11):
    u0=random.choices(range(-u_max, u_max), k=c)
    x,u,r,s = FTCS(dt,dx,t_max,x_max,k,u0)

    input[i,:]=u0
    output[i,:,:]=u
  
np.savetxt('train_data/input.csv',input,delimiter=',')
np.savetxt('train_data/output.csv',u,delimiter=',')