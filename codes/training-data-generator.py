import numpy as np 
import random
from parameters import *
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import save, savez_compressed, sin, pi, zeros, array
from FTCS import *
import os

N = 10

X = np.arange(0,x_max+dx,dx) 
t = np.arange(0,t_max+dt,dt)
r = len(t)
c = len(X)
u_max = int(u_max)

m = N*r           # number of input data

input = zeros((m,c+1))
output = zeros((m,c))

def IC(x,omega,u_max):
    u = 16*u_max*(x**2)*((x-1)**2)*sin(omega*pi*x)
    return u

n=0
for omega in range(1,N+1):
    u0 = array([IC(x,omega,u_max) for x in X])
    u = FTCS(dt, dx, t_max, x_max, k, u0)
    for s in range(0,r):
        input[n,0:c] = u0
        input[n,c] = t[s] 
        output[n,:] = u[s,:]
        n = n+1


np.save('./training-data/input.npy', input)
np.save('./training-data/output.npy', output)