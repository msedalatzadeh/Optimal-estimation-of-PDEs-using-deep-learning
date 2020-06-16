from numpy import arange

dt = 0.1
dx = 0.01
k = 0.0003
x_max = 1
t_max = 200
u_max = 5
N = 25                                  # Number of training data
act_function = 'linear'
opt_method = 'Adam'
loss_function = 'mean_squared_error' 


x = arange(0,x_max+dx,dx)
t = arange(0,t_max+dt,dt)
c = len(x)
r = len(t)


