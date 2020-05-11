import numpy as np

def FTCS(dt,dx,t_max,x_max,k,u0):
    s = k*dt/dx**2
    x = np.arange(0,x_max+dx,dx) 
    t = np.arange(0,t_max+dt,dt)
    r = len(t)
    c = len(x)
    u = np.zeros([r,c])
    u[0,:] = u0
    for n in range(0,r-1):
        for j in range(1,c-1):
            u[n+1,j] = u[n,j] + s*(u[n,j-1] - 2*u[n,j] + u[n,j+1]) 
        j = c-1 
        u[n+1, j] = u[n,j] + s*(u[n,j-1] - 2*u[n,j] + u[n,j-1])
        j = 0
        u[n+1, j] = u[n,j] + s*(u[n,j+1] - 2*u[n,j] + u[n,j+1])
    return u