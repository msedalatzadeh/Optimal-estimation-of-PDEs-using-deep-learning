# Optimal Estimation of Temperature Change

Consider a one-dimensional steal bar over the interval $[0,1]$. Let $u(x,t)$ be the temperature of the bar at location $x\in [0,1]$ and time $t$. The changes in the temperature is governed by the equation:


\begin{equation}
\begin{cases}
u_{t}(x,t)=ku_{xx}(x,t),\\
u_x(0,t)=u_x(1,t)=0.
\end{cases}
\end{equation}


The initial temperature is as follows:
\begin{equation}
u(x,0)=u_0\sin (\pi x).
\end{equation}

This temperature profile looks like the following

<p align="center">
<img src="figs/u0.png" alt="drawing" width="400"/>
</p>

Runing the following code solve the equaation.

```
> py forward_sim.py
```

The changes in the temperature is according to

<p align="center">
<img src="gifs/temp.gif" alt="drawing" width="400"/>
</p>
