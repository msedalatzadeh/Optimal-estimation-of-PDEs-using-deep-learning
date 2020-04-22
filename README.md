# Optimal Estimation of Temperature Change

Consider a one-dimensional steal bar over the interval <img src="/tex/8250e61c2154c3ca2d3f307958bfd9dd.svg?invert_in_darkmode&sanitize=true" align=middle width=31.50690839999999pt height=24.65753399999998pt/>. Let <img src="/tex/9a1205e73049dcbe49e500982405ce76.svg?invert_in_darkmode&sanitize=true" align=middle width=44.832674699999984pt height=24.65753399999998pt/> be the temperature of the bar at location <img src="/tex/b22db4945452a857d35a63a3f0ea5066.svg?invert_in_darkmode&sanitize=true" align=middle width=62.362875299999985pt height=24.65753399999998pt/> and time <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>. The changes in the temperature is governed by the equation:


<p align="center"><img src="/tex/d373f2f73270d4282a0dcef8022e6d9d.svg?invert_in_darkmode&sanitize=true" align=middle width=447.69100199999997pt height=49.315569599999996pt/></p>


The initial temperature is as follows:
<p align="center"><img src="/tex/8a5a086cab51d4108f6f351a6d9f3fd2.svg?invert_in_darkmode&sanitize=true" align=middle width=422.86057109999996pt height=16.438356pt/></p>

This temperature profile looks like the following

<p align="center">
<img src="figs/u0.png" alt="drawing" width="400"/>
</p>

Consider the following parameteres

|Time increment <img src="/tex/5a8af6f173febd968ef4c52695efcf85.svg?invert_in_darkmode&sanitize=true" align=middle width=14.492060549999989pt height=22.831056599999986pt/>|Space discretization <img src="/tex/74380e4b90b7786c87c490f3d94f2f68.svg?invert_in_darkmode&sanitize=true" align=middle width=17.95095224999999pt height=22.831056599999986pt/>|Final time <img src="/tex/b530365e03efcb672252555f637e9dfb.svg?invert_in_darkmode&sanitize=true" align=middle width=32.18570189999999pt height=20.221802699999984pt/>|Length of the bar <img src="/tex/d30a65b936d8007addc9c789d5a7ae49.svg?invert_in_darkmode&sanitize=true" align=middle width=6.849367799999992pt height=22.831056599999986pt/>|conductivity <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>|Max temperature <img src="/tex/10898c33912164da6714fe6146100886.svg?invert_in_darkmode&sanitize=true" align=middle width=15.96281939999999pt height=14.15524440000002pt/>|
|--------------------|-------------------------|----------------|--------------------------|----------------|-------------------|
|0.01               |  0.01             |  10        | 1                    | 0.005     | 10               |

The following function yields solution to the heat equation.

```python
x,u,r,s = FTCS(dt,dx,t_max,x_max,k,u0)
```

Runing the following code solve the equaation.

```
> py forward_sim.py
```

The changes in the temperature is according to

<p align="center">
<img src="gifs/temp.gif" alt="drawing" width="400"/>
</p>
