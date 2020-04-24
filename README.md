# Optimal Estimation of Temperature Change
Consider a one-dimensional steal bar over the interval <img src="/tex/8250e61c2154c3ca2d3f307958bfd9dd.svg?invert_in_darkmode&sanitize=true" align=middle width=31.50690839999999pt height=24.65753399999998pt/>. Let <img src="/tex/9a1205e73049dcbe49e500982405ce76.svg?invert_in_darkmode&sanitize=true" align=middle width=44.832674699999984pt height=24.65753399999998pt/> be the temperature of the bar at location <img src="/tex/b22db4945452a857d35a63a3f0ea5066.svg?invert_in_darkmode&sanitize=true" align=middle width=62.362875299999985pt height=24.65753399999998pt/> and time <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>. The changes in the temperature is governed by the equation:


<p align="center"><img src="/tex/4d66b0db5e0ddf2fefe7a11feb3bfb6e.svg?invert_in_darkmode&sanitize=true" align=middle width=172.35924089999997pt height=49.315569599999996pt/></p>


The initial temperature is as follows:
<p align="center"><img src="/tex/a25e1407341ef687bfff81d4c7782674.svg?invert_in_darkmode&sanitize=true" align=middle width=145.44722939999997pt height=16.438356pt/></p>

## Forward simulation
A specific initial temperature profile is chosen to run the forward simulation. That is,

<p align="center">
<img src="figs/u0.png" alt="drawing" width="400"/>
</p>

Consider the following parameteres

|Time increment <img src="/tex/5a8af6f173febd968ef4c52695efcf85.svg?invert_in_darkmode&sanitize=true" align=middle width=14.492060549999989pt height=22.831056599999986pt/>|Space discretization <img src="/tex/74380e4b90b7786c87c490f3d94f2f68.svg?invert_in_darkmode&sanitize=true" align=middle width=17.95095224999999pt height=22.831056599999986pt/>|Final time <img src="/tex/b530365e03efcb672252555f637e9dfb.svg?invert_in_darkmode&sanitize=true" align=middle width=32.18570189999999pt height=20.221802699999984pt/>|Length of the bar <img src="/tex/d14dd123d94b8b3fbafa97662f19e4a2.svg?invert_in_darkmode&sanitize=true" align=middle width=65.23347764999998pt height=22.831056599999986pt/>|conductivity <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>|Max temperature <img src="/tex/10898c33912164da6714fe6146100886.svg?invert_in_darkmode&sanitize=true" align=middle width=15.96281939999999pt height=14.15524440000002pt/>|
|:------------------:|:-----------------------:|:--------------:|:------------------------:|:--------------:|:-----------------:|
|         0.1       |            0.01         |       100       |            1            |      0.0003     |         5        |

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

## Neural-network estimator
Let the output <img src="/tex/80ff9aeba5bb75eeee655ace1f06ea28.svg?invert_in_darkmode&sanitize=true" align=middle width=44.07160889999999pt height=24.65753399999998pt/> indicate the temperature measured by a sensor in the interval <img src="/tex/357c53fb50db20e1dd55f74ed62e558b.svg?invert_in_darkmode&sanitize=true" align=middle width=49.97722619999999pt height=24.65753399999998pt/>. A set of random initial conditions is fed to the forward simulations to generate the training data. For the time being, we assume <img src="/tex/eda2a562d55167366125e1c21f91e901.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/> and <img src="/tex/4b21b432d676862d1eb707965d12e987.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/>. A sample initial conditions for the training is as follows:

<p align="center">
<img src="figs/u0_train.png" alt="drawing" width="400"/>
</p>


The system response to this training sample is as follows and will be collocted.

<p align="center">
<img src="gifs/temp_train.gif" alt="drawing" width="400"/>
</p>

