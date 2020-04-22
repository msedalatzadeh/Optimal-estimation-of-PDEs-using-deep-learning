# Optimal Estimation of Temperature Change

Consider a one-dimensional steal bar over the interval <img src="/tex/acf5ce819219b95070be2dbeb8a671e9.svg?invert_in_darkmode&sanitize=true" align=middle width=32.87674994999999pt height=24.65753399999998pt/>. Let <img src="/tex/9a1205e73049dcbe49e500982405ce76.svg?invert_in_darkmode&sanitize=true" align=middle width=44.832674699999984pt height=24.65753399999998pt/> be the temperature of the bar at location <img src="/tex/b22db4945452a857d35a63a3f0ea5066.svg?invert_in_darkmode&sanitize=true" align=middle width=62.362875299999985pt height=24.65753399999998pt/> and time <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>. The changes in the temperature is governed by the equation:


<p align="center"><img src="/tex/6972437f89dde9c037be9766a952ace2.svg?invert_in_darkmode&sanitize=true" align=middle width=446.20697055pt height=49.315569599999996pt/></p>


The initial temperature is as follows:
<p align="center"><img src="/tex/8a5a086cab51d4108f6f351a6d9f3fd2.svg?invert_in_darkmode&sanitize=true" align=middle width=422.86057109999996pt height=16.438356pt/></p>

This temperature profile looks like the following

![alt text](figs/u0.png =50x50)

<img src="figs/u0.png" alt="drawing" width="200"/>


```
> py forward_sim.py
```

![alt text](gifs/temp.gif =50x50)

<img src="gifs/temp.gif" alt="drawing" width="200"/>


