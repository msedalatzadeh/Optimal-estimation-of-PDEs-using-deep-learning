# Optimal Estimation using Deep Learning and Shape Optimization
This repository contains codes and results for optimal estimation of heat equation by means of shape optimization and neural networks. Consider a one-dimensional steal bar over the interval <img src="/tex/8250e61c2154c3ca2d3f307958bfd9dd.svg?invert_in_darkmode&sanitize=true" align=middle width=31.50690839999999pt height=24.65753399999998pt/>. Let <img src="/tex/9a1205e73049dcbe49e500982405ce76.svg?invert_in_darkmode&sanitize=true" align=middle width=44.832674699999984pt height=24.65753399999998pt/> be the temperature of the bar at location <img src="/tex/b22db4945452a857d35a63a3f0ea5066.svg?invert_in_darkmode&sanitize=true" align=middle width=62.362875299999985pt height=24.65753399999998pt/> and time <img src="/tex/ec2b6a3dd78e3d7ba87ab5db40c09436.svg?invert_in_darkmode&sanitize=true" align=middle width=36.07293689999999pt height=21.18721440000001pt/>. The changes in the temperature is governed by the equation:


<p align="center"><img src="/tex/886b2bd3eff5b30383d97fe61cc314d4.svg?invert_in_darkmode&sanitize=true" align=middle width=446.20697055pt height=69.0417981pt/></p>


## Forward simulation
Forward simulation involves a function that gives the output to the model given the inputs. For our specific example, the inputs are an initial temperature profile <img src="/tex/ef0794e5060a9aae4b0b7fd97eb5d804.svg?invert_in_darkmode&sanitize=true" align=middle width=47.11578629999999pt height=24.65753399999998pt/> and a sensor shape set <img src="/tex/01aeb83cd132e0e70a93f02602cf4b08.svg?invert_in_darkmode&sanitize=true" align=middle width=65.61628755pt height=24.65753399999998pt/>. The output <img src="/tex/80ff9aeba5bb75eeee655ace1f06ea28.svg?invert_in_darkmode&sanitize=true" align=middle width=44.07160889999999pt height=24.65753399999998pt/> is the temperature measured by a sensor in the set <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/>; that is, <img src="/tex/44353560de49ca7b89cdcc94681452ff.svg?invert_in_darkmode&sanitize=true" align=middle width=130.6233093pt height=24.65753399999998pt/>. 

 There are various methods to solve the heat equation and find the solution <img src="/tex/9a1205e73049dcbe49e500982405ce76.svg?invert_in_darkmode&sanitize=true" align=middle width=44.832674699999984pt height=24.65753399999998pt/> for every initial condition. We use forward-time central-space finite-difference discretization method to find the solution of the heat equation. The following Python function is created that yields the solution


```python
u = FTCS(dt,dx,t_max,x_max,k,u0)
```


The parameters of the function are defined below.

|Time increment: <img src="/tex/5a8af6f173febd968ef4c52695efcf85.svg?invert_in_darkmode&sanitize=true" align=middle width=14.492060549999989pt height=22.831056599999986pt/>|Space discretization: <img src="/tex/74380e4b90b7786c87c490f3d94f2f68.svg?invert_in_darkmode&sanitize=true" align=middle width=17.95095224999999pt height=22.831056599999986pt/>|Final time: <img src="/tex/b530365e03efcb672252555f637e9dfb.svg?invert_in_darkmode&sanitize=true" align=middle width=32.18570189999999pt height=20.221802699999984pt/>|Length of the bar: <img src="/tex/d14dd123d94b8b3fbafa97662f19e4a2.svg?invert_in_darkmode&sanitize=true" align=middle width=65.23347764999998pt height=22.831056599999986pt/>|conductivity: <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>|
|:------------------:|:-----------------------:|:--------------:|:------------------------:|:--------------:|
|         0.1       |            0.01         |       100       |            1            |      0.0003     |

For the specified parameters and the following initial condition <img src="/tex/ec636caf6078cba4046ace8418214441.svg?invert_in_darkmode&sanitize=true" align=middle width=140.88100619999997pt height=24.65753399999998pt/>, the solution is obtained by running the code

```cmd
>> .\forward_sim.py
```
The output for these parameters is 

<p align="center">
<img src="gifs/forward-sim.gif" width="400" />
</p>

## Neural-Network Estimator
A neural-network estimator is trained from some set of initial conditions to estimate the solution of the heat equation for any arbitrary initial condition. The set of initial conditions selected for training is

<p align="center"><img src="/tex/1e8cbd8acb333509a8917c9fead477d9.svg?invert_in_darkmode&sanitize=true" align=middle width=475.72906425pt height=29.58934275pt/></p>

where <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/> is changed from `1` to `N` to create new training sample. 

Training data are stored in `input.npy`. The input is an array with the shape `(m,c+1)` where `m=N*r` is the number of training data. In each column, an initial condition is followed by a number indicating a time at which the output is calculated.  The output is an array stored in `output.npy`. Let `u` be the solution to the heat equation with initial condition `u0` at time `t[s]`.

```python
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
```

We use a sequential model in Keras library of TensorFlow to build an estimator. The estimator is indicated by `model` and is constructed in four steps as follows. 

### 1. Defining the Layers
First, a sequential model is defined using the command `tensorflow.keras.Sequential`. Layers are added afterwards one by one using the command `model.add`. Three layers are often present: Input Layer, Dense Layer, Output Layer. 

```python
model = Sequential()
model.add(Dense(100, input_dim = c1, activation='tanh'))
model.add(Dense(500, activation='tanh'))
model.add(Dense(1000, activation='tanh'))
model.add(Dense(500, activation='tanh'))
model.add(Dense(c, activation='tanh'))
```
The architecture of the model is as follows

<p align="center">
<img src="figs/model-plot.png" alt="drawing" width="300"/>
</p>
An activation function can be chosen in each layer. In what follows, we will compare the following activation functions: 

Exponential Linear Unit activation function: `elu`

<p align="center"><img src="/tex/16132ced024c1f286094f5ae9dba3dcb.svg?invert_in_darkmode&sanitize=true" align=middle width=182.43261299999998pt height=49.315569599999996pt/></p>
\end{cases}

Hyperbolic Tangent activation function `tanh`.

### 2. Choosing the Comipiler
Optimization method, loss function, and performance metrics are chosen in this step.
```python
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
```

### 3. Training the Model
Sample data are fed to the model to train it. The data are divided into epochs and each spoch is further divided into batches.

```python
model.fit(input, output, epochs=1, batch_size=m)
```


### 4. Evaluating the Performance
We use some test data to evaluate the performance of the trained model. This includes feeding some sample input and output to the model and calculate the loss and performance metric.

```python
eval_result = model.evaluate(u0, u_real.reshape((1,c,r)), batch_size=1)
print('evaluation result, [loss, accuracy]=' , eval_result)
```

## Estimation (i.e. Making Predictions)
For the time being, we assume <img src="/tex/eda2a562d55167366125e1c21f91e901.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/> and <img src="/tex/4b21b432d676862d1eb707965d12e987.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/>. Estimation is

```python
u_pred=model.predict(np.asarray(u0).reshape((1,c)), batch_size=1)
```

### Choice of activation function
|`elu`|`tanh`|`relu`|
|-----|------|------|
|<p align="center">
<img src="gifs/real-prediction-elu.gif" width="400" />
</p>|<p align="center">
<img src="gifs/real-prediction-tanh.gif" width="400" />
</p>|<p align="center">
<img src="gifs/real-prediction-relu.gifs" width="400" />
</p>|

### Choice of optimizer
The activation function is fixed to `selu`.

|`Adadelta`|`SGD`|`RMSprop`|
|-----|------|------|
|<img src="gifs/real-prediction-selu-Adadelta.gif" width="400" />|<img src="gifs/real-prediction-selu-SGD.gif" width="400" />|<img src="gifs/real-prediction-selu-RMSprop.gif" width="400" />|

### Choice of loss function
The activation function and optimizer are fixed to `selu` and `Adam`, respectively.

|`mean_squared_error`|`huber_loss`|`mean_squared_logarithmic_error`|
|-----|------|------|
|<p align="center">
<img src="gifs/real-prediction-selu-Adam-mean_squared_error.gif" width="400" />
</p>|<p align="center">
<img src="gifs/real-prediction-selu-Adam-huber_loss.gif" width="400" />
</p>|<p align="center">
<img src="gifs/real-prediction-selu-Adam-mean_squared_logarithmic_error.gif" width="400" />
</p>|


## Shape Optimization
