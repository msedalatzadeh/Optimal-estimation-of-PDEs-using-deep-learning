# Optimal Estimation using Neural-Networks and Shape Optimization
This repository contains codes and reuslts for optimal estimation of heat equation by means of shape optimization and neural networks. Consider a one-dimensional steal bar over the interval <img src="/tex/847a53ccf6c0425b3b4a93fa63e5e1ab.svg?invert_in_darkmode&sanitize=true" align=middle width=40.63935644999999pt height=24.65753399999998pt/>. Let <img src="/tex/9d1533c6cbf31c281a27f801c252827a.svg?invert_in_darkmode&sanitize=true" align=middle width=53.96512274999999pt height=24.65753399999998pt/> be the temperature of the bar at location <img src="/tex/d8e914a701577490944d100c0e4c9564.svg?invert_in_darkmode&sanitize=true" align=middle width=71.49532334999999pt height=24.65753399999998pt/> and time <img src="/tex/2ec895532672c864b477de14a9e7cf41.svg?invert_in_darkmode&sanitize=true" align=middle width=45.205384949999996pt height=22.831056599999986pt/>. The changes in the temperature is governed by the equation:


```math
u_{t}(x,t)=ku_{xx}(x,t),\\
u_x(0,t)=u_x(\ell,t)=0,\\
u(x,0)=u_0(x).
```


## Forward simulation
Forward simulation involves a function that gives the output to the model given the inputs. For our specific example, the inputs are an initial temperature profile <img src="/tex/e625f5f2796ce0c63108299600e7ea2c.svg?invert_in_darkmode&sanitize=true" align=middle width=56.24823434999999pt height=24.65753399999998pt/> and a sensor shape set <img src="/tex/beec1d2723e2b236184ebd1a1fe4b55b.svg?invert_in_darkmode&sanitize=true" align=middle width=74.74873559999999pt height=24.65753399999998pt/>. The output <img src="/tex/3f2ce3a63bf5db536df3bb40dcdb929a.svg?invert_in_darkmode&sanitize=true" align=middle width=53.20405694999999pt height=24.65753399999998pt/> is the temperature measured by a sensor in the set <img src="/tex/d2442540953667226e65f04c65f6deab.svg?invert_in_darkmode&sanitize=true" align=middle width=19.95435419999999pt height=22.831056599999986pt/>; that is, <img src="/tex/8a174dff0a4408ad5687a6439fa841b7.svg?invert_in_darkmode&sanitize=true" align=middle width=139.75575734999998pt height=24.65753399999998pt/>. 

 There are various methods to solve the heat equation and find the solution <img src="/tex/9d1533c6cbf31c281a27f801c252827a.svg?invert_in_darkmode&sanitize=true" align=middle width=53.96512274999999pt height=24.65753399999998pt/> for every initial condition. We use forward-time central-space finite-difference discretization method to find the solution of the heat equation. The following Python function is created that yields the solution


```python
u = FTCS(dt,dx,t_max,x_max,k,u0)
```


The parameters of the function are defined below.

|Time increment: <img src="/tex/8706bd4c42629b7a8841b73ffb3388ca.svg?invert_in_darkmode&sanitize=true" align=middle width=23.62450859999999pt height=22.831056599999986pt/>|Space discretization: <img src="/tex/2d5a6a82954322565ae93f2b85141f6b.svg?invert_in_darkmode&sanitize=true" align=middle width=27.083400299999987pt height=22.831056599999986pt/>|Final time: <img src="/tex/da35e99b72019f1597dd21c1d556c5bf.svg?invert_in_darkmode&sanitize=true" align=middle width=42.14003639999999pt height=22.831056599999986pt/>|Length of the bar: <img src="/tex/98dfb3d2583fb9ba6f09be3736cdaa4e.svg?invert_in_darkmode&sanitize=true" align=middle width=74.36592569999999pt height=22.831056599999986pt/>|conductivity: <img src="/tex/fbdb696db6a1b0322aa20999d63696f2.svg?invert_in_darkmode&sanitize=true" align=middle width=18.207811049999993pt height=22.831056599999986pt/>|
|:------------------:|:-----------------------:|:--------------:|:------------------------:|:--------------:|
|         0.1       |            0.01         |       100       |            1            |      0.0003     |

For the specified parameters and the following initial condition <img src="/tex/67aeb08f793bd623d9b2de0e88cd812d.svg?invert_in_darkmode&sanitize=true" align=middle width=150.01345425pt height=24.65753399999998pt/>, the solution is obtained by running the code

```cmd
>> .\forward_sim.py
```
The output for these parameters is 
![](mp4s/forward-sim.mp4)

## Neural-Network Estimator
A neural-network estimator is trained from some set of initial conditions to estimate the solution of the heat equation for any arbitrary initial condition. The set of initial conditions selected for training is

```math
u_0(x)=16x^2(x-1)^2\sin(pi\omegax)
```

where <img src="/tex/d2442540953667226e65f04c65f6deab.svg?invert_in_darkmode&sanitize=true" align=middle width=19.95435419999999pt height=22.831056599999986pt/> is changed from `1` to `N` to create new training sample. 

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

We use a sequential model in Keras library of TensorFlow to build an estimator. The estimator is indicated by `model` and is consitruced in four steps as follows. 

### 1. Defining the Layers
First, a sequential model is defined using the comand `tensorflow.keras.Sequential`. Layers are added afterwards one by one using the command `model.add`. Three layers are often present: Input Layer, Dense Layer, Output Layer. 

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

```math
x   \quad if \; x>0,\\
\alpha (e^x-1) \quad if \; x<0.
```

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
For the time being, we assume <img src="/tex/5f8bde3c5b782d55554ed765ed3ce8eb.svg?invert_in_darkmode&sanitize=true" align=middle width=56.03873549999999pt height=22.831056599999986pt/> and <img src="/tex/f57413223cc3c0be427185ad81b7e664.svg?invert_in_darkmode&sanitize=true" align=middle width=56.03873549999999pt height=22.831056599999986pt/>. Estimation is

```python
u_pred=model.predict(np.asarray(u0).reshape((1,c)), batch_size=1)
```

### Choice of activation function
|`elu`|`tanh`|`relu`|
|-----|------|------|
|![](mp4s/real-prediction-elu.mp4)|![](mp4s/real-prediction-tanh.mp4)|![](mp4s/real-prediction-relu.mp4)

### Choice of optimizer
The activation function is fixed to `selu`.

|`Adadelta`|`SGD`|`RMSprop`|
|-----|------|------|
|![](mp4s/real-prediction-selu-Adadelta.mp4)|![](mp4s/real-prediction-selu-SGD.mp4)|![](mp4s/real-prediction-selu-RMSprop.mp4)

### Choice of loss function
The activation function and optimizer are fixed to `selu` and `Adam`, respectively.

|`mean_squared_error`|`huber_loss`|`mean_squared_logarithmic_error`|
|-----|------|------|
|![](mp4s/real-prediction-selu-Adam-mean_squared_error.mp4)|![](mp4s/real-prediction-selu-Adam-huber_loss.mp4)|![](mp4s/real-prediction-selu-Adam-mean_squared_logarithmic_error.mp4)


## Shape Optimization
