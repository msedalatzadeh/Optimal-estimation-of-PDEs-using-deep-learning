# Optimal Estimation using Deep Learning and Shape Optimization
This repository contains codes and results for optimal estimation of heat equation by means of shape optimization and neural networks. Consider a one-dimensional steal bar over the interval $[0,\ell]$. Let $u(x,t)$ be the temperature of the bar at location $x\in [0,1]$ and time $t>0$. The changes in the temperature is governed by the equation:


\begin{equation}
\begin{cases}
u_{t}(x,t)=ku_{xx}(x,t),\\
u_x(0,t)=u_x(\ell,t)=0,\\
u(x,0)=u_0(x).
\end{cases}
\end{equation}


## Forward simulation
Forward simulation involves a function that gives the output to the model given the inputs. For our specific example, the inputs are an initial temperature profile $u(x,0)$ and a sensor shape set $\omega\subset [0,1]$. The output $y(x,t)$ is the temperature measured by a sensor in the set $\omega$; that is, $y(x,t)=\chi_\omega u(x,t)$. 

 There are various methods to solve the heat equation and find the solution $u(x,t)$ for every initial condition. We use forward-time central-space finite-difference discretization method to find the solution of the heat equation. The following Python function is created that yields the solution


```python
u = FTCS(dt,dx,t_max,x_max,k,u0)
```


The parameters of the function are defined below.

|Time increment: $dt$|Space discretization: $dx$|Final time: $t_{max}$|Length of the bar: $x_{max}=\ell$|conductivity: $k$|
|:------------------:|:-----------------------:|:--------------:|:------------------------:|:--------------:|
|         0.1       |            0.01         |       100       |            1            |      0.0003     |

For the specified parameters and the following initial condition $u(x,0)=u_0\sin (\pi x)$, the solution is obtained by running the code

```cmd
>> .\forward_sim.py
```
The output for these parameters is 

<p align="center">
<img src="gifs/forward-sim.gif" width="400" />
</p>

## Neural-Network Estimator
A neural-network estimator is trained from some set of initial conditions to estimate the solution of the heat equation for any arbitrary initial condition. The set of initial conditions selected for training is

\begin{equation}
\begin{cases}
u_0(x)=16x^2(x-1)^2\sin(pi\omega x)
\end{cases}
\end{equation}

where $\omega$ is changed from `1` to `N` to create new training sample. 

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

\begin{equation}
\begin{cases}
x   \quad if \; x>0,\\
\alpha (e^x-1) \quad if \; x<0.
\end{equation}
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
For the time being, we assume $x_1=0$ and $x_2=1$. Estimation is

```python
u_pred=model.predict(np.asarray(u0).reshape((1,c)), batch_size=1)
```

### Choice of activation function
|`elu`|`tanh`|`relu`|
|-----|------|------|
|<img src="gifs/real-prediction-elu.gif" width="400" />|<img src="gifs/real-prediction-tanh.gif" width="400" />|<img src="gifs/real-prediction-relu.gif" width="400" />|

### Choice of optimizer
The activation function is fixed to `selu`.

|`Adadelta`|`SGD`|`RMSprop`|
|-----|------|------|
|<img src="gifs/real-prediction-selu-Adadelta.gif" width="400" />|<img src="gifs/real-prediction-selu-SGD.gif" width="400" />|<img src="gifs/real-prediction-selu-RMSprop.gif" width="400" />|

### Choice of loss function
The activation function and optimizer are fixed to `selu` and `Adam`, respectively.

|`mean_squared_error`|`huber_loss`|`mean_squared_logarithmic_error`|
|-----|------|------|
|<img src="gifs/real-prediction-selu-Adam-mean_squared_error.gif" width="400" />|<img src="gifs/real-prediction-selu-Adam-huber_loss.gif" width="400" />|<img src="gifs/real-prediction-selu-Adam-mean_squared_logarithmic_error.gif" width="400" />|



### Changing Initial Conditions 
For the activation function `selu`, optimizer `Adam`, loss function `huber_loss`, different initial conditions are tested to observe the performance of the estimator.

|$u_0(x)=x^2(x-1)^2(x-\frac{1}{2})^2$|$u_0(x)=x^2(x-1)^2(x+\frac{1}{2})^2$|$u_0(x)=x^2(x-1)^2(x-\frac{1}{4})^2$|
|-----|------|------|
|<img src="gifs/IC1.gif" width="400" />|<img src="gifs/IC2.gif" width="400" />|<img src="gifs/IC3.gif" width="400" />|

### Random Training Data
We also use random initial conditions to train the model. The random training data includes initial conditions generated with the following code

```python
# Random Initial Conditions
from random import sample, choices
m = N*r    # number of input data
input = zeros((m,c+1))
output = zeros((m,c))

m = int(x_max/dx/5)
n = 0
for itr in range(0,N):
    rand_location = sample(range(2,c-2), k = m)
    rand_temperature = choices(list(chain(range(-u_max, 0), range(1, u_max+1))), k = m)

    u0 = zeros(c+1)
    for i in range(0,m):
        u0[rand_location[i]] = rand_temperature[i]

    plt.plot(X, u0[0:c])
    u = FTCS(dt, dx, t_max, x_max, k, u0[0:c])
    for s in range(0,r):
        u0[c] = t[s]
        input[n,:] = u0
        output[n,:] = u[s,:]
        n = n+1
```
These initial conditions are depicted in the next figure

<p align="center">
<img src="figs/training-u0-random.png" width="400" />
</p>

The response of the model trained with random initial conditions are shown in the next table

|$u_0(x)=x^2(x-1)^2(x-\frac{1}{2})^2$|$u_0(x)=x^2(x-1)^2(x+\frac{1}{2})^2$|$u_0(x)=x^2(x-1)^2(x-\frac{1}{4})^2$|
|-----|------|------|
|<img src="gifs/IC1-random_data.gif" width="400" />|<img src="gifs/IC2-random_data.gif" width="400" />|<img src="gifs/IC3-random_data.gif" width="400" />|


## Shape Optimization
