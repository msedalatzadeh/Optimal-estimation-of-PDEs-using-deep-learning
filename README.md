# Optimal Predictor Desgin
This repository contains codes and results for optimal estimation of heat equation by means of shape optimization and neural networks. 

Consider a one-dimensional heat-conductive bar over the interval <img src="/tex/8250e61c2154c3ca2d3f307958bfd9dd.svg?invert_in_darkmode&sanitize=true" align=middle width=31.50690839999999pt height=24.65753399999998pt/>. Let <img src="/tex/9a1205e73049dcbe49e500982405ce76.svg?invert_in_darkmode&sanitize=true" align=middle width=44.832674699999984pt height=24.65753399999998pt/> be the temperature of the bar at location <img src="/tex/b22db4945452a857d35a63a3f0ea5066.svg?invert_in_darkmode&sanitize=true" align=middle width=62.362875299999985pt height=24.65753399999998pt/> and time <img src="/tex/ec2b6a3dd78e3d7ba87ab5db40c09436.svg?invert_in_darkmode&sanitize=true" align=middle width=36.07293689999999pt height=21.18721440000001pt/>. The changes in the temperature is governed by the equation:


<p align="center"><img src="/tex/ec58e0cb71596ecb38a66f09478b0762.svg?invert_in_darkmode&sanitize=true" align=middle width=173.72907089999998pt height=69.0417981pt/></p>


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

## Neural-Network Estimator
A neural-network estimator is trained from some set of initial conditions to estimate the solution of the heat equation for any arbitrary initial condition. The set of initial conditions selected for training is

<p align="center"><img src="/tex/b017451545b04a8bd7b4b48a7690ecec.svg?invert_in_darkmode&sanitize=true" align=middle width=217.8406956pt height=18.312383099999998pt/></p>

where <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/> is changed from `1` to `N` to create new training sample. The training inputs are illustrated in the following figure.
<p align="center">
<img src="figs/training-u0-sinusodial.png" width="400" />
</p>

Use the function `generate_data()` to create training data. Training data are also stored in the external files `input.npy` and `output.npy`. 


## Building a model
We build a model using `keras` library of `tensorFlow`. The estimator is indicated by `model` and is consitruced in steps as follows. 

### 1. Choosing Layers 
#### Sequential Layer
A sequential layer is defined using the comand `tensorflow.keras.Sequential`. Layers are added one by one using the command `model.add`. Three layers are often present: Input Layer, Dense Layer, Output Layer. 

```python
model = Sequential()
model.add(Dense(100, input_dim = c1, activation='selu'))
model.add(Dense(500, activation='selu'))
model.add(Dense(1000, activation='selu'))
model.add(Dense(500, activation='selu'))
model.add(Dense(c, activation='selu'))
```
The architecture of this model is as follows

<p align="center">
<img src="figs/model-plot.png" alt="drawing" width="300"/>
</p>

#### RNN Layer
An RNN layer is a recurrent neural network in which the output is fed back to the network. A schematic of the network is depicted below

|  RNN schematic | current architecture|
|----------------|---------------------|
|<img src="pics/RNN.png" width="400" />|<img src="figs/RNN-model_plot.png" width="400" />|

To add a recurrent layer to a sequential model in keras, the layer `SimpleRNN` is used.

```python
model = Sequential([
    SimpleRNN(c, activation=act_function, return_sequences=True, input_shape=[None, cl])
    ])
```

The following is the simulation result for an RNN predictor

<img src="gifs/IC1-test_type4-25modes.gif" width="400" />

#### CNN Layer
A CNN layer applies various filters to a time series and yields a time series width shorter with depending on the size of its filter. A schematic of the network is depicted below

|  CNN schematic | current architecture|
|----------------|---------------------|
|<img src="pics/CNN.png" width="400" />|<img src="figs/CNN-model_plot.png" width="400" />|

To add a 1D convolutional layer to a sequential model in keras, `Conv1D` is used as follows
```python
model = Sequential([
    Conv1D(filters=c, activation=act_function, kernel_size=cl,
     strides=1, padding="same", input_shape=(1, cl))
     ])
```
The following is the simulation result for a CNN predictor

<img src="gifs/IC1-test_type3-25modes.gif" width="400" />


### 2. Choosing the Compiler
Optimization method, loss function, and performance metrics are chosen in this step.
```python
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
```

### 3. Training the Model
Sample data are fed to the model to train it. The data are divided into epochs and each epoch is further divided into batches.

```python
model.fit(input, output, epochs=1, batch_size=m)
```

### 4. Evaluating the Performance
We use some test data to evaluate the performance of the trained model. This includes feeding some sample input and output to the model and calculate the loss and performance metric.

```python
eval_result = model.evaluate(u0, u_real.reshape((1,c,r)), batch_size=1)
print('evaluation result, [loss, accuracy]=' , eval_result)
```

### 5. Estimation (i.e. Making Predictions)
For the time being, we assume <img src="/tex/eda2a562d55167366125e1c21f91e901.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/> and <img src="/tex/4b21b432d676862d1eb707965d12e987.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/>. Estimation is

```python
u_pred=model.predict(np.asarray(u0).reshape((1,c)), batch_size=1)
```

An activation function can be chosen in each layer. An activation function should reflect the original mathematical model. Optimizer and loss fuctions are to be chosen as well.

Different initial conditions are tested to observe the performance of the estimator. We have cosidered the follwing initial conditions:

<p align="center"><img src="/tex/693210974824c39a33a3e0d1cde66d77.svg?invert_in_darkmode&sanitize=true" align=middle width=234.31458105pt height=89.044395pt/></p>

## Shape Optimization
Let <img src="/tex/7f0df74987aef0c61a80a8b3c9abe4f5.svg?invert_in_darkmode&sanitize=true" align=middle width=65.61628755pt height=24.65753399999998pt/> indicate the sensor shape, and <img src="/tex/bb0eabc300d2b8c9440794da293c2bcf.svg?invert_in_darkmode&sanitize=true" align=middle width=15.867721649999991pt height=14.15524440000002pt/> be the real solution to the heat equation and <img src="/tex/b8d349ed77e433ecbf753e95fa797022.svg?invert_in_darkmode&sanitize=true" align=middle width=16.18675079999999pt height=14.15524440000002pt/> be the prediction. Consider the following linear-quadratic cost function:

<p align="center"><img src="/tex/01fa2c09389cd565bf233955c98b3ebd.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2745596pt height=138.9225717pt/></p>

where <img src="/tex/d9e7f0a84431b44b89c00d8327f06532.svg?invert_in_darkmode&sanitize=true" align=middle width=41.98181624999999pt height=24.65753399999998pt/> is a characteristic function indicating the sensor region. 

The derivative of <img src="/tex/0ede332fd160bea1eb23f70b3f4fb939.svg?invert_in_darkmode&sanitize=true" align=middle width=34.30368974999999pt height=24.65753399999998pt/> with respect to <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/> is

<p align="center"><img src="/tex/0ea39864a528fdb76846def21fffc219.svg?invert_in_darkmode&sanitize=true" align=middle width=419.5199151pt height=40.70359755pt/></p>

The function <img src="/tex/79e1942ea38a5f59b0ea7a7717baab0a.svg?invert_in_darkmode&sanitize=true" align=middle width=70.55883945pt height=24.7161288pt/> is the derivative of <img src="/tex/7bf2489b0bee2f83837b4841dc8af150.svg?invert_in_darkmode&sanitize=true" align=middle width=70.55883945pt height=24.65753399999998pt/> with respect parameter <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/>. This derivative is

<p align="center"><img src="/tex/13ab948ea6961310ed5abc8f1469a980.svg?invert_in_darkmode&sanitize=true" align=middle width=408.37913655pt height=19.48126455pt/></p>

The function <img src="/tex/17312f477a46b8f5dd476894535cdd05.svg?invert_in_darkmode&sanitize=true" align=middle width=34.91568629999999pt height=24.7161288pt/> is the gradient of the network with respect to its input, and <img src="/tex/da88d6a5d3af785da34a56c50436454a.svg?invert_in_darkmode&sanitize=true" align=middle width=40.73191484999999pt height=24.65753399999998pt/> is the gradient of the network with respect to sensor locations. Also, the function <img src="/tex/4e15bfd19ff4755f3f3c0968d8760d19.svg?invert_in_darkmode&sanitize=true" align=middle width=41.98181624999999pt height=24.7161288pt/> is the derivative of <img src="/tex/4e381b9597d1b5ac5c2853d08aa49d10.svg?invert_in_darkmode&sanitize=true" align=middle width=41.98181624999999pt height=24.65753399999998pt/> with respect to sensor location <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/>. As a result, the gradient of the cost function with respect to sensor location is 

<p align="center"><img src="/tex/0520e7efb52a735edcac6eb8c35d56bc.svg?invert_in_darkmode&sanitize=true" align=middle width=664.8944626499999pt height=40.70359755pt/></p>

We are interested in the discrete implementation of sensor shape optimization. In this implementation, we interpret <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/> as a vector with as many entries as the vector <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>. Each entry is the probability of sensor presence at each node. As an example, <img src="/tex/64d22563bf5f4a3a05b4dbb6d0a285a9.svg?invert_in_darkmode&sanitize=true" align=middle width=127.7168013pt height=24.65753399999998pt/> shows that the probability of sensor presence at the first node is 0.5, at the second node is 0.1, and so on. The characteristic function <img src="/tex/d9e7f0a84431b44b89c00d8327f06532.svg?invert_in_darkmode&sanitize=true" align=middle width=41.98181624999999pt height=24.65753399999998pt/> is then the `diag` operation on `omega`. So, the term <img src="/tex/a299f01c209b01f6f26309cd138e9ba1.svg?invert_in_darkmode&sanitize=true" align=middle width=103.04174925pt height=24.65753399999998pt/> is replaced with the matrix multiplication `diag(omega)*u_real`. Also, the derivative <img src="/tex/4e15bfd19ff4755f3f3c0968d8760d19.svg?invert_in_darkmode&sanitize=true" align=middle width=41.98181624999999pt height=24.7161288pt/> is the derivative with respect to each probability of sensor presence. For instance, the derivative with respect to probability of sensor at node 1 is `diag([1,0,0,...,0])`.  It is assumed that the gradient of the newrok with respect to omega is negligible.

### shape_optimizer.py
Use the python function `LQ_cost(u_pred,model,omega)` to calculate the cost.

```python
def LQ_cost(omega):

    omega_ = [i for i, value in enumerate(omega) if value < 0.5]

    input = array(training_data[0])
    input[:,omega_] = 0
    output = array(training_data[1])

    model.fit(input, output, batch_size=1000, epochs=4,verbose=0)

    u_pred = model.predict(matmul(u_real,diag(omega)))
    cost = (u_pred - u_real)**2
    cost = trapz(trapz(cost, dx=dx), dx=dt) 
    cost += 350*sum(omega) #(c-len(omega_))
    
    return cost
```

Keras uses the method `keras.backend.gradients` for calculating the gradient.  Use the python function `LQ_grad(omega)` to calculate the gradient

```python
def LQ_grad(omega):

    u_pred = model.predict(matmul(u_real,diag(omega)))

    grads = K.gradients(model.output, model.input)[0]
    gradient = K.function(model.input, grads)
    g = gradient(matmul(u_real,diag(omega)))

    D = 2*g*(u_pred-u_real)
     
    grad = zeros(c)
    for i in range(0,c):
        int = D[:,i]
        grad[i] = trapz(int, dx=dt) + 350
    
    return grad
```

The optimal actuator shape is then obtained by running the code

```python
res = minimize(LQ_cost,omega_0,method='trust-constr',
           jac=LQ_grad,
           bounds=bounds,
           options={'verbose': 1, 'maxiter': 100, 'disp': True},
           callback=callback_cost)
```

Use the function `animate(u_real,model,omega,FileName)` to illustrate the performance of the estimator and save the result.


|  Time evolution for optimal sensor arrangement | Cost vs iterations |
|----------------|---------------------|
|<img src="gifs/animate_step_iteration_100.gif" width="400" />|<img src="figs/cost_step_iteration_100.png" width="400" />|
|<img src="gifs/animate_IC1_iteration_100.gif" width="400" />|<img src="figs/cost_IC1_iteration_100.png" width="400" />|
|<img src="gifs/animate_IC2_iteration_100.gif" width="400" />|<img src="figs/cost_IC2_iteration_100.png" width="400" />|
|<img src="gifs/animate_IC3_iteration_100.gif" width="400" />|<img src="figs/cost_IC3_iteration_100.png" width="400" />|


|  Time evolution when <img src="/tex/73571ccc4ac7b2a64191078ad44541b5.svg?invert_in_darkmode&sanitize=true" align=middle width=48.93254684999999pt height=21.18721440000001pt/> | Time evolution when <img src="/tex/2554e901057a2cf68f0e388d824ea02b.svg?invert_in_darkmode&sanitize=true" align=middle width=57.15175619999999pt height=21.18721440000001pt/> |
|----------------|---------------------|
|<img src="gifs/animate_step_iteration_100_alpha_50.gif" width="400" />|<img src="gifs/animate_step_iteration_100_alpha_500.gif" width="400" />|

|  Time evolution when <img src="/tex/747fe3195e03356f846880df2514b93e.svg?invert_in_darkmode&sanitize=true" align=middle width=16.78467779999999pt height=14.15524440000002pt/> is random | Time evolution when <img src="/tex/747fe3195e03356f846880df2514b93e.svg?invert_in_darkmode&sanitize=true" align=middle width=16.78467779999999pt height=14.15524440000002pt/> is zero array|
|----------------|---------------------|
|<img src="gifs/animate_step_iteration_100_alpha_300_random.gif" width="400" />|<img src="gifs/animate_step_iteration_100.gif" width="400" />|