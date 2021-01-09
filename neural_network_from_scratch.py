import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# description of neural network: it accepts 2 inputs [x,x],
# it has 2 neurons in the hidden layer
# it outputs a vector of size 1
number_of_inputs_to_neural_network = 2
number_of_neurons_in_hidden_layer = 2
number_of_outputs = 1
learning_rate = 0.1 # 0.001, 0.01 <- Magic values
reg_param = 0 # 0.001, 0.01 <- Magic values
max_iter = 2000 # 5000 <- Magic value
m = 4 # Number of training examples

# The model needs to be over fit to make predictions. Which 
np.random.seed(1)
W1 = np.random.normal(0, 1, (number_of_neurons_in_hidden_layer, number_of_inputs_to_neural_network)) # 2x2
W2 = np.random.normal(0, 1, (number_of_outputs, number_of_neurons_in_hidden_layer)) # 1x2

B1 = np.random.random((number_of_neurons_in_hidden_layer, 1)) # 2x1
B2 = np.random.random((number_of_outputs, 1)) # 1x1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)
    

def forward(x):
    something = x.reshape(x.shape[0], 1) # reshaping input to a column vector, like the following
    #[[1]
    # [1] -> this makes sure that matrix multiplication is possible.
    
    z2 = W1.dot(something) + B1 # 2x2 * 2x1 + 2x1 = 2x1
    a2 = sigmoid(z2) # 2x1

    z3 = W2.dot(a2) + B2 # 1x2 * 2x1 + 1x1 = 1x1
    a3 = sigmoid(z3)

    return a3

#These variables will store the gradients of their respective layers

dW1 = 0 # stores gradients for the weights of layer 0 
dW2 = 0 # stores gradients for the weights of layer 1

dB1 = 0 # stores gradients for the weights of the biases in layer 0
dB2 = 0 # stores gradients for the weights of the biases in layer 1

#this cost variable will be storing cost values for each iteration of stochastic gradient descent
cost = np.zeros((max_iter, 1))
for i in range(max_iter):
    c = 0

    dW1 = 0
    dW2 = 0

    dB1 = 0
    dB2 = 0
    for j in range(m):
        sys.stdout.write("\rIteration: {} and {}".format(i + 1, j + 1))

        # Forward Prop.
        a0 = X[j].reshape(X[j].shape[0], 1) # 2x1

        z1 = W1.dot(a0) + B1 # 2x2 * 2x1 + 2x1 = 2x1
        a1 = sigmoid(z1) # 2x1

        z2 = W2.dot(a1) + B2 # 1x2 * 2x1 + 1x1 = 1x1
        a2 = sigmoid(z2) # 1x1

        # Back prop.
        dz2 = a2 - y[j] # 1x1   dz2 is the error in the output layer
        dW2 += dz2 * a1.T # 1x1 .* 1x2 = 1x2 -> gradient of last layer = error_in_output_layer * activation of last layer


        dz1 = np.multiply((W2.T * dz2), sigmoid_derivative(a1)) # (2x1 * 1x1) .* 2x1 = 2x1
        dW1 += dz1.dot(a0.T) # 2x1 * 1x2 = 2x2

        dB1 += dz1 # 2x1
        dB2 += dz2 # 1x1

        #y_hat = y[j] | yi=a2
        c = c + (a2 - y[j])**2

        sys.stdout.flush() # Updating the text.
    W1 = W1 - (learning_rate * dW1)
    W2 = W2 - (learning_rate * dW2)

    B1 = B1 - (learning_rate * dB1)
    B2 = B2 - (learning_rate * dB2)
    cost[i] = c


for x in X:
    print("\n")
    print(x)
    print(forward(x))

plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()