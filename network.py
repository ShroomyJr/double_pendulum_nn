import random
import numpy as np
from make_plot import *

'''
----------------------------------------
    Network Implementation
----------------------------------------
-   Recurrent Network (Inputs used as outputs for next time step)
-   Back Propogation Network
-   N Input nodes
-   M Output nodes
-   P Hidden Nodes
-   Training Input: Training set is inputed 
'''
class RNN (object):
    def __init__(self, n, p, m):
        self.n = n #Input Nodes
        self.p = p #Hidden Layer Nodes
        self.m = m # Output Layer Nodes
        self.v = [[random.uniform(-0.5, 0.5) for j in range(p)] for i in range(n + 1)]
        self.w = [[random.uniform(-0.5, 0.5) for j in range(m)] for i in range(p + 1)]

    def nguyen_widrow(self):
        scale = 0.7*np.power(self.p, 1/self.n)
        for j in range(self.p):
            size = np.sqrt(sum([self.v[i][j]**2 for i in range(1, self.n + 1)]))
            for i in range(self.n + 1):
                if i == 0:
                    self.v[0][j] = random.uniform(-scale, scale)
                else:
                    self.v[i][j] = scale*self.v[i][j]/size
                    
    def sigmoid(self, x):
        x = np.clip(x, -1000, 1000)
        # x = x/10
        return (1/(1 + np.exp(-x))) - 1
    
    def normalize(self, x, x_i):
        # Normalize the data to within [-1, 1]
        return 2*(x_i - min(x))/(max(x) - min(x)) - 1

    # Expects x to already have been put through the sigmoid function
    def sigmoid_prime(self, x):
        return (1 + x)*(1 - x)

    def relu(self, x):
        return max([-2, x])

    def bounded(x, low=-2, hi=2):
        return min([max([-2, x]), 2])

    def distance(self, y, t):
        return (t - y)**2
        # return sum([(t[i] - y[i])**2 for i in range(len(y))])

    def feed_forward(self, x):
        z = []
        # Forward through hidden layer
        for j in range(self.p):
            z_in = self.v[0][j]
            z_in += sum([x[i-1]*self.v[i][j] for i in range(1, self.n + 1)])
            # Normalize Z_in to bring scale between 
            # z_in = self.normalize(x, z_in)
            # Apply activation function
            z_j = self.sigmoid(z_in)
            # z_j = z_in
            # Append to Z in order to broadcast to next layer
            z.append(z_j)
        y = []
        # Forward through outputs
        for k in range(self.m):
            y_in = self.w[0][k]
            y_in += sum([z[j-1]*self.w[j][k] for j in range(1, self.p + 1)])
            # Normalize Y_in
            # z_in = self.normalize(z, y_in)
            # Apply activation function
            y_j = self.sigmoid(y_in)
            # y_j = y_in
            y.append(y_j)
        # Return position data from hidden & output layers
        return z, y

    def train(self, training_set, time_steps, learning_rate=0.05, epochs=1000):
        total_error = []
        for epoch in range(epochs):
            epoch_error = 0
            # The initial input value is the starting position
            x = training_set[0]
            # Weight corrections are computed and then averaged over the # of timesteps
            delta_w = [[0 for j in range(self.m)] for i in range(self.p + 1)]
            delta_v = [[0 for j in range(self.p)] for i in range(self.n + 1)]
            # For each time step
            for time in range(1, time_steps):
                # Feedforward (passes X to the Hidden Layer)
                z, y = self.feed_forward(x)
                z_t, y_t = self.feed_forward(training_set[time])
                # Set the next input to be the output at the previous timestep
                x = y
                # print(y)
                # Set the target value to be the value at the next time step
                t = training_set[time]
                
                #Backpropogation of error
                error = []

                # Step 6 - Calculate Error Factor on Weights Between
                #          Hidden Layer and Output
                for k in range(self.m): 
                    epoch_error += (t[2] - y[2])**2
                    # error_k = self.distance(y[k], t[k])
                    error_k = (.5*((t[k] - y[k]) + (t[k]-y_t[k])))*self.sigmoid_prime(y[k])
                    # Calculate weight correction terms
                    for j in range(1, self.p + 1):
                        delta_w[j][k] += learning_rate*error_k*z[j-1]
                    # Calculate bias correction term
                    delta_w[0][k] += learning_rate*error_k
                    # Send error_k to the layer bellow
                    error.append(error_k)

                # Step 7 - Error from Output Layer is Distributed across Input
                #          Layer weights
                for j in range(1, 1 + self.p):
                    # Sum Delta Inputs for each hidden Unit
                    error_in = error[0]*self.w[j][0]
                    error_j = error_in * self.sigmoid_prime(z[j-1])
                    # Calculate wieght correction term for hidden-layer node
                    # i.e. V[1][0] & V[2][0] for Z_1
                    for i in range(1, self.n + 1):
                        delta_v[i][j - 1] += learning_rate*error_j*x[i-1]
                    delta_v[0][j - 1] += learning_rate*error_j

            # Average Weight corrections by timestamps
            delta_w = np.divide(delta_w, time_steps)
            delta_v = np.divide(delta_v, time_steps)
            # Step 8 - Update Weights after all timesteps
            self.w = np.add(self.w, delta_w)
            self.v = np.add(self.v, delta_v)
                
            # print('Epoch', epoch, 'MSE', epoch_error/time_steps)
            total_error.append(epoch_error/time_steps)
        return total_error
