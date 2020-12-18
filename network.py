import random
import numpy as np
from numpy.testing._private.utils import GetPerformanceAttributes
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
        self.v = [[random.uniform(-2, 2) for j in range(p)] for i in range(n + 1)]
        self.w = [[random.uniform(-2, 2) for j in range(m)] for i in range(p + 1)]
        # Hidden-to-hidden recurrent weights
        self.u = [random.uniform(-2, 2) for i in range(p)]
        self.h = [0 for i in range(p)]

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
        return (2/(1 + np.exp(-x))) - 1

    # Expects x to already have been put through the sigmoid function
    def sigmoid_prime(self, x):
        return .5*(1 + x)*(1 - x)

    def normalize(self, x, x_i):
        # Normalize the data to within [-1, 1]
        return 2*(x_i - min(x))/(max(x) - min(x)) - 1

    def denormalize(self, x, x_i):
        return .5*(x_i - min(x))/(max(x) - min(x)) + 1

    

    def relu(self, x):
        if x < -2 or x > 2:
            return 0
        return x 

    def feed_forward(self, x):
        # print(x)
        z = []
        # Forward through hidden layer
        for j in range(self.p):
            z_in = self.v[0][j]
            z_in += sum([x[i-1]*self.v[i][j] for i in range(1, self.n + 1)])
            # Normalize Z_in to bring scale between 
            z_in = self.normalize(x, z_in)
            # Apply activation function
            z_j = self.sigmoid(z_in)
            # z_j = z_in
            # Append to Z in order to broadcast to next layer
            z.append(z_j)
        # Hidden State Interaction
        # print(z)
        h_t = np.add(z, np.dot(self.u, self.h))
        h_t = [self.sigmoid(x) for x in h_t]
        # h_t = np.dot(self.u, self.h)
        # print('ht')
        # print(h_t)
        y = []
        # Forward through outputs
        for k in range(self.m):
            y_in = self.w[0][k]
            y_in += sum([h_t[j-1]*self.w[j][k] for j in range(1, self.p + 1)])
            # Normalize Y_in
            z_in = self.normalize(z, y_in)
            # Apply activation function
            y_j = self.sigmoid(y_in)
            # y_j = y_in
            y.append(y_j)
        # Return position data from hidden & output layers
        return z, y, h_t

    def train(self, training_set, time_steps, learning_rate=0.05, epochs=1000, window=4):
        total_error = []
        for epoch in range(epochs):
            epoch_error = 0
            # The initial input value is the starting position
            x = training_set[0:window]
            # Weight corrections are computed and then averaged over the # of timesteps
            delta_w = [[0 for j in range(self.m)] for i in range(self.p + 1)]
            delta_v = [[0 for j in range(self.p)] for i in range(self.n + 1)]
            delta_h = [[0 for j in range(self.p)] for i in range(self.p)]
            # For each time step
            for time in range(window, time_steps):
                # Feedforward (passes X to the Hidden Layer)
                z, y, h_t = self.feed_forward(x)
                # Update stored hidden values
                self.h = h_t
                # Set the next input to be the output at the previous timestep
                x = training_set[time-window:time]
                # print(y)
                # Set the target value to be the value at the next time step
                t = training_set[time]
                
                #Backpropogation of error
                error = []

                # Step 6 - Calculate Error Factor on Weights Between
                #          Hidden Layer and Output
                for k in range(self.m): 
                    epoch_error += (t - y[0])**2
                    # error_k = self.distance(y[k], t[k])
                    error_k = ((t - y[k]))*self.sigmoid_prime(y[k])
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
                
                # Step 8 - Distribute error across Hidden Layer Weights
                for j in range(1, self.p):
                    error_in = error[0]*self.w[j][0]
                    error_j = error_in * self.sigmoid_prime(z[j-1])
                    # Weight correction term for hidden-to-hidden
                    for l in range(1, self.p):
                        delta_h[l][j-1] += learning_rate*error_j*h_t[l-1]
                    delta_h[0][j-1] += learning_rate*error_j
            # Step 8 - Update Weights after all timesteps
            #          Weights are batch updated
            # Average Weight corrections by timestamps
            # print(delta_w)
            # delta_w = np.divide(delta_w, time_steps)
            # delta_v = np.divide(delta_v, time_steps)
            # delta_h = np.divide(delta_h, time_steps)
            delta_w = np.clip(delta_w, -10, 10)
            delta_v = np.clip(delta_v, -10, 10)
            delta_h = np.clip(delta_h, -10, 10)
            self.w = np.add(self.w, delta_w)
            self.v = np.add(self.v, delta_v)
            self.h = np.add(self.h, delta_h)

            print('Epoch', epoch, '\tMSE', epoch_error/time_steps)
            total_error.append(epoch_error/time_steps)

            
        return total_error
