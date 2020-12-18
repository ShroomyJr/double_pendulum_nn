# Set Up Imports
from numpy.lib.type_check import real
from util import *
from physics_pendulum import *
from make_plot import *
from network import RNN

random.seed(231)
# random.seed(420)

learning_rate = 0.001
time_steps = 20
window = 4
epochs = 500

# Initialize Network with Nguyen_Widrow weighting
x_network = RNN(window, 20, 1)
x_network.nguyen_widrow()

# Train Network on First Training Set
training_set = [float(2*(np.sin(.05*x))) for x in range(time_steps)]
error = x_network.train(training_set[:time_steps], time_steps, \
        learning_rate=learning_rate, epochs=epochs, window=window)