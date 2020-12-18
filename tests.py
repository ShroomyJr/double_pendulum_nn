from numpy.lib.type_check import real
from util import *
from physics_pendulum import *
from make_plot import *
from network import RNN
# Retrieve test data
# real_data = unpack_csv('./dataset/dpc_csv/0.csv')
# network_input = read_csv_for_network('./dataset/test.csv')
# print(network_input[:5])
# scaled_data = scale_data(real_data)

# start_pos = [0.004591383917239612,-0.008319249714070765,-0.41292958532786755,1.0373471062971127,0.44447954794333483,0.9665176561573177]
# physics_data = list(physics_model(1, 1, network_input[0]))
# physics_data.insert(0, [0 for i in range(len(physics_data[0]))])
# physics_data.insert(0, [0 for i in range(len(physics_data[0]))])
# make_plot(*physics_data)
# make_plot(*scaled_data)
# write_csv(physics_data, './dataset/physics_model/0.csv')
# write_csv(scaled_data,  './dataset/dpc_scaled/0.csv')

# Retrieve training data

# training_set = read_csv_for_network('./dataset/physics_model/0_cropped.csv')
# # training_set = unpack_csv('./dataset/physics_model/0.csv')
# print(training_set[0])

# # Train the Network on Physics Generated Dataset
# network = RNN(2, 20, 2)
# network.nguyen_widrow()
# print(len(training_set))

# # error = network.train(training_set, len(training_set), epochs=100)
# # print(error)
# print(np.array(network.w))
# print(np.array(network.v))
# plt.plot(training_set[4])
# plt.ylabel('Error')
# plt.xlabel('# of Epochs')
# # plt.ylim([0, 10])
# plt.show()

training_set = read_csv_for_network('./dataset/physics_model/0_cropped.csv')[:600]
training_set = [row[2] for row in training_set]

# Clip Into 6 training sets
training_set = [training_set[i*100:i*100+100] for i in range(6)]
training_set = [[[x] for x in group] for group in training_set]
print(len(training_set[0]))
# print(training_set)
x_network = RNN(1, 100, 1)
x_network.nguyen_widrow()
x_network.n
print(x_network.n)
print(x_network.v[0][1])
# Train Network on First Training Set
error = x_network.train(training_set[0], 100)
plt.plot(error)
plt.ylabel('Error')
plt.xlabel('# of Epochs')
plt.show()
"""
# print(np.array(network.w))

# Generate Output from Trained Network On Singular Dataset
network_output = [[] for i in range(6)]
x = training_set[0]
for t in range(1, len(training_set)):
    _, y = network.feed_forward(training_set[t])
    x = y
    # print(y)
    network_output[0].append(y[0])
    network_output[1].append(y[1])
    network_output[2].append(y[2])
    network_output[3].append(y[3])
    network_output[4].append(y[4])
    network_output[5].append(y[5])

# write_csv(network_output, './dataset/results/0.csv')
make_plot(*network_output)"""