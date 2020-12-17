from numpy.core.numeric import flatnonzero
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

def print_x(x):
    parse_dict = {
        -1: "⋅",
        0: "@",
        1: "#"
    }
    for i in range(12**2, 0, -12):
        print(" ".join([parse_dict[int(value)] for value in x[i:i+12]]))
    print('\n')

training_set = read_csv_for_network('./dataset/physics_model/0_cropped.csv')
print(training_set[-1])
to_grid = [[int(scale(x, -2, 2, 0, 12)) for x in row] for row in training_set]
print(to_grid)
# grid = [[-1 for i in range(32)] for j in range(32)]
# grid[to_grid[0]][to_grid[1]] = 1
# grid[to_grid[2]][to_grid[3]] = 1
# grid[15][15] = 1
# flat = [x for col in grid for x in col]
# print_x(flat)

# print_x()
# print(np.array(grid))
# Train the Network on Physics Generated Dataset
network = RNN(12**2, 20, 12**2)
network.nguyen_widrow()
print(len(training_set))


# error = [126615.76036610591,118642.65511415101,111429.7771726846,104850.84854805561,98803.7006440586,93208.00150002146,88002.20328697501,83140.37246511568,78589.17580729107,74325.1261588067,70332.12545419892,66599.31944074579,63119.270363114825,59886.448629907274,56896.03821470728,54143.04258963149,51621.66909035835,49324.96094772481,47244.639088936914,45371.1113116437,43693.60516710096,42200.38288620941,40879.00138039368,39716.58688206368,38700.101131531395,37816.58326179244,37053.35797949251,36398.205894262785,35839.49574625226,35366.280895977776,34968.36394902944,34636.33404873089,34361.581442888586,34136.29363789996,33953.43698221376,33806.72698956974,33690.590209029186,33600.120002043885,33531.0282138568,33479.59442733185,33442.61422616221,33417.347686021094,33401.46911380139,33393.018870530985,33390.357936814326,33392.12570673148,33397.20132994798,33404.668773004705,33413.78562783098,33423.95559555298,33434.70446665608,33445.65936960374,33456.531004225435,33467.098561592575,33477.1970217871,33486.70653072986,33495.54357168962,33503.65367081278,33511.005404049356,33517.58549308216,33523.39481495765,33528.44516375394,33532.756637379505,33536.355533448885,33539.2726630132,33541.542004117655,33543.19962995287,33544.282860862724,33544.829595430914,33544.877785076555,33544.46502614034,33543.62824356348,33542.40344903196,33540.82555836334,33538.928254737955,33536.7438918131,33534.30342312947,33531.63635825204,33528.77073565714,33525.73311103193,33522.54855741248,33519.240674718014,33515.83160759331,33512.34206767079,33508.79136267698,33505.19742872744,33501.576866097865,33497.944977756924,33494.31581077456,33490.702198743034,33487.115807700364,33483.56718105581,33480.06578835565,33476.6200722457,33473.237499028764,33469.924606934335,33466.68705730868,33463.529683870926,33460.456543347194,33457.47096448287]

error = network.train(training_set, len(training_set), epochs=20)
print(error)
print(np.array(network.w))
print(np.array(network.v))
with open("w.txt","w") as txt_file:
    for row in network.w:
        txt_file.write(str(row))

with open("v.txt","w") as txt_file:
    for row in network.v:
        txt_file.write(str(row))
plt.plot(error)
plt.ylabel('Error')
plt.xlabel('# of Epochs')
# plt.ylim([0, 10])
plt.show()

# print(np.array(network.w))

# Generate Output from Trained Network On Singular Dataset
network_output = [[] for i in range(6)]
x = to_grid[0]
grid = [[-1 for i in range(32)] for j in range(32)]
grid[x[0]][x[1]] = 1
grid[x[2]][x[3]] = 1
flat = [x for col in grid for x in col]
for t in range(1, len(training_set)):
    _, y = network.feed_forward(flat)
    flat = y
    print_x(y)
    network_output[0].append(0)
    network_output[1].append(0)
    network_output[2].append(y[2])
    network_output[3].append(y[3])
    network_output[4].append(y[4])
    network_output[5].append(y[5])

# write_csv(network_output, './dataset/results/0.csv')
# make_plot(*network_output)