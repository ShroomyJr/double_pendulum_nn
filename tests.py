from numpy.lib.type_check import real
from util import *
from physics_pendulum import *
from make_plot import *

# Retrieve test data
real_data = unpack_csv('./dataset/dpc_csv/0.csv')
network_input = read_csv_for_network('./dataset/test.csv')
print(network_input[:5])
scaled_data = scale_data(real_data)
# print(scaled_data[:5])
# # make_plot(*real_data)
write_csv(scaled_data, './dataset/test.csv')
# csvs = retrieve_csvs('./dataset/dpc_csv')
# print(csvs)

# for file in csvs:
#     real_data = unpack_csv('./dataset/dpc_c/'+file)
#     write_csv(real_data, './dataset/dpc_rotated/'+file)

# r_min = min([min(real_data[0]), min(real_data[1]), \
#              min(real_data[2]), min(real_data[3]), \
#              min(real_data[4]), min(real_data[5])])
# r_max = max([max(real_data[0]), max(real_data[1]), \
#              max(real_data[2]), max(real_data[3]), \
#              max(real_data[4]), max(real_data[5])])
# t_min = -2
# t_max = 2
# print(r_min, r_max)


# scaled_data = [[scale(value, r_min, r_max, t_min, t_max) for value in col] for col in real_data]

# avg_x = sum(scaled_data[0])/len(scaled_data[0])
# avg_y = sum(scaled_data[1])/len(scaled_data[1])

# x1 = [value - avg_x for value in scaled_data[0]]
# y1 = [value + avg_x for value in scaled_data[1]]
# x2 = [value - avg_x for value in scaled_data[2]]
# y2 = [value + avg_x for value in scaled_data[3]]
# x3 = [value - avg_x for value in scaled_data[4]]
# y3 = [value + avg_x for value in scaled_data[5]]
# scaled_data=[x1, y1, x2, y2, x3, y3]
# write_csv(real_data, './dataset/dpc_scaled/0.csv')
# make_plot(x1, y1, x2, y2, x3, y3)

# start_pos = [0.004591383917239612,-0.008319249714070765,-0.41292958532786755,1.0373471062971127,0.44447954794333483,0.9665176561573177]
physics_data = list(physics_model(1, 1, network_input[0]))
physics_data.insert(0, [0 for i in range(len(physics_data[0]))])
physics_data.insert(0, [0 for i in range(len(physics_data[0]))])
make_plot(*physics_data)
# make_plot(*scaled_data)
write_csv(physics_data, './dataset/physics_model/0.csv')
# write_csv(scaled_data,  './dataset/dpc_scaled/0.csv')