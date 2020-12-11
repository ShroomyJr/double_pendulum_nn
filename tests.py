from numpy.lib.type_check import real
from util import *
from physics_pendulum import *
from make_plot import *

# Retrieve test data
real_data = unpack_csv('./dataset/dpc_rotated/0.csv')
# # make_plot(*real_data)
write_csv(real_data, './dataset/test.csv')