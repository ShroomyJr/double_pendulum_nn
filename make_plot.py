import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import Animation, FuncAnimation

"""
--------------------------------------------------------------------------------
    Make Plot
--------------------------------------------------------------------------------
    -   plots and saves an animation from the x & y positions for a set of 
        pendulum positions
    Input:  X_1, Y_1, X_2, Y_2, X_3, X_4, path
    Output: None (shows plot of provided double pendulum csv)
"""

def plot_time(i, x_1, y_1, x_2, y_2, x_3, y_3, line, ax):
    line.set_data([x_1[i], x_2[i], x_3[i]], [y_1[i], y_2[i], y_3[i]])
    return line,
    # Circles corresponding to position of weights on pendulum
    # c0 = Circle((x_1[i], y_1[i]), 10, fc="r", ec="r", zorder=10)
    # c1 = Circle((x_2[i], y_2[i]), 10, fc="g", ec="g", zorder=10)
    # c2 = Circle((x_3[i], y_3[i]), 10, fc="b", ec="b", zorder=10)
    
    # patches = [ax.add_patch(c0), ax.add_patch(c1), ax.add_patch(c2), line]
    # return patches

def make_plot(x_1, y_1, x_2, y_2, x_3, y_3, path, limit=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 500), ylim=(0, 500))
    ax.set_facecolor('black')
    line, = ax.plot([],[], 'o-', lw=2)
    fargs = (x_1[:limit], y_1[:limit], x_2[:limit], y_2[:limit], x_3[:limit], y_3[:limit], line, ax,)
    ani = FuncAnimation(fig, plot_time, np.arange(1, len(fargs[0])),
                        interval=12, blit=True, fargs=fargs)
    plt.show()
    ani.save(path, writer='imagemagick', fps=30)
    

def unpack_csv(path):
    with open(path, newline='') as csvfile:
        position_reader = csv.reader(csvfile)
        x_1, x_2, x_3 = [], [], []
        y_1, y_2, y_3 = [], [], []
        for row in position_reader:
            x_1.append(int(row[0])/5)
            y_1.append(int(row[1])/5)
            x_2.append(int(row[2])/5)
            y_2.append(int(row[3])/5)
            x_3.append(int(row[4])/5)
            y_3.append(int(row[5])/5)
        
        return x_1, y_1, x_2, y_2, x_3, y_3

x_1, y_1, x_2, y_2, x_3, y_3 = unpack_csv('./dataset/dpc_csv/0.csv')   
make_plot(x_1, y_1, x_2, y_2, x_3, y_3, './IMG/pendulum_0.gif', 300)


    