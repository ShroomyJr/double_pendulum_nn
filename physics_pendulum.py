from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
from make_plot import *

G = 9.8  # acceleration due to gravity, in m/s^2
# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.01
t = np.arange(0.0, 10, dt)


def get_angle(x_1, y_1, x_2, y_2, x_3, y_3):
    L1 = ((x_2 - x_1)**2 + (y_2 - y_1)**2)**0.5
    L2 = ((x_2 - x_3)**2 + (y_2 - y_3)**2)**0.5

    THETA1 = np.arcsin((y_2 - y_1)/L1) + np.pi/2
    THETA2 = np.arcsin((x_3 - x_2)/L2)

    x1 = sin(THETA1)
    y1 = -cos(THETA1)

    return L1, L2, THETA1, THETA2



def physics_model(M1, M2, start_pos):

    def derivs(state, t):

        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        del_ = state[2] - state[0]
        den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
        dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
                M2*G*sin(state[2])*cos(del_) +
                M2*L2*state[3]*state[3]*sin(del_) -
                (M1 + M2)*G*sin(state[0]))/den1

        dydx[2] = state[3]

        den2 = (L2/L1)*den1
        dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
                (M1 + M2)*G*sin(state[0])*cos(del_) -
                (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
                (M1 + M2)*G*sin(state[2]))/den2

        return dydx
        
    # th1 and th2 are the initial angles (degrees)
    # L1 and L2 are the length of the pendulum arms
    L1, L2, th1, th2 = get_angle(*start_pos)
    # Initial state has angular velocity 0 (starts from rest)
    state = [th1, 0, th2, 0]
    y = integrate.odeint(derivs, state, t)
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])
    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1

    return x1, y1, x2, y2