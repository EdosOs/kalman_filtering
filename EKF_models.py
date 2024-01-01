import numpy as np
from math import sqrt
'''
created a class here to associate model more generically
if i know the dim of the state and also its profile i can associate an appropriate model
'''
def __init(self):
    pass
def range_measurement_model_2d(state , x_index , y_index, agent):
    '''This function takes range and state x,y and returns the corresponding model matrix H'''
    x = state[x_index] - agent.position[0,0]
    y = state[y_index] - agent.position[0,1]
    state_to_meas_transform = np.array([sqrt(x**2 + y**2)], dtype='float64') #h(x) w
    H = np.zeros([1,len(state)] , dtype='float64')
    H[0,x_index] = x / state_to_meas_transform
    H[0,y_index] = y / state_to_meas_transform
    return H , state_to_meas_transform

# y = z - hx