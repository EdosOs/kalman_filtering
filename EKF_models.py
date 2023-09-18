import numpy as np

'''
created a class here to associate model more generically
if i know the dim of the state and also its profile i can associate an appropriate model
'''
def __init(self):
    pass
def range_measurement_model_2d(state , x_index , y_index):
    '''This function takes range and state x,y and returns the corresponding model matrix H'''
    state_to_meas_transform = [(state[x_index]**2 + state[y_index]**2) **.5] #h(x)
    H = np.zeros([1,len(state)] , dtype='float64')
    H[0,x_index] = state[x_index]/state_to_meas_transform
    H[0,y_index] = state[y_index]/state_to_meas_transform
    return H , state_to_meas_transform

# y = z - hx