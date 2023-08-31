import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from numpy import array , abs
from kalman import add_gaussian_noise
class Agent:
    def __init__(self ,position, state , measurement_var , process_var , id , catch_flag = 0):
        self.position = position
        self.state = state
        self.measurement_var = measurement_var
        self.process_var = process_var
        self.measurements = []
        self.positions = []
        self.id = id
        self.catch_flag = catch_flag
    def measure(self , target_real_position):
        return target_real_position + array([randn() , randn()]) *self.measurement_var
    def noise_factor(self , target_real_position):
        Real_distance_X = np.abs(target_real_position[0] - self.position[0])
        Real_distance_Y = np.abs(target_real_position[1] - self.position[1])
        noise_factor = (Real_distance_X**2 + Real_distance_Y**2)**0.5/10
        return noise_factor
    def move(self , step): #defining the step is needed
        self.state[0] += step + (randn() * self.process_var)
        self.positions.append(self.state.copy())
        return self.state

        