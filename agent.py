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
        self.baseline_measurement_var = measurement_var
        self.process_var = process_var
        self.positions = []
        self.id = id
        self.catch_flag = catch_flag
        self.distance_arr = []
    def measure(self , target_real_position , noise_factor, max_noise):
        position_real = target_real_position - self.position.T
        distance_real = (position_real[0,:]**2 + position_real[1,:]**2)**0.5
        measurement_var_by_distance = distance_real*noise_factor
        distance_noised = distance_real + (self.baseline_measurement_var + measurement_var_by_distance)**.5 * randn(*measurement_var_by_distance.shape)
        self.R_arr = measurement_var_by_distance
        self.measurements_clean = distance_real
        self.measurements = distance_noised
    def calc_distance(self , target_real_position):
        Real_distance_X = target_real_position[0] - self.position[0]
        Real_distance_Y = target_real_position[1] - self.position[1]
        distance = (Real_distance_X**2 + Real_distance_Y**2)**0.5
        self.distance_arr.append(distance)
        return distance
    def update_agent_measurement_noise(self , noise_factor):
        self.measurement_var = self.baseline_measurement_var*noise_factor

    def move(self , step): #defining the step is needed
        self.state[0] += step + (randn() * self.process_var)
        self.positions.append(self.state.copy())
        return self.state

        