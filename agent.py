import numpy as np
from numpy.linalg import norm
from numpy.random import randn
from numpy import array , abs
from kalman import add_gaussian_noise
from math import sqrt , atan2
# seed_value = 42
# random.seed(seed_value)
# np.random.seed(seed_value)
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
    def measure_iteratively(self, real_measurement):
        return real_measurement + randn() * sqrt(self.measurement_var)
    def measure(self , target_real_position , noise_factor, noise_limit = np.array([[0,100]])):
        position_real = target_real_position - self.position.T
        distance_real = (position_real[0,:]**2 + position_real[1,:]**2)**0.5
        elevation_real = [atan2(position_real[1,i],position_real[0,i]) for i in range(len(position_real[0,:]))]

        measurement_var_by_distance = distance_real*noise_factor
        # limit noise
        for noise_index in range(len(measurement_var_by_distance)):
            if measurement_var_by_distance[noise_index] > noise_limit[0,1]:
                measurement_var_by_distance[noise_index] = noise_limit[0,1]
            elif measurement_var_by_distance[noise_index] < noise_limit[0,0]:
                measurement_var_by_distance[noise_index] = noise_limit[0,0]

        elevation_noised = np.array(elevation_real) + (self.baseline_measurement_var)**.5 * randn(*measurement_var_by_distance.shape)
        distance_noised = np.abs(distance_real + (self.baseline_measurement_var + measurement_var_by_distance)**.5 * randn(*measurement_var_by_distance.shape))
        R_arr = np.zeros([2,2,len(measurement_var_by_distance)])
        R_arr[0,0,:] = (measurement_var_by_distance + self.baseline_measurement_var)
        R_arr[1,1,:] = (measurement_var_by_distance + self.baseline_measurement_var)

        self.R_arr = R_arr
        self.measurements_clean = distance_real
        self.measurements = distance_noised
        self.measurements_clean_angle = elevation_real
        self.measurements_angle = elevation_noised
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

        