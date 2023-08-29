import numpy as np
from numpy.linalg import norm
from numpy.random import randn
class Agent:
    def __init__(self , state , measurement_var , process_var , id , catch_flag = 0):
        self.state = state
        self.measurement_var = measurement_var
        self.process_var = process_var
        self.measurements = []
        self.positions = []
        self.id = id
        self.catch_flag = catch_flag
    def measure(self , measurement):
        distance = norm(measurement[0] - self.state[0]) + (randn() * self.measurement_var)
        direction = 1 if measurement[0] - self.state[0] > 0 else -1
        self.measurements.append(distance)
        return distance*direction
    def move(self , step): #defining the step is needed
        self.state[0] += step + (randn() * self.process_var)
        self.positions.append(self.state.copy())
        return self.state

        