import numpy as np
class KalmanFilter:
    def __init__(self , init_state , init_P , init_Q , init_R , dy): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        self.state = init_state
        self.P = init_P
        self.Q = init_Q
        self.R = init_R
        self.dy = dy
        #note that usually Q , R dont change (unless IMM)
    def prediction(self): # going from x_hat(k|k) to x_hat(k+1|k)
        self.state += self.dy# + u #??????????????
        self.P += self.Q
        return self.state

    def update(self , measurement):    #going from x_hat(k+1|k) to x_hat(k+1|k+1)
        y = measurement - self.state #the residual y
        kalman_gain = np.matmul(self.P , np.linalg.inv(self.R + self.P)) # the kalman gain
        self.state += np.matmul(kalman_gain , y) #posterior
        self.P = np.matmul((np.eye(1) - kalman_gain) , self.P) # posterior cov
    def add_predicted_state(self , state):
        return 0

class Gaussian:
    def __init__(self ,mean , var):
        self.mean = mean
        self.var = var
    def __add__(self , other):
        return Gaussian(self.mean + other.mean , self.var+other.var)
    def __mul__(self , other):
        mean_mul = (self.var * other.mean + other.var * self.mean) / (self.var + other.var)
        variance = (self.var * other.var) / (self.var + other.var)
        return Gaussian(mean_mul , variance)

def add_gaussian_noise(measurements , mean , var):
    for m in measurements:
        noise = [(var * np.random.randn() + mean) for i in range(len(measurements[0,:]))]
        m += noise
    return measurements
