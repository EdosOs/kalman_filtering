import numpy as np
import pandas as pd
class KalmanFilter:
    def __init__(self , x0 , P , Q , R , F, B, H, u ): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        self.state = x0
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.B = B
        self.H = H
        self.u = u
        self.predicted_state = []
        self.updated_state = []
        self.updated_covs = []
        self.predicted_covs = []
    def prediction(self): # going from x_hat(k|k) to x_hat(k+1|k)
        self.state = self.F @ self.state + self.B * self.u
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.predicted_state.append(self.state), self.predicted_covs.append(np.diag(self.P))

        return self.state , self.state #PRIOR X(k+1|k)

    def update(self , measurement):    #going from x_hat(k+1|k) to x_hat(k+1|k+1)
        S = self.R + self.H @ self.P @ self.H.T # SYSYEM UNCERTAINTY ( cov of res)
        K =self.P @ self.H.T @ np.linalg.inv(S) # the kalman gain
        y = measurement - self.H @ self.state #the residual y
        self.state += K @ y #posterior
        self.P = (np.eye(len(self.state)) - K@self.H) @ self.P @ (np.eye(len(self.state)) - K@self.H).T + K @ self.R @ K.T # posterior cov (Joseph)
        # self.P = self.P - K @ self.H @ self.P # posterior cov
        self.updated_state.append(self.state), self.updated_covs.append(np.diag(self.P))
        return self.state #POSTIERIOR X(k+1|k+1)
    def add_predicted_state(self ,predicted_state, predicted_covs):
        return self.predicted_state.append(predicted_state) , self.predicted_covs.append(predicted_covs)
    def add_updated_state(self ,updated_state, updated_covs):
        return self.updated_state.append(updated_state) , self.updated_covs.append(updated_covs)
#𝐏=(𝐈−𝐊𝐇)𝐏¯(𝐈−𝐊𝐇)𝖳+𝐊𝐑𝐊𝖳 Joseph
    # predict
    # x = F @ x
    # P = F @ P @ F.T + Q
# update
    # S = H @ P @ H.T + R
    # K = P @ H.T @ inv(S)
    # y = z - H @ x
    # x += K @ y
    # P = P - K @ H @ P
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
    noise = pd.DataFrame(measurements.copy())
    for m in range(len(measurements)):
        noise.iloc[m] = [(var[m] * np.random.randn() + mean) for i in range(len(measurements[0,:]))]
    noisy_measurements = noise+pd.DataFrame(measurements.copy())
    return np.array(noisy_measurements)
