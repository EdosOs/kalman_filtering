import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import array , diag , exp , pi
class KalmanFilter:
    def __init__(self , x0 , P , Q , R , F, B, H, u ,dt ): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        self.state = x0
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.B = B
        self.H = H
        self.u = u
        self.dt = dt
        self.predicted_state = array([] , dtype='float64')
        self.updated_state = array([] , dtype='float64')
        self.updated_covs = array([] , dtype='float64')
        self.predicted_covs = array([] , dtype='float64')
        self.R_arr = array([] , dtype='float64')
    def prediction(self): # going from x_hat(k|k) to x_hat(k+1|k)
        self.state = self.F @ self.state + self.B * self.u  # u is not present in the P_pred because it is deterministic
        self.P = self.F @ self.P @ self.F.T + self.Q # compute P(k+1|k)
        self.predicted_state = np.append(self.predicted_state , self.state.copy())
        self.predicted_covs = np.append(self.predicted_covs , self.P.copy())

        return self.state , self.state #PRIOR X(k+1|k)

    def update(self , measurement , R):    #going from x_hat(k+1|k) to x_hat(k+1|k+1)
        S = R + self.H @ self.P @ self.H.T # SYSYEM UNCERTAINTY (cov of res)
        invs =inv(S)
        K =self.P @ self.H.T @ invs # the kalman gain
        y = measurement - self.H @ self.state #the residual y
        likelihood = 1/(2*pi*S) * exp(-0.5*y.T @ invs @ y)
        self.state += K @ y #posterior
        self.P = (np.eye(len(self.state)) - K@self.H) @ self.P @ (np.eye(len(self.state)) - K@self.H).T + K @ self.R @ K.T # posterior cov (Joseph)
        # self.P = self.P - K @ self.H @ self.P # posterior cov
        self.updated_state = np.append(self.updated_state , self.state.copy())
        self.updated_covs = np.append(self.updated_covs , self.P.copy())
        self.R_arr = np.append(self.R_arr , R)
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
    # P_update = (I - K @ H) @ P_pred


class KalmanFilterInfo(KalmanFilter):
    def __init__(self , x0 , P , Q , R , F, B, H, u): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        super().__init__( x0 , P , Q , R , F, B, H, u)
    def prediction(self): # going from x_hat(k|k) to x_hat(k+1|k)
        self.state = self.F @ self.state + self.B * self.u
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.predicted_state.append(self.state), self.predicted_covs.append(np.diag(self.P))

        return self.state , self.state #PRIOR X(k+1|k)

    def update(self , measurement):    #going from x_hat(k+1|k) to x_hat(k+1|k+1)
        # S = self.R + self.H @ self.P @ self.H.T # SYSYEM UNCERTAINTY ( cov of res)
        R_inv = inv(self.R)
        K =self.P @ self.H.T @ R_inv # the kalman gain
        y = measurement - self.H @ self.state #the residual y
        P_inv = inv(self.P) + self.H.T @ R_inv @ self.H

        self.state += K @ y #posterior
        self.P = inv(P_inv)

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

class UnscentedKalmanFilter(KalmanFilter):
    '''
    steps:
    step 1: generate sigma poisns and corresponding weights Wm(weight mean) , Wc(weight cov)
    step 2: pass sima points through nonlinear model to get the sigma points in the nonlinear realm.
    step 3: compute mean and covariance using the unscented transform equations
    '''
    def __init__(self , x0 , P , Q , R , F, B, H, u,  fx , hx ,):
        super().__init__(x0 , P , Q , R , F, B, H, u)
        self.fx = fx
        self.hx = hx

    # def generate_sigma_points():
    #     weight_covs = 0
    #     weight_mean = 0
    #     sigma_points = 0
    #     return sigma_points , weight_mean , weight_covs
    def prediction(self):
        transformed_sigma , weight_mean , weight_covs = generate_sigma_points()
        self.state = sum(weight_mean * transformed_sigma)
        self.P = sum(weight_covs * (transformed_sigma - self.state) * (transformed_sigma - self.state).T) + self.Q
        self.predicted_state.append(self.state), self.predicted_covs.append(np.diag(self.P))

    def update(self , measurement):
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
    noise = pd.DataFrame(measurements.copy())
    for m in range(measurements.shape[0]):
        noise.iloc[m] = [(var[m] * np.random.randn() + mean) for i in range(len(measurements[0,:]))]
    noisy_measurements = noise+pd.DataFrame(measurements.copy())
    return array(noisy_measurements , dtype='float64')
