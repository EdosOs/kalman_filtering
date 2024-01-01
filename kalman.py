import numpy as np
import pandas as pd
from numpy.linalg import inv , solve
from numpy import array , diag , exp , pi
from EKF_models import range_measurement_model_2d
from copy import copy , deepcopy
from math import sqrt , atan2
from scipy.integrate import solve_ivp
# from Assimilation import assimilate
class KalmanFilter:
    def __init__(self , x0 , P , Q , R , F, B, H, u ,dt , G ): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        self.state = x0
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.B = B
        self.H = H
        self.G = G
        self.u = u
        self.dt = dt
        self.counter = 0
        self.predicted_state = array([x0] , dtype='float64')
        self.updated_state = array([x0] , dtype='float64')
        self.update_state = x0
        self.updated_covs = array([P] , dtype='float64')
        self.predicted_covs = array([P] , dtype='float64')
        self.R_arr = array([] , dtype='float64')
        self.u_arr = array([] , dtype='float64')
        self.residual_arr = array([] , dtype='float64') # Array for saving the residuals (measurement - H*predicted_state)
        self.assim_state = array([x0] , dtype='float64')
        self.assim_covs = array([P] , dtype='float64')
    def prediction(self): # going from x_hat(k|k) to x_hat(k+1|k)
        self.state = self.F @ self.state + self.B *np.array([self.u[self.counter,:].T]).T  # u is not present in the P_pred because it is deterministic
        self.P = self.F @ self.P @ self.F.T + self.Q # compute P(k+1|k)
        P_inv = inv(self.P)
        self.predicted_state = np.append(self.predicted_state , self.state.copy())
        self.predicted_covs = np.append(self.predicted_covs , self.P.copy())


        self.P_updated_inv = P_inv.copy()
        self.u_arr = np.append(self.u_arr , self.u.copy())
        self.P_pred = self.P.copy()
        self.state_pred = self.state.copy()
        self.counter+=1
        return self.state , self.state #PRIOR X(k+1|k)

    def update(self , measurement , R):    #going from x_hat(k+1|k) to x_hat(k+1|k+1)
        S = R + self.H @ self.P @ self.H.T # SYSYEM UNCERTAINTY (cov of res)
        invs =inv(S)
        K =self.P @ self.H.T @ invs # the kalman gain
        y = measurement - self.H @ self.state #the residual y

        # likelihood = 1/(2*pi*S) * exp(-0.5*y.T @ invs @ y)

        self.state += K @ y #posterior
        self.P = (np.eye(len(self.state)) - K@self.H) @ self.P @ (np.eye(len(self.state)) - K@self.H).T + K @ R @ K.T # posterior cov (Joseph)

        self.updated_state = np.append(self.updated_state , deepcopy(self.state.copy()))
        self.updated_covs = np.append(self.updated_covs , self.P.copy())
        self.R_arr = np.append(self.R_arr , R)

        return self.state #POSTIERIOR X(k+1|k+1)

class KalmanFilterInfo(KalmanFilter):
    def __init__(self , x0 , P , Q , R , F, B, H, u, dt , G): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        super().__init__( x0 , P , Q , R , F, B, H, u, dt , G)

    # def update(self , measurement , R):    #going from x_hat(k+1|k) to x_hat(k+1|k+1)
    #     '''
    #     note here P_inv is actually updated information matrix.
    #     in contrast to the covariance mode kalman filter here we need
    #     to calculate the updated covariance matrix first as it is necessary
    #     in calculaing the kalman gain (K = P_update*H_transpose*R_inverse)
    #
    #     another important thing to keep in mind is that R is an input of this function only because
    #     R is changing , we used changing R like in the article.
    #     '''
    #     R_inv = inv(R)
    #     P_inv = inv(self.P) + self.H.T @ R_inv @ self.H #update covariance
    #     self.P = inv(P_inv)
    #
    #     y = measurement - self.H @ self.state #the residual y
    #     K =self.P @ self.H.T @ R_inv # the kalman gain
    #     self.state += K @ y #posterior
    #
    #
    #     self.P_updated_inv = P_inv.copy()
    #     self.update_state = self.state.copy()
    #     self.updated_state = np.append(self.updated_state , self.state.copy())
    #     self.updated_covs = np.append(self.updated_covs , self.P.copy())
    #     self.R_arr = np.append(self.R_arr , R)
    def assimilate(self ,agents):
        P_pred_inv = np.array([inv(agent.filter.P_pred.copy()) for agent in agents])
        P_update_inv = np.array([agent.filter.P_updated_inv.copy() for agent in agents])
        P_pred_state = np.array([agent.filter.state_pred.copy() for agent in agents])
        P_update_state = np.array([agent.filter.update_state.copy() for agent in agents])

        P_inv_assim = inv(self.P_pred)  + np.sum(P_update_inv - P_pred_inv , axis=0)

        self.P =inv(P_inv_assim)
        self.state = self.P @ (inv(self.P_pred) @ self.state_pred + np.sum(P_update_inv @ P_update_state - P_pred_inv @ P_pred_state , axis=0))
        self.assim_state = np.append(self.assim_state , self.state.copy())
        self.assim_covs = np.append(self.assim_covs ,self.P.copy())

         # self.P = inv(self.P_pred)  + np.sum(agents_P_inv_updated) - np.sum(agents_P_inv_predicted)
         # self.state =self.P @ (inv(self.P_pred) @ self.state_pred +   np.array(agents_P_inv_updated)@np.array(agents_states_updated)  -  np.array(agents_P_inv_predicted) @ np.array(agents_states_predicted))
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

    '''
    Choosing the Sigma Parameters
    Van der Merwe suggests using  𝛽=2  for Gaussian problems, and  𝜅=3−𝑛
    So let's start there and vary  𝛼 I will let  𝑛=1
    to minimize the size of the arrays we need to look at and to avoid having to compute the square root of matrices.
    '''

    '''
     as  𝛼gets larger the sigma points get more spread out
     If our data was Gaussian we'd be incorporating data many standard deviations away from the mean; (big alpha)
     for nonlinear problems this is unlikely to produce good results. But suppose our distribution was not Gaussian,
     but instead had very fat tails? We might need to sample from those tails to get a good estimate,
     and hence it would make sense to make  𝜅
     larger (not 200, which was absurdly large to make the change in the sigma points stark).

        With a similar line of reasoning, suppose that our distribution has nearly no tails - the probability distribution
 looks more like an inverted parabola. In such a case we'd probably want to pull the
  sigma points in closer to the mean to avoid sampling in regions where there will never be real data.
    '''
    def __init__(self , x0 , P , Q , R , F, B, H, u):
        super().__init__(x0 , P , Q , R , F, B, H, u)

    # def generate_sigma_points():
    #     weight_covs = 0
    #     weight_mean = 0
    #     sigma_points = 0
    #     return sigma_points , weight_mean , weight_covs
    # def prediction(self):
    #     transformed_sigma , weight_mean , weight_covs = generate_sigma_points()
    #     self.state = sum(weight_mean * transformed_sigma)
    #     self.P = sum(weight_covs * (transformed_sigma - self.state) * (transformed_sigma - self.state).T) + self.Q
    #     self.predicted_state.append(self.state), self.predicted_covs.append(np.diag(self.P))

    def update(self , measurement):
        return 0
class ExtendedKalmanFilter(KalmanFilterInfo):
    def __init__(self , x0 , P , Q , R , F, B, H, u, dt , G): #P - initial covariance (changing each iter) , Q - dynamic model cov , R - measurement cov
        super().__init__( x0 , P , Q , R , F, B, H, u, dt , G)
        self.assim_state = array([] , dtype='float64')
        self.assim_covs = array([] , dtype='float64')
        self.hx = 0

    def model_update(self ,H , hx):
        self.H, self.hx = H, hx #linearizing the measurement matrix and calulating the relevant measurement from prediction

    def predict_EKF(self):

        self.state = self.F_sol # u is not present in the P_pred because it is deterministic
        self.P = self.F @ self.P @ self.F.T + self.Q # compute P(k+1|k)
        # P_inv = inv(self.P)
        self.predicted_state = np.append(self.predicted_state , self.state.copy())
        self.predicted_covs = np.append(self.predicted_covs , self.P.copy())


        # self.P_updated_inv = P_inv.copy()
        self.u_arr = np.append(self.u_arr , self.u.copy())
        self.P_pred = self.P.copy()
        self.state_pred = self.state.copy()
        self.counter+=1
        return self.state , self.state #PRIOR X(k+1|k)

    def update_EKF(self, measurement, R):
        K = self.P @ self.H.T @ np.linalg.solve(self.H @ self.P @ self.H.T + R, np.eye(R.shape[0]))
        # K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        y = K @ (measurement.T - self.hx)

        # likelihood = 1/(2*pi*S) * exp(-0.5*y.T @ invs @ y)

        self.state += y #posterior
        self.P = (np.eye(len(self.state)) - K@self.H) @ self.P @ (np.eye(len(self.state)) - K @ self.H).T + K @ R @ K.T # posterior cov (Joseph)
        P_inv = inv(self.P)


        self.updated_state = np.append(self.updated_state , self.state.copy())
        self.updated_covs = np.append(self.updated_covs , self.P.copy())
        self.R_arr = np.append(self.R_arr , R)
        self.update_state = self.state.copy()
        self.P_updated_inv = P_inv.copy()
        self.residual_arr = np.append(self.residual_arr ,(measurement.T - self.hx))
        #self.residual_arr = np.append(self.residual_arr ,(measurement.T - self.hx))

        return self.state #POSTIERIOR X(k+1|k+1)

    def range_measurement_model_2d(self, x_index, y_index, agent):
        '''This function takes range and state x,y and returns the corresponding model matrix H
        we use X_prediction as the nominal state'''
        # x = self.update_state[x_index] - agent.position[0, 0]
        # y = self.update_state[y_index] - agent.position[0, 1]
        x = self.state[x_index] - agent.position[0, 0]
        y = self.state[y_index] - agent.position[0, 1]
        #
        epsilon = 0.1
        if (x**2)**0.5 <epsilon and x>0:
            x = epsilon
        elif (x**2)**0.5 <epsilon and x<0:
            x = -epsilon
        if (y**2)**0.5 <epsilon and x>0:
            y = epsilon
        elif (y**2)**0.5 <epsilon and x<0:
            y = -epsilon

        distance_state_to_meas_transform = np.array([sqrt(x ** 2 + y ** 2)], dtype='float64')  # h(x) w
        angle_state_to_meas_transform = np.array([atan2(y, x)])
        '''
        h(x) is of size 1x2 because we have 2 measurements - h(x) = [distance , angle]
        H is of size 2x4 because we have 2 measurements and 4 states
        H = [[X/sqrt(X^2+Y^2) , 0 , Y/sqrt(X^2+Y^2) , 0
               -Y/(X^2+Y^2)   , 0 , X/(X^2+Y^2) )   , 0]]
               Jacobian matrix
        '''


        H = np.zeros([2, len(self.state)], dtype='float64')
        H[0, x_index] = x / distance_state_to_meas_transform
        H[0, y_index] = y / distance_state_to_meas_transform
        H[1, x_index] = -(y / (x**2 + y**2))
        H[1, y_index] =  (x / (x**2 + y**2))

        self.H = H
        self.hx = np.array([distance_state_to_meas_transform,angle_state_to_meas_transform])
        return H, distance_state_to_meas_transform
    def range_measurement_model_2d_no_angle(self, x_index, y_index, agent):
        x = self.state[x_index] - agent.position[0, 0]
        y = self.state[y_index] - agent.position[0, 1]

        distance_state_to_meas_transform = np.array([sqrt(x ** 2 + y ** 2)], dtype='float64')  # h(x) w


        H = np.zeros([1, len(self.state)], dtype='float64')
        H[0, x_index] = x / distance_state_to_meas_transform
        H[0, y_index] = y / distance_state_to_meas_transform

        self.H = H
        self.hx = np.array([distance_state_to_meas_transform])

    def range_measurement_model_2d_no_distance(self, x_index, y_index, agent):
        x = self.state[x_index] - agent.position[0, 0]
        y = self.state[y_index] - agent.position[0, 1]

        distance_state_to_meas_transform = np.array([sqrt(x ** 2 + y ** 2)], dtype='float64')  # h(x) w
        angle_state_to_meas_transform = np.array([atan2(y, x)])

        H = np.zeros([1, len(self.state)], dtype='float64')
        H[0, x_index] = -(y / (x**2 + y**2))
        H[0, y_index] =  (x / (x**2 + y**2))

        self.H = H
        self.hx = np.array([angle_state_to_meas_transform])
    def vdp_F(self,state,C,K,M,dt):

        f = np.array([[state[1,0]], [(-2*C/M) * (state[0,0]**2-1)*state[1,0]-K*state[0,0]/M]])
        F = np.array([[0 , 1] , [-4*C*state[0,0]*state[1,0] / M - K/M , -2*C*(state[0,0]**2 - 1)/M]])*dt + np.eye(len(state))
        self.f = f
        self.F = F

        ode_fcn = lambda T, X: [X[1], (-2 * C / M) * (X[0] ** 2 - 1) * X[1] - K * X[0] / M]
        self.F_sol = solve_ivp(ode_fcn, t_span=[self.counter * dt , (self.counter+1) * dt], y0=state[:,0],
                        t_eval=[(self.counter+1) * dt]).y

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
# self.P = (np.eye(len(self.state)) - K@self.H) @ self.P
