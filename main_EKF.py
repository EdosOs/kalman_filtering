from scipy.integrate import solve_ivp
import numpy as np
from numpy import expand_dims ,squeeze , array , diag , eye , linspace
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , KalmanFilterInfo , UnscentedKalmanFilter,ExtendedKalmanFilter , Gaussian , add_gaussian_noise
from control import input
from agent import Agent
from ode import acceleration_model , velocity_model
from EKF_models import range_measurement_model_2d
from position_models import circle , single_circle
from noise_modeling import velocity_model_noise , acceleration_model_noise


# General parameters
#set filter params:
state_dim = 4
measurement_dim = 1
dt = .1
process_noise_intensity = 100
measurement_noise_intensity = 0.1
x_index = 0
y_index = 2

#define simulation parameters
t_initial = 0
t_final = 30
simulation_process_noise =.1
simulation_input_amplitude = 1
simulation_input_type: str = 'step' # step , cos , sin , ramp

# define agents parameters
measurement_noise_limit_agent = 250000000.
noise_factor_agent_coeff = 1
num_of_agents = 4
agent_positions = np.array([[0 , 0] ,[35, 0] , [0 , 30] , [30 , 30]])
agents = [Agent(np.array([[agent_positions[i][0], agent_positions[i][1]]]),[randn()*0, randn()*0, randn()*0] , measurement_noise_intensity , 0, id=i + 1) for i in range(num_of_agents)]


x_acc_intensity = .05
y_acc_intensity = .02


# Define kalman filter properties
initial_state = array([[1, 0 ,1 , 0]] , dtype='float64').T  # Initial state [x, y]
P = eye(state_dim , dtype='float64') * 10.0  # Initial estimation error covariance - P
Q = array([[0. , .1 , 0. , .1]]).T *process_noise_intensity @ array([[0. , .1 , 0. , .1]])*dt  # Q Process noise covariance
# Q = velocity_model_noise(var=process_noise_intensity**2 ,dt=dt)
R = eye(measurement_dim , dtype='float64') * measurement_noise_intensity #R Measurement noise covariance
F = array([[1, dt , 0 , 0],[0, 1 , 0 , 0] , [0 , 0 , 1 , dt],[0,0,0,1]] , dtype='float64') # F the process transformation matrix
H = array([[1 , 0 , 0 , 0],[0 , 0, 1 ,0 ]],dtype='float64') # H the measurements transformation matrix
B = array([[0., x_acc_intensity, 0., y_acc_intensity]] , dtype='float64').T * dt # Input matrix
G = array([[0, .0, 0, .0]] , dtype='float64').T * dt #Dynamic Model Noise


initial_condition_X = [1 ,0 ,0]
initial_condition_Y = [1 ,0 ,0]
input_mat_X = array([[0., x_acc_intensity, 0.0]] , dtype='float64')
noise_mat_X = array([[0., 0, 0.]] , dtype='float64')
input_mat_Y = array([[0., y_acc_intensity, 0.0]] , dtype='float64')
noise_mat_Y = array([[0., 0, 0.]] , dtype='float64')
#Check how to start input at certain time.
T , X = acceleration_model(t_start=t_initial , t_stop=t_final ,initial_cond=initial_condition_X,  input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X)
T , Y = acceleration_model(t_start=t_initial , t_stop=t_final,initial_cond=initial_condition_Y, input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude , dt=dt , B=input_mat_Y, G=noise_mat_Y)
u = np.squeeze(input('step' , T , 1))

# X , Y = circle(t_start=t_initial,t_stop=t_final,r=10 , dt=dt)
# plt.figure()
# plt.plot(T,X[0])
# plt.show()

# define kalman for each sensor
for agent in agents:
    agent.filter = ExtendedKalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )


real_measurements =np.array([(X[0]**2+Y[0]**2)**.5]).T
for agent in agents:
    agent.measure(np.array([X[0],Y[0]]) , noise_factor=noise_factor_agent_coeff, max_noise = measurement_noise_limit_agent)
# real_measurements_noised =real_measurements + randn(len(real_measurements))*measurement_noise_intensity
for measurement_index in range(len(agent.measurements)):
    # print(measurement_index)
    for agent in agents:
        #prediction
        agent.filter.prediction()

        #update model
        H, hx = range_measurement_model_2d(state=agent.filter.state, x_index=x_index, y_index=y_index , agent = agent)
        agent.filter.model_update(H, hx)

        #update
        # distance_agent = agent.measurements[measurement_index]
        # noise_factor_agent = distance_agent*noise_factor_agent_coeff if distance_agent*noise_factor_agent_coeff < measurement_noise_limit_agent else measurement_noise_limit_agent
        agent.filter.update_EKF(agent.measurements_clean[measurement_index], agent.filter.R + agent.R_arr[measurement_index])  # feeding the update with measurement cov*distance factor
        # agent.update_agent_measurement_noise(noise_factor_agent)




    # for agent in agents:
    #     agent.filter.assimilate(agents)


    # rearrange datac
for agent in agents:

    agent.filter.updated_covs = agent.filter.updated_covs.reshape(len(X[0]),state_dim ,state_dim )
    agent.filter.updated_state= agent.filter.updated_state.reshape(len(X[0]) ,state_dim )
    agent.filter.predicted_state = agent.filter.predicted_state.reshape(len(X[0]) ,state_dim )
    agent.filter.predicted_covs = agent.filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
    agent.filter.R_arr = agent.filter.R_arr.reshape(len(X[0]) ,measurement_dim  ,measurement_dim )
    # Ground thruth : X_tilde = X_real - X_estimated
    # agent.filter.estimation_error = [X[0] - agent.filter.updated_state[:, 0] , X[1] - agent.filter.updated_state[:, 1]]
    # agent.filter.assim_covs = agent.filter.assim_covs.reshape(len(X[0]), state_dim, state_dim)
    # agent.filter.assim_state = agent.filter.assim_state.reshape(len(X[0]), state_dim)

print('done')