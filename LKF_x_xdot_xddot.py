import numpy as np
from numpy import expand_dims ,squeeze , array , diag , eye , linspace
from numpy.random import randn , uniform
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , KalmanFilterInfo , UnscentedKalmanFilter,ExtendedKalmanFilter , Gaussian , add_gaussian_noise
from control import input , constant_input
from agent import Agent
from ode import acceleration_model , velocity_model
from noise_modeling import velocity_model_noise, acceleration_model_noise, velocity_model_noise_3x3
from time import time
start_time = time()
state_dim = 3
measurement_dim = 1
dt = .1

#noise parameters
meas_var = np.array([0.1]) # M_squared

#define simulation parameters
t_initial = 0
t_final = 30
simulation_process_noise =.01
simulation_input_amplitude =[1, 1]
simulation_input_type: str = 'step'
acceleration_factor = 0.03

# Kalman filter initiallization
P = eye(state_dim , dtype='float64') * 1000.0  # Initial estimation error covariance - P
# initial_state = array([[10, 0 , 0]] , dtype='float64').T  # Initial state [x, y]
# Q = velocity_model_noise_3x3(var=process_noise_factor ,dt=dt)
R = diag(meas_var)  #R Measurement noise covariance
F = array([[1, dt , 0],[0, 1 , dt] , [0 , 0 , 1]] , dtype='float64') # F the process transformation matrix
H = array([[1 , 0, 0]],dtype='float64') # H the measurements transformation matrix
B = array([[0., acceleration_factor,0.]] , dtype='float64').T * dt # Input matrix
G = array([[0., 0., 0.]] , dtype='float64').T * dt #Dynamic Model Noise

# simulation initiallization
initial_state_X = [0 ,0 ,0]
initial_state_Y = [0 ,0 ,0]
input_mat_X = array([[0., acceleration_factor,0.0]] , dtype='float64')
noise_mat_X = array([[0., 0, 0.]] , dtype='float64')
#Check how to start input at certain time.
T , X = acceleration_model(t_start=t_initial , t_stop=t_final ,initial_cond=initial_state_X,  input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X)

u = np.array([np.squeeze(constant_input('step' , T , 1))]).T
delay = 100
len_arr = 1
experiments_num = 1
mc_runs = 1
experiments_arr =[]
mc_arr = []
process_noise_factor_arr = np.linspace(0.01,1,mc_runs)
# define kalman
for run in range(mc_runs):
    Q_arr = [velocity_model_noise_3x3(var=process_noise_factor ,dt=dt) for process_noise_factor in process_noise_factor_arr]
    print(f'mc run = {run}')
    for experiment in range(experiments_num):
        # noise the measurements
        real_measurements = np.array([X[0]])
        noised_measurements = real_measurements + R**0.5 @ randn(*real_measurements.shape)
        Q = Q_arr[run]
        initial_state = array([[0+randn()*1000, 0 , 0]] , dtype='float64').T  # Initial state [x, y]
        filter = KalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )
        for measurement in noised_measurements.T[0:-1]:

            #prediction
            filter.prediction()
            delay = np.floor(uniform(0,200))
            #update
            if filter.counter > delay :
                filter.update(expand_dims(noised_measurements.T[filter.counter - int(delay)], axis=1),R)  # feeding the update with measurement cov*distance factor
                len_arr += 1
        # rearrange data
        filter.updated_covs = filter.updated_covs.reshape(len_arr,state_dim ,state_dim )
        filter.updated_state= filter.updated_state.reshape(len_arr ,state_dim )
        filter.predicted_state = filter.predicted_state.reshape(len(X[0]) ,state_dim )
        filter.predicted_covs = filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
        filter.R_arr = filter.R_arr.reshape(len_arr-1 ,measurement_dim  ,measurement_dim )
        experiments_arr.append(filter)
        len_arr = 1
    mc_arr.append(experiments_arr)
    experiments_arr =[]
# Ground thruth : X_tilde = X_real - X_estimated
# filter.estimation_error = [X[0] - filter.updated_state[:, 0] , X[1] - filter.updated_state[:, 1]]
# NEES (Normalized Estimated Error Squared ) : err = X_tilde.T @ P^-1 @ X_tilde
'''
The math is outside the scope of this book, but a random variable in the form  𝐱̃ 𝖳𝐏−1𝐱̃ 
is said to be chi-squared distributed with n degrees of freedom,
and thus the expected value of the sequence should be  𝑛.
Bar-Shalom [1] has an excellent discussion of this topic.
'''
end_time = time()
run_time = end_time - start_time
len_update = len(filter.updated_covs)
print('done')
state_index = 0
mc_run_index = 0
experiment_index = 0
ax = plt.figure().add_subplot()
ax.plot(T[:len_update], squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, state_index] - X[state_index,:len_update], 'r')
plt.plot(T[:len_update], mc_arr[mc_run_index][experiment_index].updated_covs[:, state_index, state_index] ** .5, '--k')
plt.plot(T[:len_update],-mc_arr[mc_run_index][experiment_index].updated_covs[:, state_index, state_index] ** .5, '--k')
plt.legend(['X', '1 Sigma envelope'])
plt.title(f' X Position (updated) Errors')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

ax = plt.figure().add_subplot()
ax.plot(T[:len_update], squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, state_index], 'r')
ax.plot(T[:len_update], X[state_index,:len_update]+0.03, '--r')
plt.legend(['X estimation', 'X real'])
plt.title(f'position - estimation and real')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

ax = plt.figure().add_subplot()
ax.plot(T, np.squeeze(noised_measurements), 'r')
ax.plot(T, np.squeeze(real_measurements), 'b')
plt.legend(['noised simulation', 'clean simulation'])
plt.title(f'  measurements vs real range in time')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

def plot_mc_estimation_error(mc_number, state_index, mc_arr, simulation_time
                       , simulation_measurement,number_of_experiments,Q_arr , start_index ):
    theoretical_P_arr = []
    estimation_error_arr = []
    for run in range(len(mc_arr)):
        # mean_state += (squeeze(experiments[experiment][run][agent_idx].predicted_state)[:,state_index])/number_of_mc_runs
        theoretical_P_arr.append(np.array([mc_arr[run][experiment].updated_covs[:, state_index, state_index] ** .5 for experiment in range(number_of_experiments)]))
        estimation_error_arr.append(np.array([squeeze(mc_arr[run][experiment].updated_state)[:,state_index] - simulation_measurement[:] for experiment in range(number_of_experiments)]))

    theoretical_P_mean = [np.mean(theoretical_P,axis=0 ) for theoretical_P in theoretical_P_arr]
    MC_P_std = [np.std(estimation_error,axis=0) for estimation_error in estimation_error_arr ]
    ax = plt.figure().add_subplot()
    ax.plot(simulation_time[:],MC_P_std[run], 'b')
    plt.plot(simulation_time[:],
             theoretical_P_mean[mc_number], '--k')

    plt.plot(simulation_time[:],
             -theoretical_P_mean[mc_number], '--k')

    plt.legend(['MC calculated STD', '1 Sigma envelope'])
    plt.title(f' MC calculated estimation error vs. theoretically calculated estimation error')
    plt.show()

    MSE = [np.mean((theoretical_P_mean[run][start_index:] - MC_P_std[run][start_index:]) ** 2) for run in range(len(mc_arr))]
    ax = plt.figure().add_subplot()
    ax.plot(Q_arr,MSE, 'b')
    plt.legend(['MC calculated MSE'])
    plt.title(f'MSE error as a function of process noise intensity')

    plt.show()
    return MC_P_std , theoretical_P_mean , MSE

mmm , xxx , ppp = plot_mc_estimation_error(mc_number = 0, state_index = 0, mc_arr = mc_arr, simulation_time = T,start_index = 100
                       , simulation_measurement = X[0],number_of_experiments = experiments_num,Q_arr = process_noise_factor_arr)
