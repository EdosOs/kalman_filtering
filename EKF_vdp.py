import numpy as np
from numpy import expand_dims ,squeeze , array , diag , eye , linspace
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , KalmanFilterInfo , UnscentedKalmanFilter,ExtendedKalmanFilter , Gaussian , add_gaussian_noise
from control import input , constant_input
from agent import Agent
from ode import acceleration_model , velocity_model, vdp
from noise_modeling import velocity_model_noise, acceleration_model_noise, velocity_model_noise_3x3,velocity_model_noise_2x2
from time import time
start_time = time()
state_dim = 2
measurement_dim = 1
dt = .1

#noise parameters
meas_var = np.array([0.01]) # M_squared

#define simulation parameters
t_initial = 0
t_final = 20
simulation_process_noise =.01
simulation_input_amplitude =[1, 1]
simulation_input_type: str = 'step'
acceleration_factor = 0.03

# Kalman filter initiallization
P = eye(state_dim , dtype='float64') * 1000.0  # Initial estimation error covariance - P
R = diag(meas_var)  #R Measurement noise covariance
H = array([[1 , 0]],dtype='float64') # H the measurements transformation matrix
B = array([[0., 0. ]] , dtype='float64').T * dt # Input matrix
G = array([[0., 0]] , dtype='float64').T * dt #Dynamic Model Noise
F = []
initial_state = []
C,M,K = 1,1,1
C_KF,M_KF,K_KF = 1.5,1,1.2
# simulation initiallization
initial_state_X = [1 ,0]
initial_state_Y = [0 ,0]
input_mat_X = array([[0., 0]] , dtype='float64')
noise_mat_X = array([[0., 0]] , dtype='float64')
#Check how to start input at certain time.
T , X = vdp(t_start=t_initial , t_stop=t_final ,initial_cond=initial_state_X,C = C, K = K , M = M,  input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X)
u = np.array([np.squeeze(constant_input('step' , T , 1))]).T
# plt.plot(T,X[0])
# plt.plot(T,X[1])
# plt.show()

experiments_num = 100
mc_runs = 1
experiments_arr =[]
mc_arr = []
process_noise_factor_arr = np.linspace(0.01,1,mc_runs)
# define kalman
for run in range(mc_runs):
    Q_arr = [velocity_model_noise_2x2(var=process_noise_factor ,dt=dt) for process_noise_factor in process_noise_factor_arr]
    print(f'mc run = {run}')
    for experiment in range(experiments_num):
        # noise the measurements
        real_measurements = np.array([X[0]])
        noised_measurements = real_measurements + R**0.5 @ randn(*real_measurements.shape)
        Q = Q_arr[run]
        initial_state = array([[1+randn()*1, 0]] , dtype='float64').T  # Initial state [x, y]
        filter = ExtendedKalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )
        filter.diff_measurement = (noised_measurements[0,1:] - noised_measurements[0,:-1])/dt
        for measurement in noised_measurements.T[0:-1]:
            filter.vdp_F(filter.state , C = C_KF, K = K_KF , M = M_KF,dt = dt)
            #prediction
            filter.predict_EKF()

            #update
            filter.update(expand_dims(measurement, axis=1),R)  # feeding the update with measurement cov*distance factor

        # rearrange data
        filter.updated_covs = filter.updated_covs.reshape(len(X[0]),state_dim ,state_dim )
        filter.updated_state= filter.updated_state.reshape(len(X[0]) ,state_dim )
        filter.predicted_state = filter.predicted_state.reshape(len(X[0]) ,state_dim )
        filter.predicted_covs = filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
        filter.R_arr = filter.R_arr.reshape(len(X[0])-1 ,measurement_dim  ,measurement_dim )
        experiments_arr.append(filter)
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
print('done')
state_index = 0
mc_run_index = 0
experiment_index = 0
ax = plt.figure().add_subplot()
ax.plot(T, squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, state_index] - X[state_index], 'r')
plt.plot(T, mc_arr[mc_run_index][experiment_index].updated_covs[:, state_index, state_index] ** .5, '--k')
plt.plot(T,3*mc_arr[mc_run_index][experiment_index].updated_covs[:, state_index, state_index] ** .5, '--m')
plt.plot(T, -mc_arr[mc_run_index][experiment_index].updated_covs[:, state_index, state_index] ** .5, '--k')
plt.plot(T,-3*mc_arr[mc_run_index][experiment_index].updated_covs[:, state_index, state_index] ** .5, '--m')
plt.legend(['X', '1 Sigma envelope','3 Sigma envelope'])
plt.title(f'agent  X Position (updated) Errors')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()


fig, axs = plt.subplots(3)
axs[0].plot(T, squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, 0], '--b')
axs[0].plot(T, X[0], 'r')

axs[1].plot(T, squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, 1], '--b')
axs[1].plot(T[:-1], squeeze(mc_arr[mc_run_index][experiment_index].diff_measurement), '.m',linewidth = 10)
axs[1].plot(T, X[1], 'r')

axs[2].plot(squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, 0], squeeze(mc_arr[mc_run_index][experiment_index].updated_state)[:, 1], '--b')
axs[2].plot(X[0], X[1], 'r')

axs[0].legend(['estimation', 'real'])
axs[1].legend(['estimation', 'real' , 'differentiation'])
axs[2].legend(['estimation', 'real'])

axs[0].title.set_text(f'Position estimation vs. real')
axs[1].title.set_text(f'Velocity estimation vs. real vs. differentiation of position measurement')
axs[2].title.set_text(f'Phase space')

axs[0].set_xlabel(f'time')
axs[1].set_xlabel(f'time')
axs[2].set_xlabel(f'state 1 - position')

axs[0].set_ylabel(f'amplitude')
axs[1].set_ylabel(f'amplitude')
axs[2].set_ylabel(f'state 2 - velocity')



ax = plt.figure().add_subplot()
ax.plot(T, np.squeeze(noised_measurements), 'r')
ax.plot(T, np.squeeze(real_measurements), 'b')
plt.legend(['noised simulation', 'clean simulation'])
plt.title(f'agent  measurements vs real range in time')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(T, X[0])
axs[1].plot(T, X[1])
axs[2].plot(X[0], X[1])

axs[0].set_title("X_1 - position")
axs[1].set_title("X_2 - velocity")
axs[2].set_title("X_1, X_2 - phase space")
axs[0].grid(color='k', linestyle='--', linewidth=.2)
axs[1].grid(color='k', linestyle='--', linewidth=.2)
axs[2].grid(color='k', linestyle='--', linewidth=.2)

axs[0].set_xlabel('t')
axs[0].set_ylabel('amplitude')

axs[1].set_xlabel('t')
axs[1].set_ylabel('amplitude')

axs[2].set_xlabel('position')
axs[2].set_ylabel('velocity')

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
    plt.title(f'agent MC calculated estimation error vs. theoretically calculated estimation error')
    plt.show()

    MSE = [np.mean((theoretical_P_mean[run][start_index:] - MC_P_std[run][start_index:]) ** 2) for run in range(len(mc_arr))]
    ax = plt.figure().add_subplot()
    ax.plot(Q_arr,MSE, 'b')
    plt.legend(['MC calculated STD'])
    plt.title(f'MSE error as a function of process noise intensity')

    plt.show()
    return MC_P_std , theoretical_P_mean , MSE

mmm , xxx , ppp = plot_mc_estimation_error(mc_number = 0, state_index = 0, mc_arr = mc_arr, simulation_time = T,start_index = 100
                       , simulation_measurement = X[0],number_of_experiments = experiments_num,Q_arr = process_noise_factor_arr)
