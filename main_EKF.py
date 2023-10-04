from scipy.integrate import solve_ivp
import numpy as np
from copy import copy , deepcopy
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
from noise_modeling import velocity_model_noise,velocity_model_noise_2 , acceleration_model_noise
import plots
from plots import print_updated_state

# General parameters
#set filter params:
first_iter_flag = 0
state_dim = 4
measurement_dim = 1
dt = 0.1
process_noise_intensity = 1 #M^2
measurement_noise_intensity = 0.01#M^2
x_index = 0
y_index = 2
use_assimilation = 0
use_prediction = 1
use_update = 1

#define simulation parameters
t_initial = 0
t_final = 10
simulation_process_noise =.1
simulation_input_amplitude = 1
simulation_input_type: str = 'step' # step , cos , sin , ramp

# define agents parameters
measurement_noise_limit_agent = np.array([[0.05 , 100.]],dtype='float64')
noise_factor_agent_coeff = 0.5
num_of_agents = 6
agent_positions = np.array([[0 , 0] ,[10, 10] , [20 , 20] , [30 , 30] , [40, 40], [50 , 50] , [5000 , 5000]])
agents = [Agent(np.array([[agent_positions[i][0], agent_positions[i][1]]]),[randn()*0, randn()*0, randn()*0] , measurement_noise_intensity , 0, id=i + 1) for i in range(num_of_agents)]


x_acc_intensity = 1
y_acc_intensity = 1

# MC parameters
agents_mc = []
number_of_mc_runs = 5
process_noise_intensity_arr = np.linspace(process_noise_intensity, process_noise_intensity*100, number_of_mc_runs)

experiments = 50

# Define kalman filter properties
sigma = 1000**0.5
P = np.diag([1., 0.0001 , 1. , .0001]) * sigma**2  # Initial estimation error covariance - P
R = eye(measurement_dim , dtype='float64') * measurement_noise_intensity #R Measurement noise covariance
F = array([[1, dt , 0 , 0],[0, 1 , 0 , 0] , [0 , 0 , 1 , dt],[0,0,0,1]] , dtype='float64') # F the process transformation matrix
H = array([[]],dtype='float64') # H the measurements transformation matrix
B = array([[0., x_acc_intensity, 0., y_acc_intensity]] , dtype='float64').T * dt # Input matrix
G = array([[0, .0, 0, .0]] , dtype='float64').T * dt #Dynamic Model Noise


initial_condition_X = [0 ,0 ,0]
initial_condition_Y = [0 ,0 ,0]
input_mat_X = array([[0., x_acc_intensity, 0.0]] , dtype='float64')
noise_mat_X = array([[0., 0, 0.]] , dtype='float64')
input_mat_Y = array([[0., y_acc_intensity, 0.0]] , dtype='float64')
noise_mat_Y = array([[0., 0, 0.]] , dtype='float64')

T , X = acceleration_model(t_start=t_initial , t_stop=t_final ,initial_cond=initial_condition_X,  input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X)
T , Y = acceleration_model(t_start=t_initial , t_stop=t_final,initial_cond=initial_condition_Y, input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude , dt=dt , B=input_mat_Y, G=noise_mat_Y)
u = np.squeeze(input(simulation_input_type , T , simulation_input_amplitude))


for agent in agents:
    agent.measure(np.array([X[0],Y[0]]) , noise_factor=noise_factor_agent_coeff, noise_limit = measurement_noise_limit_agent)


# define kalman for each sensor
for run in range(number_of_mc_runs):
    print(run)
    # Q = array([[0., 1, 0., 1]]).T * process_noise_intensity_arr[run] @ array([[0., 1, 0., 1]]) * dt  # Q Process noise covariance
    # Q = array([[0., 1, 0., 1]]).T * process_noise_intensity @ array([[0., 1, 0., 1]]) * dt  # Q Process noise covariance
    Q = velocity_model_noise_2(var=process_noise_intensity_arr[run], dt=dt)
    initial_state = array([[0 + randn() * sigma , 0., 0 + randn() * sigma , 0.]], dtype='float64').T  # Initial state [x, y]
    # initialize KF
    for agent in agents:
        agent.filter = ExtendedKalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )

    for measurement_index in range(len(agent.measurements)-1):
        for agent in agents:
            # if use_update == 1 and first_iter_flag == 0 :#update model
            #     H, hx = range_measurement_model_2d(state=agent.filter.state, x_index=x_index, y_index=y_index , agent = agent)
            #     agent.filter.model_update(H = H, hx =  hx)
            #     #update
            #     agent.filter.update_EKF(measurement = agent.measurements[measurement_index], R = np.array([[agent.R_arr[measurement_index]]],dtype='float64'))  # feeding the update with measurement cov*distance factor
            #     first_iter_flag=1
            #
            #
            if use_prediction == 1:
                #prediction
                agent.filter.prediction()

            if use_update == 1:#update model
                agent.filter.range_measurement_model_2d(x_index=x_index, y_index=y_index , agent = agent)
                # agent.filter.model_update(H = H, hx =  hx)

            if use_update == 1:#update model
                #update
                agent.filter.update_EKF(measurement = agent.measurements[measurement_index], R = np.array([[agent.R_arr[measurement_index]]],dtype='float64'))  # feeding the update with measurement cov*distance factor


        if use_assimilation == 1:
            for agent in agents:
                agent.filter.assimilate(agents)


        # rearrange data
    for agent in agents:
        if use_prediction == 1:
            agent.filter.predicted_state = agent.filter.predicted_state.reshape(len(X[0]) ,state_dim )
            agent.filter.predicted_covs = agent.filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
        if use_update == 1:
            agent.filter.R_arr = agent.filter.R_arr.reshape(len(X[0])-1 ,measurement_dim  ,measurement_dim )
            agent.filter.updated_covs = agent.filter.updated_covs.reshape(len(X[0]),state_dim ,state_dim )
            agent.filter.updated_state= agent.filter.updated_state.reshape(len(X[0]) ,state_dim )
        # Ground thruth : X_tilde = X_real - X_estimated
        # agent.filter.estimation_error = [X[0] - agent.filter.updated_state[:, 0] , X[1] - agent.filter.updated_state[:, 1]]
        if use_assimilation == 1:
            agent.filter.assim_covs = agent.filter.assim_covs.reshape(len(X[0]), state_dim, state_dim)
            agent.filter.assim_state = agent.filter.assim_state.reshape(len(X[0]), state_dim)
    agents_mc.append(deepcopy(agents))
plots.print_updated_state(mc_number=0 , state_index= 0 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)
plots.print_updated_state(mc_number=0 , state_index= 1 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[1] , simulation_time=T)
plots.print_updated_state(mc_number=0 , state_index= 2 , agent_range=[0,1] , agents_mc=agents_mc , simulation_measurement=Y[0] , simulation_time=T)
plots.print_updated_state(mc_number=0 , state_index= 3 , agent_range=[0,1] , agents_mc=agents_mc , simulation_measurement=Y[3] , simulation_time=T)
plots.print_updated_covariance(mc_number=0 , state_index=0  , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)
plots.print_updated_covariance(mc_number=0 , state_index=2  , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement=Y[0] , simulation_time=T)

plots.print_residual(agents_mc=agents_mc,mc_number=0,agent_range=[0,10],simulation_time=T[:-1])
plots.print_measurement_comparison(agents_mc=agents_mc,mc_number=0,agent_range=[0,10],simulation_time=T)

plots.print_assimilated_state(mc_number=0 , state_index= 2 , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)
plots.print_assimilated_covariance(mc_number=0 , state_index= 0 , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)
#
plots.print_predicted_state(mc_number=0 , state_index= 0 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)
plots.print_predicted_state(mc_number=0 , state_index= 1 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[1] , simulation_time=T)
plots.print_predicted_covariance(mc_number=0 , state_index= 0 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[1]  , simulation_time=T)

plots.plot_estimation_error(mc_number=0 , state_index= 0 , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)

plots.plot_R_Q(mc_number=0 , state_index= 0 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])
plots.plot_R_Q(mc_number=0 , state_index= 1 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])
plots.plot_R_Q(mc_number=0 , state_index= 2 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])
plots.plot_R_Q(mc_number=0 , state_index= 3 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])

plots.print_xy(mc_number=0  , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement_x=X[0], simulation_measurement_y=Y[0] )
mmm , mm = plot_mc_estimation_error(mc_number=0,state_index = 0,agent_idx = 1, agents_mc=agents_mc ,simulation_time = T ,simulation_measurement = X[0]  ,number_of_mc_runs= number_of_mc_runs)
print('done')

mean_state = 0
theoretical_P_mean = []

for run in range(number_of_mc_runs):
    mean_state += (squeeze(agents_mc[run][0].filter.updated_state)[:, 0]) / number_of_mc_runs
    theoretical_P_mean.append(agents_mc[run][0].filter.updated_covs[:, 0, 0] ** .5 )
def plot_mc_estimation_error(mc_number, state_index, agent_idx, agents_mc, simulation_time
                       , simulation_measurement,number_of_mc_runs):
    mean_state = 0
    theoretical_P_arr = []
    estimation_error_arr = []
    for run in range(number_of_mc_runs) :
        mean_state += (squeeze(agents_mc[run][agent_idx].filter.updated_state)[:,state_index])/number_of_mc_runs
        theoretical_P_arr.append(agents_mc[run][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5 )
        estimation_error_arr.append(squeeze(agents_mc[run][agent_idx].filter.updated_state)[:,state_index] - simulation_measurement)

    ax = plt.figure().add_subplot()
    ax.plot(simulation_time,mean_state - simulation_measurement, 'r')
    ax.plot(simulation_time,
            squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, state_index] - simulation_measurement, 'b')
    plt.plot(simulation_time,
             agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.plot(simulation_time,
             -agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.legend(['estimation error MC' , 'estimation error single' , 'i sigma envalope'])
    plt.title(f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
    f'{agents_mc[mc_number][agent_idx].position[0, 1]}) Estimation Error Vs. MC Mean Estimation Error')
    plt.show()

    theoretical_P_mean = np.mean(theoretical_P_arr,axis=0 )
    ax = plt.figure().add_subplot()
    ax.plot(simulation_time,theoretical_P_mean, 'b')
    plt.plot(simulation_time,
             agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.plot(simulation_time,
             -agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.legend(['MC calculated STD', '1 Sigma envelope'])
    MC_P_std = np.std(estimation_error_arr,axis=0)
    plt.show
    return MC_P_std , theoretical_P_mean