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
from ode import acceleration_model , velocity_model , circle
from EKF_models import range_measurement_model_2d
from noise_modeling import velocity_model_noise, acceleration_model_noise
import plots
from time import time
from plots import print_updated_state
t_start = time()
# General parameters
#set filter params:
state_dim = 4
measurement_dim = 2
dt = 0.01
process_noise_intensity = 0.01 #M^2
measurement_noise_intensity = 0.01#M^2
x_index = 0
y_index = 2
use_assimilation = 1
use_prediction = 1
use_update = 1
iterations_between_updates = 1
#define simulation parameters
t_initial = 0
t_final = 50
steps = int((t_final-t_initial)/dt)
simulation_process_noise =0
simulation_input_amplitude = np.append(np.array([1 for _ in range(int(steps/2))]),np.array([-1 for _ in range(int(steps/2))]))
simulation_t_transition = [1,4,6,8]
simulation_input_amplitude = [-5,10,0,-5,0]
# can handle step , cos , sin , ramp , variable step(step with random amplitude)
simulation_input_type_x: str = 'step'
simulation_input_type_y: str = 'step'
# define agents parameters
measurement_noise_limit_agent = np.array([[0.01 , 1000000]],dtype='float64')
noise_factor_agent_coeff = 0.01
num_of_agents = 3
agent_positions = np.array([[0 , 0] ,[10, 10] , [20 , 20] , [30 , 30] , [40, 40], [50 , 50] , [5000 , 5000]])
agents = [Agent(np.array([[agent_positions[i][0], agent_positions[i][1]]]),[randn()*0, randn()*0, randn()*0] , measurement_noise_intensity , 0, id=i + 1) for i in range(num_of_agents)]


x_acc_intensity = 0.05
y_acc_intensity = 0.05

# MC parameters
agents_mc = []
experiments_arr = []
number_of_mc_runs = 2
experiments = 1
process_noise_intensity_arr = np.linspace(process_noise_intensity, process_noise_intensity*100, experiments)


# Define kalman filter properties
sigma = 1000**0.5
P = np.diag([1., 1. , 1. , 1.]) * sigma**2  # Initial estimation error covariance - P
R = eye(measurement_dim , dtype='float64') * measurement_noise_intensity #R Measurement noise covariance
F = array([[1, dt+dt**2/2 , 0 , 0],[0, 1 , 0 , 0] , [0 , 0 , 1 , dt+dt**2/2 ],[0,0,0,1]] , dtype='float64') # F the process transformation matrix
H = array([[]],dtype='float64') # H the measurements transformation matrix
B = array([[0., x_acc_intensity, 0., y_acc_intensity]] , dtype='float64').T * dt # Input matrix
G = array([[0, .0, 0, .0]] , dtype='float64').T * dt #Dynamic Model Noise


initial_condition_X = [0 ,0 ,0]
initial_condition_Y = [0 ,0 ,0]
input_mat_X = array([[0., x_acc_intensity, 0.0]] , dtype='float64')
noise_mat_X = array([[0., 0, 0.]] , dtype='float64')
input_mat_Y = array([[0., y_acc_intensity, 0.0]] , dtype='float64')
noise_mat_Y = array([[0., 0, 0.]] , dtype='float64')
print('start simulating target')
T , X = acceleration_model(t_start=t_initial , t_stop=t_final ,initial_cond=initial_condition_X,  input_type=simulation_input_type_x, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X , t_transition=simulation_t_transition)
T , Y = acceleration_model(t_start=t_initial , t_stop=t_final,initial_cond=initial_condition_Y, input_type=simulation_input_type_y, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude , dt=dt , B=input_mat_Y, G=noise_mat_Y , t_transition=simulation_t_transition)
print('done simulating target')

x_input = np.array([np.squeeze(input(simulation_input_type_x , t , simulation_input_amplitude , t_transition=simulation_t_transition)) for t in T])
y_input = np.array([np.squeeze(input(simulation_input_type_y , t , simulation_input_amplitude,t_transition=simulation_t_transition)) for t in T])

u = array([np.zeros([1 , len(x_input)])[0], x_input ,np.zeros([1 , len(y_input)])[0], y_input]).T




# define kalman for each sensor
for experiment in range(experiments):
    print(f'runing mc num:{experiment}')
    Q = velocity_model_noise(var=process_noise_intensity_arr[experiment], dt=dt)
    for run in range(number_of_mc_runs):
        for agent in agents:
            agent.measure(np.array([X[0],Y[0]]) , noise_factor=noise_factor_agent_coeff, noise_limit = measurement_noise_limit_agent)
            initial_state = array([[0 + randn() * sigma , 0., 0 + randn() * sigma , 0.]], dtype='float64').T  # Initial state [x, y]
            # initialize KF
        for agent in agents:
            agent.filter = ExtendedKalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )

        for measurement_index in range(len(agent.measurements)-1):
            for agent in agents:
                angle_flag = 1
                distance_flag = 1
                if angle_flag == 0 or distance_flag == 0:
                    measurement_dim = 1

                if use_prediction == 1:
                    #prediction
                    agent.filter.prediction()
                if agent.filter.counter % iterations_between_updates == 0:
                    if use_update == 1:#update model
                        if angle_flag == 1 and distance_flag == 1:
                            agent.filter.range_measurement_model_2d(x_index=x_index, y_index=y_index , agent = agent)
                        elif angle_flag == 0:
                            agent.filter.range_measurement_model_2d_no_angle(x_index=x_index, y_index=y_index, agent=agent)
                        elif distance_flag == 0:
                            agent.filter.range_measurement_model_2d_no_distance(x_index=x_index, y_index=y_index, agent=agent)
                    if use_update == 1:#update
                        #update
                        if angle_flag == 1 and distance_flag ==1 :
                            agent.filter.update_EKF(measurement = np.array([[agent.measurements[measurement_index], agent.measurements_angle[measurement_index]]]), R = agent.R_arr[:,:,measurement_index] ) # feeding the update with measurement cov*distance factor
                        elif angle_flag == 1 and distance_flag ==0 :
                            agent.filter.update_EKF(measurement = np.array([[agent.measurements_angle[measurement_index]]]), R = np.array([[agent.R_arr[1,1,measurement_index]]])) # feeding the update with measurement cov*distance factor
                        elif distance_flag == 1 and angle_flag == 0:
                            agent.filter.update_EKF(measurement = np.array([[agent.measurements[measurement_index]]]), R = np.array([[agent.R_arr[0,0,measurement_index]]])) # feeding the update with measurement cov*distance factor

            if agent.filter.counter % iterations_between_updates == 0:
                if use_assimilation == 1:
                    for agent in agents:
                        agent.filter.assimilate(agents)


            # rearrange data
        for agent in agents:
            if use_prediction == 1:
                agent.filter.predicted_state = agent.filter.predicted_state.reshape(len(X[0]) ,state_dim )
                agent.filter.predicted_covs = agent.filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
            if use_update == 1:
                agent.filter.R_arr = agent.filter.R_arr.reshape(int(len(agent.filter.R_arr)/measurement_dim**2),measurement_dim  ,measurement_dim )
                agent.filter.updated_covs = agent.filter.updated_covs.reshape(int(len(agent.filter.updated_covs)/state_dim**2),state_dim ,state_dim )
                agent.filter.updated_state= agent.filter.updated_state.reshape(int(len(agent.filter.updated_state)/state_dim) ,state_dim )
            # Ground thruth : X_tilde = X_real - X_estimated
            # agent.filter.estimation_error = [X[0] - agent.filter.updated_state[:, 0] , X[1] - agent.filter.updated_state[:, 1]]
            if use_assimilation == 1:
                agent.filter.assim_covs = agent.filter.assim_covs.reshape(int(len(agent.filter.assim_covs)/state_dim**2), state_dim, state_dim)
                agent.filter.assim_state = agent.filter.assim_state.reshape(int(len(agent.filter.assim_state)/state_dim), state_dim)
        agents_mc.append(deepcopy(agents))
    experiments_arr.append(deepcopy(agents_mc))
    agents_mc=[]
t_end = time()
run_time = t_end-t_start
# plots.print_updated_state(mc_number=0 , state_index= 0 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)

def plot_mc_estimation_error(mc_number, state_index, agent_idx, agents_mc, simulation_time
                       , simulation_measurement,number_of_mc_runs,Q_arr , experiments):
    mean_state = 0
    theoretical_P_arr = []
    estimation_error_arr = []
    for experiment in range(len(experiments)) :
        # mean_state += (squeeze(experiments[experiment][run][agent_idx].filter.predicted_state)[:,state_index])/number_of_mc_runs
        theoretical_P_arr.append(np.array([experiments[experiment][run][agent_idx].filter.predicted_covs[:, state_index, state_index] ** .5 for run in range(number_of_mc_runs)]))
        estimation_error_arr.append(np.array([squeeze(experiments[experiment][run][agent_idx].filter.predicted_state)[:,state_index] - simulation_measurement for run in range(number_of_mc_runs)]))

    # ax = plt.figure().add_subplot()
    # ax.plot(simulation_time,mean_state - simulation_measurement, 'r')
    # ax.plot(simulation_time,
    #         squeeze(agents_mc[mc_number][agent_idx].filter.predicted_state)[:, state_index] - simulation_measurement, 'b')
    # plt.plot(simulation_time,
    #          agents_mc[mc_number][agent_idx].filter.predicted_covs[:, state_index, state_index] ** .5, '--k')
    # plt.plot(simulation_time,
    #          -agents_mc[mc_number][agent_idx].filter.predicted_covs[:, state_index, state_index] ** .5, '--k')
    # plt.legend(['estimation error MC' , 'estimation error single' , 'i sigma envalope'])
    # plt.title(f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
    # f'{agents_mc[mc_number][agent_idx].position[0, 1]}) Estimation Error Vs. MC Mean Estimation Error')
    # plt.show()

    theoretical_P_mean = [np.mean(theoretical_P,axis=0 ) for theoretical_P in theoretical_P_arr]
    MC_P_std = [np.std(estimation_error,axis=0) for estimation_error in estimation_error_arr ]
    ax = plt.figure().add_subplot()
    ax.plot(simulation_time,MC_P_std[mc_number], 'b')
    plt.plot(simulation_time,
             theoretical_P_mean[mc_number], '--k')
    plt.plot(simulation_time,
             3*theoretical_P_mean[mc_number], '--m')
    plt.plot(simulation_time,
             -theoretical_P_mean[mc_number], '--k')
    plt.plot(simulation_time,
             -3*theoretical_P_mean[mc_number], '--m')
    plt.legend(['MC calculated STD', '1 Sigma envelope'])
    plt.title(f'agent MC calculated estimation error vs. theoretically calculated estimation error')
    plt.show()

    MSE = [np.mean((theoretical_P_mean[experiment] - MC_P_std[experiment]) ** 2) for experiment in range(len(experiments))]
    ax = plt.figure().add_subplot()
    ax.plot(Q_arr,MSE, 'b')
    plt.legend(['MC calculated STD'])
    plt.title(f'MSE error as a function of process noise intensity')

    plt.show()
    return MC_P_std , theoretical_P_mean , MSE

mmm , mm,mmmm = plot_mc_estimation_error(mc_number=1,state_index = 0,agent_idx = 0, agents_mc=agents_mc ,simulation_time = T ,simulation_measurement = X[0]  ,number_of_mc_runs= number_of_mc_runs,Q_arr = process_noise_intensity_arr , experiments = experiments_arr)
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
plots.print_predicted_covariance(mc_number=0 , state_index= 0 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=X[0]  , simulation_time=T)

plots.plot_estimation_error(mc_number=0 , state_index= 0 , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement=X[0] , simulation_time=T)
plots.print_predicted_state(mc_number=0 , state_index= 2 , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement=Y[0] , simulation_time=T)

plots.plot_R_Q(mc_number=0 , state_index= 0 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])
plots.plot_R_Q(mc_number=0 , state_index= 1 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])
plots.plot_R_Q(mc_number=0 , state_index= 2 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])
plots.plot_R_Q(mc_number=0 , state_index= 3 , agent_range=[0,1] , agents_mc=agents_mc , simulation_time=T[:-1])

plots.print_xy(mc_number=0  , agent_range=[0,10] , agents_mc=agents_mc , simulation_measurement_x=X[0,10:], simulation_measurement_y=Y[0,10:] , start_idx = 10 )
plots.print_3d(mc_number=0  , agent_range=[0,4] , agents_mc=agents_mc , simulation_measurement_x=X[0,10:], simulation_measurement_y=Y[0,10:] , simulation_time=T[10:],start_index = 10 )

mmm , mm,mmmm = plot_mc_estimation_error(mc_number=0,state_index = 0,agent_idx = 0, agents_mc=agents_mc ,simulation_time = T ,simulation_measurement = X[0]  ,number_of_mc_runs= number_of_mc_runs,Q_arr = process_noise_intensity_arr , experiments = experiments_arr)
print('done')
# T[0:-1:iterations_between_updates]
mean_state = 0
theoretical_P_mean = []

for run in range(number_of_mc_runs):
    mean_state += (squeeze(agents_mc[run][0].filter.predicted_state)[:, 0]) / number_of_mc_runs
    theoretical_P_mean.append(agents_mc[run][0].filter.predicted_covs[:, 0, 0] ** .5 )
def plot_mc_estimation_error(mc_number, state_index, agent_idx, agents_mc, simulation_time
                       , simulation_measurement,number_of_mc_runs,Q_arr , experiments):
    mean_state = 0
    theoretical_P_arr = []
    estimation_error_arr = []
    for experiment in range(experiments) :
        # mean_state += (squeeze(experiments[experiment][run][agent_idx].filter.predicted_state)[:,state_index])/number_of_mc_runs
        theoretical_P_arr.append(np.array([experiments[experiment][run][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5 for run in range(number_of_mc_runs)]))
        estimation_error_arr.append(np.array([squeeze(experiments[experiment][run][agent_idx].filter.updated_state)[:,state_index] - simulation_measurement] for run in range(number_of_mc_runs)))

    # ax = plt.figure().add_subplot()
    # ax.plot(simulation_time,mean_state - simulation_measurement, 'r')
    # ax.plot(simulation_time,
    #         squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, state_index] - simulation_measurement, 'b')
    # plt.plot(simulation_time,
    #          agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    # plt.plot(simulation_time,
    #          -agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    # plt.legend(['estimation error MC' , 'estimation error single' , 'i sigma envalope'])
    # plt.title(f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
    # f'{agents_mc[mc_number][agent_idx].position[0, 1]}) Estimation Error Vs. MC Mean Estimation Error')
    # plt.show()

    theoretical_P_mean = np.mean(theoretical_P_arr,axis=0 )
    MC_P_std = np.std(estimation_error_arr,axis=0)
    ax = plt.figure().add_subplot()
    ax.plot(simulation_time,MC_P_std, 'b')
    plt.plot(simulation_time,
             theoretical_P_mean, '--k')
    plt.plot(simulation_time,
             -theoretical_P_mean, '--k')
    plt.legend(['MC calculated STD', '1 Sigma envelope'])
    plt.title(f'agent MC calculated estimation error vs. theoretically calculated estimation error')
    plt.show()

    MSE = np.mean((theoretical_P_mean - MC_P_std) ** 2)
    ax = plt.figure().add_subplot()
    ax.plot(Q_arr,MSE, 'b')
    plt.legend(['MC calculated STD'])
    plt.title(f'MSE error as a function of process noise intensity')

    plt.show()
    return MC_P_std , theoretical_P_mean , MSE


