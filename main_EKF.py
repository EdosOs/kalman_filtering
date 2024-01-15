from scipy.integrate import solve_ivp
import numpy as np
import alg_utils
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
import noise_modeling
import plots
from time import time
import random
from plots import print_updated_state
print('start')
t_start = time()
seed_value = 5
random.seed(seed_value)
np.random.seed(seed_value)
dtype = 'float16'

# General parameters
unit_arr= ['Position[M]' , 'Velocity[M/sec]' , 'Acceleration[M/sec^2]']

#set filter params:
# parameters to change:

number_of_mc_runs = 1
experiments = 1
use_assimilation = 1 # use data fusion algorithm
use_prediction = 1 # use the KF prediction step
use_update = 1# use the KF upadte step
iterations_between_updates = 1
assim_type = 'article' # article or mean or min_P


if use_assimilation == 0 : assim_type = 'none'
# kalman filter params
mode ='velocity'
state_dim = 6 if mode == 'acceleration' else 4 # enter KF state vector length
measurement_dim = 2 # enter KF measurement vector length
process_noise_intensity = [10 , 10] #M^2 Enter minimum and maximum noise , the range between will be divided by experiment number
measurement_noise_intensity = 0.1#
x_index = 0  # enter the relevant state index of x
y_index = 3 if mode == 'acceleration' else 2 # enter the relevant state index of y
distance_factor = 1


# simulation params
dt = .1
t_initial = 0
t_final = 30

# initialization params
# MC parameters
agents_mc = []
experiments_arr = []
process_noise_intensity_arr = np.linspace(process_noise_intensity[0], process_noise_intensity[1], experiments)
#define simulation parameters
steps = int((t_final-t_initial)/dt)
agent_distance_from_line = 0.9


simulation_process_noise =0
simulation_t_transition = [0,3,6,18,20,30] # when to change input amplitude
simulation_input_amplitude = np.dot(1, [0,3,-1.52,4.4,0])#define transition to simulation input , in simulation_input_amplitude the first and last are initial and final conditions

# can handle step , cos , sin , ramp , variable step(step with random amplitude)
simulation_input_type_x: str = 'step'
simulation_input_type_y: str = 'step'



# define agents parameters
measurement_noise_limit_agent = np.array([[0.000 , 1000]],dtype=dtype)
noise_factor_agent_coeff = 0.01 # sensor noise factor coeff (changes R as a function of distance and coeff)
num_of_agents = 5
init_agent_pos = np.linspace(0,20*distance_factor ,num_of_agents )
#set agents positions uniformally
agent_positions = np.array([])
for idx in range(num_of_agents):
    if idx%2 != 0:
        agent_positions = np.append(agent_positions , np.array([init_agent_pos[idx] , init_agent_pos[idx] + agent_distance_from_line]))
    else:
        agent_positions = np.append(agent_positions , np.array([init_agent_pos[idx] , init_agent_pos[idx] + agent_distance_from_line]))
agent_positions = np.reshape(agent_positions,[num_of_agents,2])
# agent_positions = np.array([[x,x] for x in init_agent_pos])
# agent_positions = [[5,35], [15 , 0],[5,1],[6,0],[10,17],[12,8],[10,15],[17,11],[18,23],[13,9],[20,26]]
agents = [Agent(np.array([[agent_positions[i][0], agent_positions[i][1]]]),[randn()*0, randn()*0, randn()*0] , measurement_noise_intensity , 0, id=i + 1) for i in range(num_of_agents)]


x_acc_intensity = .5
y_acc_intensity = .5

# Define kalman filter properties
sigma = 1**0.5
R = eye(measurement_dim , dtype=dtype) * measurement_noise_intensity #R Measurement noise covariance
P = np.eye(state_dim,dtype=dtype) * sigma**2  # Initial estimation error covariance - P
if mode == 'velocity':
    A = array([[0, 1 , 0 , 0],[0, 0 , 0 , 0] , [0 , 0 , 0 , 1 ],[0,0,0,0]] , dtype=dtype) # F the process transformation matrix
    F = eye(state_dim)+A*dt # F the process transformation matrix
    B = array([[0., x_acc_intensity , 0., y_acc_intensity]] , dtype=dtype).T * dt  # Input matrix
    G = array([[0, .0, 0, .0]] , dtype=dtype).T * dt #Dynamic Model Noise

elif mode == 'acceleration':
    F = array([[1, dt+dt**2/2 , dt**2/2 , 0 , 0 , 0],[0, 1 , dt+dt**2/2 , 0, 0 , 0 ],[0,0,1,0,0,0] , [0 , 0 ,0 , 1 , dt+dt**2/2, dt**2/2],[0,0,0,0,1, dt+dt**2/2],[0,0,0,0,0,1]] , dtype=dtype) # F the process transformation matrix
    B = array([[0., 0.,x_acc_intensity, 0., 0.,y_acc_intensity]] , dtype=dtype).T   # Input matrix
    G = array([[0, .0, 0, .0, 0, 0]] , dtype=dtype).T * dt #Dynamic Model Noise
H = array([[]],dtype=dtype) # H the measurements transformation matrix

initial_condition_X = [0 ,0 ,0]
initial_condition_Y = [0 ,0 ,0]
input_mat_X = array([[0., x_acc_intensity, 0.0]] , dtype=dtype)
noise_mat_X = array([[0., 0, 0.]] , dtype=dtype)
input_mat_Y = array([[0., y_acc_intensity, 0.0]] , dtype=dtype)
noise_mat_Y = array([[0., 0, 0.]] , dtype=dtype)
print('start simulating target')
T , X = acceleration_model(t_start=t_initial , t_stop=t_final ,initial_cond=initial_condition_X,  input_type=simulation_input_type_x, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X , t_transition=simulation_t_transition)
T , Y = acceleration_model(t_start=t_initial , t_stop=t_final,initial_cond=initial_condition_Y, input_type=simulation_input_type_y, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude , dt=dt , B=input_mat_Y, G=noise_mat_Y , t_transition=simulation_t_transition)
print('done simulating target')
x_input = np.array([np.squeeze(input(simulation_input_type_x , t , simulation_input_amplitude , t_transition=simulation_t_transition)) for t in T])
y_input = np.array([np.squeeze(input(simulation_input_type_y , t , simulation_input_amplitude,t_transition=simulation_t_transition)) for t in T])

if mode == 'velocity':
    u = array([np.zeros([1 , len(x_input)])[0], x_input ,np.zeros([1 , len(y_input)])[0], y_input]).T
    X[2,:] = u[:,1]*x_acc_intensity
    Y[2,:] = u[:,3]*y_acc_intensity
elif mode == 'acceleration':
    u = array([np.zeros([1 , len(x_input)])[0], np.zeros([1 , len(x_input)])[0],x_input,np.zeros([1 , len(y_input)])[0], np.zeros([1 , len(y_input)])[0], y_input]).T
    X[2,:] = u[:,2]*x_acc_intensity
    Y[2,:] = u[:,5]*y_acc_intensity




# define kalman for each sensor
for experiment in range(experiments):
    print(f'runing mc num:{experiment}')
    Q = noise_modeling.velocity_model_noise_no_axis_correlation_2_d(var=process_noise_intensity_arr[experiment], dt=dt)\
        if mode == 'velocity' else noise_modeling.acceleration_model_noise_2_d(var=process_noise_intensity_arr[experiment], dt=dt)
    for run in range(number_of_mc_runs):
        for agent in agents:
            agent.measure(np.array([X[0],Y[0]]) , noise_factor=noise_factor_agent_coeff, noise_limit = measurement_noise_limit_agent)
            initial_state = array([[0 + randn() * sigma , 0.,0, 0 + randn() * sigma , 0.,0]], dtype=dtype).T\
                if mode == 'acceleration' else array([[0 + randn() * sigma , 0., 0 + randn() * sigma , 0.]], dtype=dtype).T  # Initial state [x, y]
            # initialize KF
        for agent in agents:
            agent.filter = ExtendedKalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )

        for measurement_index in range(len(agent.measurements)-1):
            for agent in agents:
                angle_flag = 1
                distance_flag = 0
                if angle_flag == 0 or distance_flag == 0:
                    measurement_dim = 1

                if use_prediction == 1:
                    #prediction
                    agent.filter.prediction()
                if agent.filter.counter % iterations_between_updates == 0:
                    if use_update == 1:
                        #update model
                        if angle_flag == 1 and distance_flag == 1:
                            agent.filter.range_measurement_model_2d(x_index=x_index, y_index=y_index , agent = agent)
                        elif angle_flag == 0:
                            agent.filter.range_measurement_model_2d_no_angle(x_index=x_index, y_index=y_index, agent=agent)
                        elif distance_flag == 0:
                            agent.filter.range_measurement_model_2d_no_distance(x_index=x_index, y_index=y_index, agent=agent)
                    if use_update == 1:
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
                        if assim_type == 'article': agent.filter.assimilate(agents)
                        if assim_type == 'mean': agent.filter.assimilate_mean(agents)
                        if assim_type == 'min_P':agent.filter.assimilate_min_P(agents)


            # rearrange data
        for agent in agents:
            if use_prediction == 1:
                agent.filter.predicted_state = agent.filter.predicted_state.reshape(len(X[0]) ,state_dim )
                agent.filter.predicted_covs = agent.filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
            if use_update == 1:
                agent.filter.R_arr = agent.filter.R_arr.reshape(int(len(agent.filter.R_arr)/measurement_dim**2),measurement_dim  ,measurement_dim )
                agent.filter.updated_covs = agent.filter.updated_covs.reshape(int(len(agent.filter.updated_covs)/state_dim**2),state_dim ,state_dim )
                agent.filter.updated_state = agent.filter.updated_state.reshape(int(len(agent.filter.updated_state)/state_dim) ,state_dim )
            # Ground thruth : X_tilde = X_real - X_estimated
            # agent.filter.estimation_error = [X[0] - agent.filter.updated_state[:, 0] , X[1] - agent.filter.updated_state[:, 1]]
            if use_assimilation == 1:
                agent.filter.assim_covs = agent.filter.assim_covs.reshape(int(len(agent.filter.assim_covs)/state_dim**2), state_dim, state_dim)
                agent.filter.assim_state = agent.filter.assim_state.reshape(int(len(agent.filter.assim_state)/state_dim), state_dim)
        agents_mc.append(deepcopy(agents))
    experiments_arr.append(deepcopy(agents_mc))
    # agents_mc=[]


t_end = time()
run_time = t_end-t_start
print(f'process took {run_time} seconds')
save_figs_mode = 0
cutoff = -1 if use_assimilation == 1 else len(T)
figs_path =r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\graphs'
plots.agents_mean_vs_agents_assim(mc_number=0 , number_of_agents=num_of_agents ,simulation_time = T[:cutoff] ,simulation_measurement = X[:,:cutoff], start_index = 1  ,number_of_mc_runs= number_of_mc_runs , experiments = experiments_arr , mode = mode, is_assim = use_assimilation , path = figs_path, state_index = 0 , angle_flag = angle_flag , distance_flag = distance_flag , assim_type = assim_type , line_distance = agent_distance_from_line)

RMSE = plots.plot_mc_estimation_error_all(mc_number=0 ,agent_idx = 0,simulation_time = T[:cutoff] ,simulation_measurement = X[:,:cutoff], start_index = 10  ,number_of_mc_runs= number_of_mc_runs,Q_arr = process_noise_intensity_arr , experiments = experiments_arr , mode = mode, is_assim = use_assimilation)
plots.print_xy_assim(mc_number=0  , agent_range=[0,num_of_agents] , agents_mc=experiments_arr[0] , simulation_measurement_x=X[0,10:], simulation_measurement_y=Y[0,10:] , start_idx = 10 ,mode = mode)
plots.print_updated_state(mc_number=0, agent_range=[0,10] , agents_mc=experiments_arr[0] , simulation_measurement=X , simulation_time=T , mode = mode ,save_figs=save_figs_mode , fig_save_path=figs_path)
plots.print_updated_covariance(mc_number=0, agent_range=[0,10] , agents_mc=experiments_arr[0] , simulation_measurement=np.concatenate((X[:-1,:],Y[:-1,:]),axis=0) , simulation_time=T , mode = mode)
alg_utils.select_min_P_position_sensor(agents_mc[0])
plots.print_sensors_error(T , X)
plots.plot_mc_estimation_error(mc_number=0,state_index = 0 ,agent_idx = 0,simulation_time = T ,simulation_measurement = X[0] , start_index = 50  ,number_of_mc_runs= number_of_mc_runs,Q_arr = process_noise_intensity_arr , experiments = experiments_arr)

plots.print_measurement_comparison(agents_mc=experiments_arr[0],mc_number=0,agent_range=[0,10],simulation_time=T)

plots.print_assimilated_state(mc_number=0 , state_index= 2 , agent_range=[0,1] , agents_mc=experiments_arr[0] , simulation_measurement=X[:,:-1] , simulation_time=T[:-1] , mode = mode , save_figs=save_figs_mode ,fig_save_path = figs_path)
plots.print_assimilated_covariance(mc_number=0 , agent_range=[0,1] , agents_mc=experiments_arr[0] , simulation_measurement=X[:,:-1] , simulation_time=T[:-1] , mode = mode)

plots.plot_R_Q(mc_number=0 , state_index= 0 , agent_range=[0,1] , agents_mc=experiments_arr[0] , simulation_time=T[:-1])

plots.print_xy(mc_number=0  , agent_range=[0,10] , agents_mc=experiments_arr[0] , simulation_measurement_x=X[0,10:], simulation_measurement_y=Y[0,10:] , start_idx = 10 ,mode = mode)
plots.print_3d(mc_number=0  , agent_range=[0,4] , agents_mc=experiments_arr[0] , simulation_measurement_x=X[0,10:], simulation_measurement_y=Y[0,10:] , simulation_time=T[10:],start_index = 10 )
print('done')
plots.two_STD_graphs(T)
mean_state = 0
theoretical_P_mean = []

for run in range(number_of_mc_runs):
    mean_state += (squeeze(agents_mc[run][0].filter.predicted_state)[:, 0]) / number_of_mc_runs
    theoretical_P_mean.append(agents_mc[run][0].filter.predicted_covs[:, 0, 0] ** .5 )
