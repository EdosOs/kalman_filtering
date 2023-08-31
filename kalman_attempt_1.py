from scipy.integrate import solve_ivp
import numpy as np
from numpy import expand_dims ,squeeze , array , diag , eye , linspace
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , Gaussian , add_gaussian_noise
from control import input
from agent import Agent
from ode import acceleration_model , velocity_model
# define agents
state_dim = 4
measurement_dim = 2
num_of_agents = 1
agents = [Agent([5 , 1],[randn()*100, randn()*.1, randn()*.01] , 1.0 , 0, id=i + 1) for i in range(num_of_agents)]

#set filter params:
dt = .01
meas_var = [3, 3] # M_squared
procc_var = 1.

initial_cov_var = 100.0
initial_state = array([[0, 0 ,0 , 0]] , dtype='float64').T  # Initial state [x, y]
initial_covariance = eye(state_dim , dtype='float64') * initial_cov_var  # Initial estimation error covariance - P
process_variance = diag([0 , 0 , 0 , 1]) * procc_var  # Q Process noise covariance
measurement_variance = diag(meas_var)*1.0  #R Measurement noise covariance
process_transformation = array([[1, dt , 0 , 0],[0, 1 , 0 , 0] , [0 , 0 , 1 , dt],[0,0,0,1]] , dtype='float64') # F the process transformation matrix
measurements_transformation = array([[1 , 0 , 0 , 0],[0 , 0, 1 ,0 ]],dtype='float64') # H the measurements transformation matrix
B = array([[0., 1., 0., 0.2]] , dtype='float64').T * dt # Input matrix
G = array([[0, 0, 0, 0]] , dtype='float64').T#Dynamic Model Noise

initial_state_X = [0 ,0 ,0]
initial_state_Y = [0 ,0 ,0]
input_mat = array([[0. , 1 ,0.0]] , dtype='float64')
noise_mat = array([[0. ,1.0 ,0.]] , dtype='float64')
#Check how to start input at certain time.
T , X = acceleration_model(t_start=0 , t_stop=50 ,initial_cond=initial_state_X, input_type='step',input_amplitude=1 , model_noise_var= .1 , dt=dt , B=input_mat, G=noise_mat)
T , Y = acceleration_model(t_start=0 , t_stop=50,initial_cond=initial_state_Y, input_type='step',input_amplitude=0.2 , model_noise_var= .1 , dt=dt , B=input_mat, G=noise_mat)

# plt.figure()
# plt.plot(T,X[2])
# plt.show()
u = input('step' , T , 1)
# uy = input('step' , T , 1)
# uz =
# #Dynamic model
# #enter equations from order 1 as parameters
# model_noise_var = 1
# t_span = [0, 10]
# ode_fcn = lambda T,X: [X[1]+X[2]*T,X[2] , 0] + squeeze(B , axis=1) * input('step' , T , 1) + squeeze(G , axis=1) * randn() * model_noise_var
# sol = solve_ivp (ode_fcn, t_span=t_span  ,y0=[0, 0, 1] ,t_eval=linspace(0 , 10 , 100))
# T = sol.t
# X = sol.y #STATE , STATE_DOT
# u = input('step' , T , 1)

# # plot
# plt.figure()
# plt.plot(T , X[0])
# plt.plot(T , X[1])
# plt.plot(T , X[2])
# plt.legend(['x' , 'x_dot' , 'x_ddot'])
# plt.show()

# noisy_measurements = add_gaussian_noise(array([X[0],Y[0]])  , 0 , meas_var)
# noisy_measurements_org = noisy_measurements.copy()
for agent in agents:
    agent.filt = KalmanFilter(x0 = initial_state,P =  initial_covariance,Q =  process_variance,R =  measurement_variance ,F = process_transformation , B = B , H = measurements_transformation ,u = u,dt = dt  )
#initialize arrays for storing state
for agent in agents:
    for real_measurement in array([X[0],Y[0]]).T:
        measurement = agent.measure(real_measurement) # sensor measuring using real data
        #prediction
        agent.filt.prediction()

        #update
        agent.filt.update(expand_dims(measurement, axis=1), agent.filt.R * agent.noise_factor(measurement))  # feeding the update with measurement cov*distance factor
    agent.filt.R_update_arr = agent.filt.R_update_arr.reshape(len(X[0]) ,measurement_dim  ,measurement_dim )
    agent.filt.R_pred_arr = agent.filt.R_pred_arr.reshape(len(X[0]),measurement_dim , measurement_dim )
    agent.filt.updated_covs = agent.filt.updated_covs.reshape(len(X[0]),state_dim ,state_dim )
    agent.filt.updated_state= agent.filt.updated_state.reshape(len(X[0]) ,state_dim )
    agent.filt.predicted_state = agent.filt.predicted_state.reshape(len(X[0]) ,state_dim )
    agent.filt.predicted_covs = agent.filt.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
    # for agent in agents:
    #     if agent.catch_flag == 0:
    #         # predict
    #         agent_measurement = agent.measure(measurement)
    #         # update
    #         if abs(agent_measurement)<1:
    #             agent.catch_flag = 1
    #             continue
    #         agent.move(agent_measurement / 2)
# res = noisy_measurements_org[0] - squeeze(agent.filt.updated_state)[:,0]
#PLOTS
for agent in agents:
    plt.figure()
    plt.plot(T ,squeeze(agent.filt.updated_state)[:,0],'r' ,T , squeeze(agent.filt.updated_state)[:,1], 'b')
    plt.plot(T ,X[0] ,'--r', T  , X[1] ,'--b')
    plt.plot(agent.position[0] , agent.position[1] , '*g')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    plt.fill_between(T , X[0] + agent.filt.updated_covs[:,0,0]**.5 , X[0] - agent.filt.updated_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.fill_between(T ,  X[1] + agent.filt.updated_covs[:,1,1]**.5 , X[1] - agent.filt.updated_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['X estimation' , 'X\' estimation', 'X real' , 'X\' real' ,'sensor position'])
    plt.title(f'agent {agent.id} X state estimation and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    plt.figure()
    plt.plot(T ,squeeze(agent.filt.updated_state)[:,2],'r' ,T , squeeze(agent.filt.updated_state)[:,3], 'b')
    plt.plot(T ,Y[0] ,'--r', T  , Y[1] ,'--b')
    plt.plot(agent.position[0] , agent.position[1] , '*g')

    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    plt.fill_between(T , X[0] + agent.filt.updated_covs[:,0,0]**.5 , X[0] - agent.filt.updated_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.fill_between(T ,  X[1] + agent.filt.updated_covs[:,1,1]**.5 , X[1] - agent.filt.updated_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T,X[2] + squeeze(agent.filt.updated_covs)[:,2]**.5 , X[2] - squeeze(agent.filt.updated_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y estimation' , 'Y\' estimation', 'Y real' , 'Y\' real' , 'sensor position'])
    plt.title(f'agent {agent.id} Y state estimation and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


#plot agent position
plt.figure()
plt.plot(T ,X[0] ,'--r')
for agent in agents:
    plt.plot(T, squeeze(agent.positions)[:,0] , ':')
plt.title('agents position')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend(['target']+['agent '+str(i+1) for i in range(len(agents))] )
plt.show()

plt.figure()
plt.plot(T , agent.filt.updated_state.iloc[:,1] , 'g')
plt.plot(T ,agent.filt.updated_state.iloc[:,0]  , 'k')
plt.plot(T ,agent.filt.updated_state.iloc[:,1] , 'm')
plt.plot(T ,X[0] ,'b')
plt.plot(T ,X[1] ,'c')
plt.show()
plt.legend(['state 1 est' , 'state 2 est' , 'state 1 real' , 'state 2 real'])
plt.title('state estimation and measurements')
plt.xlabel('time')
plt.ylabel('amplitude')
print('done')
