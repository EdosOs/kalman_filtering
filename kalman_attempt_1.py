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
num_of_agents = 5
agents = [Agent([randn()*100 , randn()*100],[randn()*100, randn()*.1, randn()*.01] , 10.0 , 0, id=i + 1) for i in range(num_of_agents)]

#set filter params:
dt = .01
meas_var = [3, 3] # M_squared
procc_var = 50.
'''
Divide the kalman for each uncorrelated state o improve preformance
'''
initial_cov_var = 100.0
initial_state = array([[0, 0 ,0 , 0]] , dtype='float64').T  # Initial state [x, y]
initial_covariance = eye(state_dim , dtype='float64') * initial_cov_var  # Initial estimation error covariance - P
process_variance = diag([0 , 1 , 0 , .2]) * procc_var  # Q Process noise covariance
measurement_variance = diag(meas_var)*1.0  #R Measurement noise covariance
process_transformation = array([[1, dt , 0 , 0],[0, 1 , 0 , 0] , [0 , 0 , 1 , dt],[0,0,0,1]] , dtype='float64') # F the process transformation matrix
measurements_transformation = array([[1 , 0 , 0 , 0],[0 , 0, 1 ,0 ]],dtype='float64') # H the measurements transformation matrix
B = array([[0., 1., 0., 0.2]] , dtype='float64').T * dt # Input matrix
G = array([[0, .1, 0, .02]] , dtype='float64').T * dt #Dynamic Model Noise

initial_state_X = [0 ,0 ,0]
initial_state_Y = [0 ,0 ,0]
input_mat_X = array([[0., 1, 0.0]] , dtype='float64')
noise_mat_X = array([[0., 0, 0.]] , dtype='float64')
input_mat_Y = array([[0., .2, 0.0]] , dtype='float64')
noise_mat_Y = array([[0., 0, 0.]] , dtype='float64')
#Check how to start input at certain time.
T , X = acceleration_model(t_start=0 , t_stop=50 ,initial_cond=initial_state_X ,  input_type='pulse',input_amplitude=1 , model_noise_var= .1 , dt=dt , B=input_mat_X, G=noise_mat_X)
T , Y = acceleration_model(t_start=0 , t_stop=50,initial_cond=initial_state_Y, input_type='step',input_amplitude=1 , model_noise_var= .1 , dt=dt , B=input_mat_Y, G=noise_mat_Y)

# plt.figure()
# plt.plot(T,X[1])
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

# plot
# plt.figure()
# plt.plot(T , X[0])
# # plt.plot(T , X[1])
# # plt.plot(T , X[2])
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
    agent.filt.R_arr = agent.filt.R_arr.reshape(len(X[0]) ,measurement_dim  ,measurement_dim )
    agent.filt.updated_covs = agent.filt.updated_covs.reshape(len(X[0]),state_dim ,state_dim )
    agent.filt.updated_state= agent.filt.updated_state.reshape(len(X[0]) ,state_dim )
    agent.filt.predicted_state = agent.filt.predicted_state.reshape(len(X[0]) ,state_dim )
    agent.filt.predicted_covs = agent.filt.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
    # Ground thruth : X_tilde = X_real - X_estimated
    agent.filt.estimation_error = [X[0] - agent.filt.updated_state[:, 0] , X[1] - agent.filt.updated_state[:, 1]]
    # NEES (Normalized Estimated Error Squared ) : err = X_tilde.T @ P^-1 @ X_tilde
    '''The math is outside the scope of this book, but a random variable in the form  𝐱̃ 𝖳𝐏−1𝐱̃ 
  is said to be chi-squared distributed with n degrees of freedom, and thus the expected value of the sequence should be  𝑛
 . Bar-Shalom [1] has an excellent discussion of this topic.'''


                            # agent.filt.residual = agent.measurements - a
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
    #plot X X'
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(T ,squeeze(agent.filt.updated_state)[:,0],'r' ,T , squeeze(agent.filt.updated_state)[:,1], 'b')
    ax.plot(T ,X[0] ,'--r', T  , X[1] ,'--b')
    ax.scatter(T[0] , agent.position[0] ,agent.position[0],'green')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    # plt.fill_between(T , X[0] + agent.filt.updated_covs[:,0,0]**.5 , X[0] - agent.filt.updated_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T ,  X[1] + agent.filt.updated_covs[:,1,1]**.5 , X[1] - agent.filt.updated_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['X estimation' , 'X\' estimation', 'X real' , 'X\' real' ,'sensor position'])
    plt.title(f'agent {agent.id} X state estimation and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    #plot Y Y'
    plt.figure()
    plt.plot(T ,squeeze(agent.filt.updated_state)[:,2],'r' ,T , squeeze(agent.filt.updated_state)[:,3], 'b')
    plt.plot(T ,Y[0] ,'--r', T  , Y[1] ,'--b')
    plt.plot(agent.position[0] , agent.position[1] , '*g')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    plt.fill_between(T , Y[0] + agent.filt.updated_covs[:, 2, 2]**.5 , Y[0] - agent.filt.updated_covs[:, 2, 2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.fill_between(T ,  Y[1] + agent.filt.updated_covs[:, 3, 3]**.5 , Y[1] - agent.filt.updated_covs[:, 3, 3]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T,X[2] + squeeze(agent.filt.updated_covs)[:,2]**.5 , X[2] - squeeze(agent.filt.updated_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y estimation' , 'Y\' estimation', 'Y real' , 'Y\' real' , 'sensor position'])
    plt.title(f'agent {agent.id} Y state estimation and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    #plot res
    plt.figure()
    plt.plot()

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
#The measurements are fixing the covariance of the positions but not the velocities
"""
When the Kalman filter is of higher order than your physical process it also has an infinite number of solutions to choose from. The answer is not just non-optimal, it often diverges and never recovers.

For best performance you need a filter whose order matches the system's order.

For best performance you need a filter whose order matches the system's order. In many cases that will be easy to do - if you are designing a Kalman filter to read the thermometer of a freezer it seems clear that a zero order filter is the right choice. But what order should we use if we are tracking a car? Order one will work well while the car is moving in a straight line at a constant speed, but cars turn, speed up, and slow down, in which case a second order filter will perform better. That is the problem addressed in the Adaptive Filters chapter. There we will learn how to design a filter that adapts to changing order in the tracked object's behavior.

With that said, a lower order filter can track a higher order process so long as you add enough process noise and you keep the discretization period small (100 samples a second are usually locally linear). The results will not be optimal, but they can still be very good, and I always reach for this tool first before trying an adaptive filter
"""