from scipy.integrate import solve_ivp
import numpy as np
from numpy import expand_dims ,squeeze , array , diag , eye , linspace
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , KalmanFilterInfo , UnscentedKalmanFilter,ExtendedKalmanFilter , Gaussian , add_gaussian_noise
from control import input, constant_input
from agent import Agent
from ode import acceleration_model , velocity_model
from position_models import circle , single_circle
from noise_modeling import velocity_model_noise , acceleration_model_noise , velocity_model_noise_3x3
import plots
# General parameters
#set filter params:
state_dim = 3
measurement_dim = 2
dt = .1
meas_var = np.array([1., 1.]) # M_squared
process_noise_factor = .1

#define simulation parameters
t_initial = 0
t_final = 30
simulation_process_noise =.1
simulation_input_amplitude =[1]
simulation_input_type: str = 'step'

# define agents parameters
measurement_noise_limit_agent = 5.
noise_factor_agent_coeff = 0.0
num_of_agents = 1
agent_positions = np.array([[0 , 0] ,[35, 0] , [0 , 30] , [30 , 30]])
agents = [Agent([agent_positions[i][0], agent_positions[i][1]],[randn()*0, randn()*0, randn()*0] , 1 , 0, id=i + 1) for i in range(num_of_agents)]


'''
reaserch suggestion:
Divide the kalman for each uncorrelated state to improve preformance
'''

x_acc_factor = .5
y_acc_factor = .2

initial_state = array([[1, 0 ,1 , 0]] , dtype='float64').T  # Initial state [x, y]
P = eye(state_dim , dtype='float64') * 10.0  # Initial estimation error covariance - P
Q = velocity_model_noise_3x3(var=process_noise_factor ,dt=dt)
R = diag(meas_var)  #R Measurement noise covariance
F = array([[1, dt , 0 , 0],[0, 1 , 0 , 0] , [0 , 0 , 1 , dt],[0,0,0,1]] , dtype='float64') # F the process transformation matrix
H = array([[1 , 0 , 0 , 0],[0 , 0, 1 ,0 ]],dtype='float64') # H the measurements transformation matrix
B = array([[0., x_acc_factor, 0.]] , dtype='float64').T * dt # Input matrix
G = array([[0, .0, 0]] , dtype='float64').T * dt #Dynamic Model Noise


initial_state_X = [1 ,0 ,0]
initial_state_Y = [1 ,0 ,0]
input_mat_X = array([[0., x_acc_factor,0.0]] , dtype='float64')
noise_mat_X = array([[0., 0, 0.]] , dtype='float64')
input_mat_Y = array([[0., y_acc_factor,0.0]] , dtype='float64')
noise_mat_Y = array([[0., 0, 0.]] , dtype='float64')
#Check how to start input at certain time.
T , X = acceleration_model(t_start=t_initial , t_stop=t_final ,initial_cond=initial_state_X,  input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude, dt=dt , B=input_mat_X, G=noise_mat_X)
T , Y = acceleration_model(t_start=t_initial , t_stop=t_final,initial_cond=initial_state_Y, input_type=simulation_input_type, model_noise_var=simulation_process_noise, input_amplitude=simulation_input_amplitude , dt=dt , B=input_mat_Y, G=noise_mat_Y)
u = np.array([np.squeeze(constant_input('step' , T , 1))]).T

real_measurements = np.array([X[0] , X[1]])
noised_measurements = real_measurements * randn(*real_measurements.shape)


# define kalman for each sensor
for agent in agents:
    agent.filter = KalmanFilter(x0 = initial_state,P =  P,Q =  Q,R =  R ,F = F , B = B , H = H ,u = u,dt = dt ,G=G )


#initialize arrays for storing state
for measurement in noised_measurements.T[0:-1]:
    for agent in agents:
        # measurement = agent.measure_iteratively(real_measurement) # sensor measuring using real data

        #prediction
        agent.filter.prediction()

        #update
        distance_agent = agent.calc_distance(real_measurement)
        noise_factor_agent = distance_agent*noise_factor_agent_coeff if distance_agent*noise_factor_agent_coeff < measurement_noise_limit_agent else measurement_noise_limit_agent
        agent.filter.update(expand_dims(measurement, axis=1), agent.filter.R + np.eye(2)*noise_factor_agent)  # feeding the update with measurement cov*distance factor
        agent.update_agent_measurement_noise(noise_factor_agent)
    # for agent in agents:
    #     agent.filter.assimilate(agents)
    #     agent.filter.assim_covs = agent.filter.assim_covs.reshape(len(X[0]), state_dim, state_dim)
    #     agent.filter.assim_state = agent.filter.assim_state.reshape(len(X[0]), state_dim)

    # rearrange data
for agent in agents:

    agent.filter.updated_covs = agent.filter.updated_covs.reshape(len(X[0]),state_dim ,state_dim )
    agent.filter.updated_state= agent.filter.updated_state.reshape(len(X[0]) ,state_dim )
    agent.filter.predicted_state = agent.filter.predicted_state.reshape(len(X[0]) ,state_dim )
    agent.filter.predicted_covs = agent.filter.predicted_covs.reshape(len(X[0]),state_dim ,state_dim )
    agent.filter.R_arr = agent.filter.R_arr.reshape(len(X[0]) ,measurement_dim  ,measurement_dim )
    # Ground thruth : X_tilde = X_real - X_estimated
    agent.filter.estimation_error = [X[0] - agent.filter.updated_state[:, 0] , X[1] - agent.filter.updated_state[:, 1]]
    # NEES (Normalized Estimated Error Squared ) : err = X_tilde.T @ P^-1 @ X_tilde
'''
The math is outside the scope of this book, but a random variable in the form  𝐱̃ 𝖳𝐏−1𝐱̃ 
is said to be chi-squared distributed with n degrees of freedom,
and thus the expected value of the sequence should be  𝑛.
Bar-Shalom [1] has an excellent discussion of this topic.
'''

print('done')



































for agent in agents:
    #error Assim X
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,0] - X[0],'r' )
    plt.plot(T, agent.filter.assim_covs[:, 0, 0] ** .5,'--k')
    plt.plot(T, -agent.filter.assim_covs[:, 0, 0] ** .5,'--k')
    # plt.fill_between(T ,agent.filter.assim_covs[:,0,0]**.5 , -agent.filter.assim_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['X' ,  '1 Sigma envelope'])
    plt.title(f'agent {agent.id} X Position (Assimilation) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    for agent in agents:
        # plot X
        ax = plt.figure().add_subplot()
        ax.plot(T, squeeze(agent.filter.assim_state)[:, 0], 'r')
        ax.plot(T, X[0], '--r')
        # ax.scatter(T[0] , agent.position[0] ,agent.position[0],'green')
        # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
        plt.fill_between(T, X[0] + agent.filter.assim_covs[:, 0, 0] ** .5, X[0] - agent.filter.assim_covs[:, 0, 0] ** .5,
                         facecolor='white', alpha=.2, edgecolor='black')
        # plt.fill_between(T ,  X[1] + agent.filter.assim_covs[:,1,1]**.5 , X[1] - agent.filter.assim_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.legend(['X estimation', 'X real'])
        plt.title(f'agent {agent.id} X state estimation (Assimilation) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

        # plot X'
        ax = plt.figure().add_subplot()
        ax.plot(T, squeeze(agent.filter.assim_state)[:, 1], 'b')
        ax.plot(T, X[1], '--b')
        # ax.scatter(T[0] , agent.position[0] ,agent.position[0],'green')
        # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
        # plt.fill_between(T , X[0] + agent.filter.assim_covs[:,0,0]**.5 , X[0] - agent.filter.assim_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.fill_between(T, X[1] + agent.filter.assim_covs[:, 1, 1] ** .5, X[1] - agent.filter.assim_covs[:, 1, 1] ** .5,
                         facecolor='yellow', alpha=.2, edgecolor='black')
        plt.legend(['X\' estimation', 'X\' real'])
        plt.title(f'agent {agent.id} X state estimation (Assimilation) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

        # plot Y
        plt.figure()
        plt.plot(T, squeeze(agent.filter.assim_state)[:, 2], 'r')
        plt.plot(T, Y[0], '--r')
        # plt.plot(agent.position[0] , agent.position[1] , '*g')
        # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
        plt.fill_between(T, Y[0] + agent.filter.assim_covs[:, 2, 2] ** .5, Y[0] - agent.filter.assim_covs[:, 2, 2] ** .5,
                         facecolor='yellow', alpha=.2, edgecolor='black')
        # plt.fill_between(T ,  Y[1] + agent.filter.assim_covs[:, 3, 3]**.5 , Y[1] - agent.filter.assim_covs[:, 3, 3]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        # plt.fill_between(T,X[2] + squeeze(agent.filter.assim_covs)[:,2]**.5 , X[2] - squeeze(agent.filter.assim_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.legend(['Y estimation', 'Y\' estimation', 'Y real', 'Y\' real', 'sensor position'])
        plt.title(f'agent {agent.id} Y state estimation(Assimilation) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

        # plot Y'
        plt.figure()
        plt.plot(T, squeeze(agent.filter.assim_state)[:, 3], 'b')
        plt.plot(T, Y[1], '--b')
        # plt.plot(agent.position[0] , agent.position[1] , '*g')
        # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
        # plt.fill_between(T , Y[0] + agent.filter.assim_covs[:, 2, 2]**.5 , Y[0] - agent.filter.assim_covs[:, 2, 2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.fill_between(T, Y[1] + agent.filter.assim_covs[:, 3, 3] ** .5, Y[1] - agent.filter.assim_covs[:, 3, 3] ** .5,
                         facecolor='yellow', alpha=.2, edgecolor='black')
        # plt.fill_between(T,X[2] + squeeze(agent.filter.assim_covs)[:,2]**.5 , X[2] - squeeze(agent.filter.assim_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.legend(['Y estimation', 'Y\' estimation', 'Y real', 'Y\' real', 'sensor position'])
        plt.title(f'agent {agent.id} Y state estimation(Assimilation) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

    for agent in agents:
        # error Assim X
        ax = plt.figure().add_subplot()
        ax.plot(T, squeeze(agent.filter.assim_state)[:, 0] - X[0], 'r')
        plt.plot(T, agent.filter.assim_covs[:, 0, 0] ** .5, '--k')
        plt.plot(T, -agent.filter.assim_covs[:, 0, 0] ** .5, '--k')
        # plt.fill_between(T ,agent.filter.assim_covs[:,0,0]**.5 , -agent.filter.assim_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.legend(['X', '1 Sigma envelope'])
        plt.title(f'agent {agent.id} X Position (Assimilation) Errors')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

    # Err X'
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,1] - X[1],'r' )
    plt.plot(T, agent.filter.assim_covs[:, 1, 1] ** .5,'--k')
    plt.plot(T, -agent.filter.assim_covs[:, 1, 1] ** .5,'--k')
    # plt.fill_between(T ,agent.filter.assim_covs[:,1,1]**.5 , -agent.filter.assim_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['X' ,  '1 Sigma envelope'])
    plt.title(f'agent {agent.id} X Velocity (Assimilation) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


    # Err Y
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,2] - Y[0],'r' )
    plt.plot(T, agent.filter.assim_covs[:, 2, 2] ** .5,'--k')
    plt.plot(T, -agent.filter.assim_covs[:, 2, 2] ** .5,'--k')
    # plt.fill_between(T ,agent.filter.assim_covs[:,2,2]**.5 , -agent.filter.assim_covs[:,2,2]**.5 ,facecolor = 'white' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y' ,  '1 Sigma envelope'])
    plt.title(f'agent {agent.id} Y position (Assimilation) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


    # Err Y'
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,3] - Y[1],'r' )
    plt.plot(T, agent.filter.assim_covs[:, 3, 3] ** .5,'--k')
    plt.plot(T, -agent.filter.assim_covs[:, 3, 3] ** .5,'--k')
    # plt.fill_between(T ,agent.filter.assim_covs[:,3,3]**.5 , -agent.filter.assim_covs[:,3,3]**.5 ,facecolor = 'white' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y\'' ,  '1 Sigma envelope'])
    plt.title(f'agent {agent.id} Y velocity (Assimilation) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()



    # Plot XY in 3d
    ax = plt.figure().add_subplot(projection = '3d')
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,0] ,squeeze(agent.filter.assim_state)[:,2] , 'b' )
    ax.plot(T ,X[0] , Y[0] ,'--r')
    for agent in agents:
        ax.scatter(T[0] , agent.position[0] ,agent.position[1],'green')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    # plt.fill_between(T , X[0] + agent.filter.assim_covs[:,0,0]**.5 , X[0] - agent.filter.assim_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T ,  X[1] + agent.filter.assim_covs[:,1,1]**.5 , X[1] - agent.filter.assim_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['estimation', 'real' ,'sensor position'])
    plt.title(f'agent {agent.id} position estimation (update) and measurements')
    ax.set_xlabel('Time')
    ax.set_ylabel('$X$')
    ax.set_zlabel(r'$Y$')
    plt.show()

    # Plot XY in 2d
    ax = plt.figure().add_subplot()
    ax.plot(squeeze(agent.filter.assim_state)[:,0] ,squeeze(agent.filter.assim_state)[:,2] , 'b' )
    ax.plot(X[0] , Y[0] ,'--r')
    for agent in agents:
        ax.scatter(agent.position[0] ,agent.position[1])
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    # plt.fill_between(T , X[0] + agent.filter.assim_covs[:,0,0]**.5 , X[0] - agent.filter.assim_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T ,  X[1] + agent.filter.assim_covs[:,1,1]**.5 , X[1] - agent.filter.assim_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['estimation', 'real' ,'sensor position'])
    plt.title(f'agent {agent.id} position estimation (update) and measurements')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # plt.xlim(-100 , 100)
    # plt.ylim(-100 , 100)
    # ax.set_aspect('equal')
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
plt.plot(T , agent.filter.updated_state.iloc[:,1] , 'g')
plt.plot(T ,agent.filter.updated_state.iloc[:,0]  , 'k')
plt.plot(T ,agent.filter.updated_state.iloc[:,1] , 'm')
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