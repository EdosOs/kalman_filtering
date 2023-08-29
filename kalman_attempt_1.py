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
num_of_agents = 5 
agents = [Agent([randn()*100, randn()*.1, randn()*.01] , 0 , 0, id=i + 1) for i in range(5)]

#set filter params:
dt = .1
meas_var = [1., 0., 0.] # M_squared
procc_var = 1000

initial_cov_var = 100.0
initial_state = array([[0, 0 ,1]] , dtype='float64').T  # Initial state [x, y]
initial_covariance = eye(3 , dtype='float64') * initial_cov_var  # Initial estimation error covariance - P
process_variance = eye(3 , dtype='float64') * procc_var  # Q Process noise covariance
measurement_variance = diag(meas_var)*1.0  #R Measurement noise covariance
process_transformation = array([[1, dt , dt**2/2],[0, 1 , dt] , [0 , 0 , 1]] , dtype='float64') # F the process transformation matrix
measurements_transformation = eye(3) # H the measurements transformation matrix
B = array([[0, 0, 1]] , dtype='float64').T # Input matrix
G = array([[1, 1, 1]] , dtype='float64').T#Dynamic Model Noise

#Dynamic model
#enter equations from order 1 as parameters
model_noise_var = 1
t_start = 0
t_stop = 10
t_span = [t_start, t_stop]
ode_fcn = lambda T,X: [X[1]+X[2]*T,X[2] , 0] + squeeze(B , axis=1) * input('sin' , T*3 , 5) + squeeze(G , axis=1) * randn() * model_noise_var
sol = solve_ivp (ode_fcn, t_span=t_span  ,y0=[0, 0, 1] ,t_eval=linspace(t_start , t_stop , int((t_stop-t_start) / dt)) )
T , X = acceleration_model(t_start=0 , t_stop=10 , model_noise_var= 1 , dt=dt , B=B, G=G)
T = sol.t
X = sol.y #STATE , STATE_DOT
u = input('step' , T , 1)

# #Dynamic model
# #enter equations from order 1 as parameters
# model_noise_var = 1
# t_span = [0, 10]
# ode_fcn = lambda T,X: [X[1]+X[2]*T,X[2] , 0] + squeeze(B , axis=1) * input('step' , T , 1) + squeeze(G , axis=1) * randn() * model_noise_var
# sol = solve_ivp (ode_fcn, t_span=t_span  ,y0=[0, 0, 1] ,t_eval=linspace(0 , 10 , 100))
# T = sol.t
# X = sol.y #STATE , STATE_DOT
# u = input('step' , T , 1)

#plot
# plt.figure()
# plt.plot(T , X[0])
# plt.plot(T , X[1])
# plt.plot(T , X[2])
# plt.legend(['x' , 'x_dot' , 'x_ddot'])
# plt.show()

noisy_measurements = add_gaussian_noise(X , 0 , meas_var)
noisy_measurements_org = noisy_measurements.copy()

sys_filter = KalmanFilter(x0 = initial_state,P =  initial_covariance,Q =  process_variance,R =  measurement_variance ,F = process_transformation , B = B , H = measurements_transformation ,u = u  )
# set agents filters:
# agent_P =
# agent_Q =
# agent_R =
# agent_F =
# agent_B =
# agent_H =
# agent_u =
for agent in agents:
    agent.filter = KalmanFilter(x0 = agent.state , P = agent_P , Q = agent_Q, R = agent_R , F=agent_F , B=agent_B, H= agent.H, u = agent_u)
#initialize arrays for storing state
predicted_state = []
updated_state = []
updated_covs = []
predicted_covs = []
for measurement in noisy_measurements.T:
    #prediction
    sys_filter.prediction()
    #update
    sys_filter.update(expand_dims(measurement , axis= 1))
    for agent in agents:
        if agent.catch_flag == 0:
            # predict
            agent_measurement = agent.measure(measurement)
            # update
            if abs(agent_measurement)<1:
                agent.catch_flag = 1
                continue
            agent.move(agent_measurement / 2)

#PLOTS
plt.figure()
plt.plot(T ,squeeze(sys_filter.updated_state)[:,0],'r' ,T , squeeze(sys_filter.updated_state)[:,1], 'b',T ,squeeze(sys_filter.updated_state)[:,2], 'g')
plt.plot(T ,X[0] ,'--r', T  , X[1] ,'--b', T  , X[1] ,'--g')
plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
plt.fill_between(T , X[0] + squeeze(sys_filter.updated_covs)[:,0]**.5 , X[0] - squeeze(sys_filter.updated_covs)[:,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
plt.fill_between(T ,  X[1] + squeeze(sys_filter.updated_covs)[:,1]**.5 , X[1] - squeeze(sys_filter.updated_covs)[:,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
plt.fill_between(T,X[2] + squeeze(sys_filter.updated_covs)[:,2]**.5 , X[2] - squeeze(sys_filter.updated_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
plt.legend(['state 1 est' , 'state 2 est', 'state 3 est' , 'state 1 clean' , 'state 2 clean', 'state 3 clean' , 'state 1 noisy' , 'state 2 noisy', 'state 3 noisy'])
plt.title('state estimation and measurements')
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
plt.plot(T , sys_filter.updated_state.iloc[:,1] , 'g')
plt.plot(T ,sys_filter.updated_state.iloc[:,0]  , 'k')
plt.plot(T ,sys_filter.updated_state.iloc[:,1] , 'm')
plt.plot(T ,X[0] ,'b')
plt.plot(T ,X[1] ,'c')
plt.show()
plt.legend(['state 1 est' , 'state 2 est' , 'state 1 real' , 'state 2 real'])
plt.title('state estimation and measurements')
plt.xlabel('time')
plt.ylabel('amplitude')
print('done')
