from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , Gaussian , add_gaussian_noise
num_of_points = 100
t_span = [0, 100]
vdp1 = lambda T,Y: [Y[1], -Y[1]*T - Y[0]]
sol = solve_ivp (vdp1, t_span , [1, 1] ,t_eval=np.linspace(t_span[0] , t_span[1] , num_of_points))
T = sol.t
Y = sol.y
#set filter params:
meas_var = 0.5
procc_var = 100
initial_state = np.array([0, 1] , dtype='float64')  # Initial state [x, y]
initial_covariance = np.eye(2) * 1000.0  # Initial estimation error covariance - P
process_variance = np.eye(2) * procc_var  # Q Process noise covariance
measurement_variance = np.eye(2) * meas_var  #R Measurement noise covariance


#Dynamic model
#enter equations from order 1 as parameters
# vdp1 = lambda T,Y: [Y[1], (1 - Y[0]**2) * Y[1] - Y[0]]
t_span = [0, 10]
num_of_points = 100
vdp1 = lambda T,Y: [Y[1], -Y[1]*T - Y[0]]
sol = solve_ivp (vdp1, t_span , [1, 1] ,t_eval=np.linspace(t_span[0] , t_span[1] , num_of_points))
T = sol.t
Y = sol.y
#plot measurements
# plt.figure()
# plt.plot(T ,Y[0] ,'C1')
# plt.plot(T ,Y[1] ,'C2')
# plt.show()


#define measurements
T = np.array(np.linspace(0,100,100) , dtype='float64')
Y = np.array([np.linspace(0,1000,100) , np.linspace(0,100,100)] , dtype='float64')
dy = np.array([10 , 1])
noisy_measurements = add_gaussian_noise(Y , 0 , meas_var)


sys_filter = KalmanFilter(initial_state, initial_covariance, process_variance, measurement_variance , dy)

#initialize arrays for storing state
predicted_state = []
updated_state = []
updated_covs = []
predicted_covs = []
for measurement in noisy_measurements.T:
    #prediction
    sys_filter.prediction()
    #save prediction data
    predicted_state.append(sys_filter.state.copy())
    predicted_covs.append(np.diag(sys_filter.P).copy())

    #update
    sys_filter.update(measurement)
    #save update data
    updated_state.append(sys_filter.state.copy())
    updated_covs.append(np.diag(sys_filter.P).copy())

sys_filter.predicted_state = pd.DataFrame(predicted_state)
sys_filter.updated_state = pd.DataFrame(updated_state)
sys_filter.predicted_P = pd.DataFrame(predicted_covs)
sys_filter.updated_P = pd.DataFrame(updated_covs)

plt.figure()
plt.plot(T , sys_filter.updated_state.iloc[:,0] , 'r')
plt.plot(T , sys_filter.updated_state.iloc[:,1] , 'g')
# plt.plot(T , sys_filter.predicted_state.iloc[:,0] , 'k')
# plt.plot(T , sys_filter.predicted_state.iloc[:,1] , 'm')
plt.plot(T ,Y[0] ,'b')
plt.plot(T ,Y[1] ,'c')
plt.show()
plt.legend(['state 1 est' , 'state 2 est' , 'state 1 real' , 'state 2 real'])
plt.title('state estimation and measurements')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.figure()
plt.plot(T ,  , 'r')
plt.plot(T , sys_filter.updated_state.iloc[:,1] , 'g')
plt.plot(T ,sys_filter.updated_state.iloc[:,0]  , 'k')
plt.plot(T ,sys_filter.updated_state.iloc[:,1] , 'm')
plt.plot(T ,Y[0] ,'b')
plt.plot(T ,Y[1] ,'c')
plt.show()
plt.legend(['state 1 est' , 'state 2 est' , 'state 1 real' , 'state 2 real'])
plt.title('state estimation and measurements')
plt.xlabel('time')
plt.ylabel('amplitude')
print('done')
