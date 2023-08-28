from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kalman import KalmanFilter , Gaussian , add_gaussian_noise
from control import input

#set filter params:
dt = .5
meas_var = [1, 0, 0] # M_squared
procc_var = 50

initial_cov_var = 100.0
initial_state = np.array([[0, 0 ,1]] , dtype='float64').T  # Initial state [x, y]
initial_covariance = np.eye(3 , dtype='float64') * initial_cov_var  # Initial estimation error covariance - P
process_variance = np.eye(3 , dtype='float64') * procc_var  # Q Process noise covariance
measurement_variance = np.diag([meas_var])  #R Measurement noise covariance
process_transformation = np.array([[1, dt , dt**2/2],[0, 1 , dt] , [0 , 0 , 1]] , dtype='float64') # F the process transformation matrix
measurements_transformation = np.eye(3) # H the measurements transformation matrix
B = np.array([[0, 0, 1]] , dtype='float64').T # Input matrix
G = np.array([[1, 1, 1]] , dtype='float64').T#Dynamic Model Noise

#Dynamic model
#enter equations from order 1 as parameters
model_noise_var = 1
t_span = [0, 10]
ode_fcn = lambda T,Y: [Y[1]+Y[2]*T,Y[2] , 0] + np.squeeze(B , axis=1) * input('step' , T , 1) + np.squeeze(G , axis=1) * np.random.randn() * model_noise_var
sol = solve_ivp (ode_fcn, t_span=t_span  ,y0=[0, 0, 1] ,t_eval=np.linspace(0 , 10 , 100))
T = sol.t
Y = sol.y #X , X_DOT
u = input('step' , T , 1)

#plot
# plt.figure()
# plt.plot(T , Y[0])
# plt.plot(T , Y[1])
# plt.plot(T , Y[2])
# plt.legend(['x' , 'x_dot' , 'x_ddot'])
# plt.show()

noisy_measurements = add_gaussian_noise(Y , 0 , meas_var)
noisy_measurements_org = noisy_measurements.copy()

sys_filter = KalmanFilter(x0 = initial_state,P =  initial_covariance,Q =  process_variance,R =  measurement_variance ,F = process_transformation , B = B , H = measurements_transformation ,u = u  )

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
    sys_filter.update(np.expand_dims(measurement , axis= 1))
    #save update data
    updated_state.append(sys_filter.state.copy())
    updated_covs.append(np.diag(sys_filter.P).copy())

sys_filter.predicted_state = pd.DataFrame(np.squeeze(predicted_state , axis=2))
sys_filter.updated_state = pd.DataFrame(np.squeeze(updated_state , axis=2))
sys_filter.predicted_P = pd.DataFrame(predicted_covs )
sys_filter.updated_P = pd.DataFrame(updated_covs)
delta_state_1 = sys_filter.updated_state.iloc[:,0] - Y[0]
delta_state_2 = sys_filter.updated_state.iloc[:,1] - Y[1]


#PLOTS
plt.figure()
plt.plot(T , sys_filter.updated_state.iloc[:,0],'r' ,T , sys_filter.updated_state.iloc[:,1], 'b',T , sys_filter.updated_state.iloc[:,2], 'g')
plt.plot(T ,Y[0] ,'--r', T  , Y[1] ,'--b', T  , Y[1] ,'--g')
plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
plt.fill_between(T , Y[0] + sys_filter.updated_P[0]**.5 , Y[0] - sys_filter.updated_P[0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
plt.fill_between(T ,  Y[1] + sys_filter.updated_P[1]**.5 , Y[1] - sys_filter.updated_P[1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
plt.fill_between(T,Y[2] + sys_filter.updated_P[2]**.5 , Y[2] - sys_filter.updated_P[2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
plt.legend(['state 1 est' , 'state 2 est', 'state 3 est' , 'state 1 clean' , 'state 2 clean', 'state 3 clean' , 'state 1 noisy' , 'state 2 noisy', 'state 3 noisy'])
plt.title('state estimation and measurements')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

plt.figure()
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
