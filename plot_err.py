import matplotlib.pyplot as plt
from kalman_attempt_1 import agents,T,X,Y
from numpy import squeeze

for agent in agents:
    # error X
    # plot X
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filt.updated_state)[:, 0] - X[0], 'r')
    plt.plot(T, agent.filt.updated_covs[:, 0, 0] ** .5, '--k')
    plt.plot(T, -agent.filt.updated_covs[:, 0, 0] ** .5, '--k')
    # plt.fill_between(T ,agent.filt.updated_covs[:,0,0]**.5 , -agent.filt.updated_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['X', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} at ({agent.position[0]},{agent.position[1]}) X state estimation (updated) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    # Err X'
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filt.updated_state)[:, 1] - X[1], 'r')
    plt.plot(T, agent.filt.updated_covs[:, 1, 1] ** .5, '--k')
    plt.plot(T, -agent.filt.updated_covs[:, 1, 1] ** .5, '--k')
    # plt.fill_between(T ,agent.filt.updated_covs[:,1,1]**.5 , -agent.filt.updated_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['X', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} X Velocity (update) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()
for agent in agents:

    # Err Y
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filt.updated_state)[:, 2] - Y[0], 'r')
    plt.plot(T, agent.filt.updated_covs[:, 2, 2] ** .5, '--k')
    plt.plot(T, -agent.filt.updated_covs[:, 2, 2] ** .5, '--k')
    # plt.fill_between(T ,agent.filt.updated_covs[:,2,2]**.5 , -agent.filt.updated_covs[:,2,2]**.5 ,facecolor = 'white' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} Y position (update) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


for agent in agents:

    # Err Y'
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filt.updated_state)[:, 3] - Y[1], 'r')
    plt.plot(T, agent.filt.updated_covs[:, 3, 3] ** .5, '--k')
    plt.plot(T, -agent.filt.updated_covs[:, 3, 3] ** .5, '--k')
    # plt.fill_between(T ,agent.filt.updated_covs[:,3,3]**.5 , -agent.filt.updated_covs[:,3,3]**.5 ,facecolor = 'white' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y\'', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} Y velocity (update) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    ax = plt.figure().add_subplot()
    ax.plot(T, agent.filt.R_arr[:, 0, 0] ,'r')
    plt.title(f'agent {agent.id} at ({agent.position[0]},{agent.position[1]}) Measurement noise')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


