import matplotlib.pyplot as plt
from main_EKF import agents,T,X,Y , agents_mc
from numpy import squeeze
run_number = 0
for agent_idx in range(len(agents)):
    # error X
    # plot X
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agents_mc[run_number][agent_idx].filter.updated_state)[:, 0] - X[0], 'r')
    plt.plot(T, agents_mc[run_number][agent_idx].filter.updated_covs[:, 0, 0] ** .5, '--k')
    plt.plot(T, -agents_mc[run_number][agent_idx].filter.updated_covs[:, 0, 0] ** .5, '--k')
    plt.legend(['X', '1 Sigma envelope'])
    plt.title(f'agent {agents_mc[run_number][agent_idx].id} at ({agents_mc[run_number][agent_idx].position[0,0]},{agents_mc[run_number][agent_idx].position[0,1]}) X state estimation (updated) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    # Err X'
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filter.updated_state)[:, 1] - X[1], 'r')
    plt.plot(T, agent.filter.updated_covs[:, 1, 1] ** .5, '--k')
    plt.plot(T, -agent.filter.updated_covs[:, 1, 1] ** .5, '--k')
    plt.legend(['X', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} X Velocity (update) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()
for agent in agents:

    # Err Y
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filter.updated_state)[:, 2] - Y[0], 'r')
    plt.plot(T, agent.filter.updated_covs[:, 2, 2] ** .5, '--k')
    plt.plot(T, -agent.filter.updated_covs[:, 2, 2] ** .5, '--k')
    plt.legend(['Y', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} Y position (update) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


for agent in agents:

    # Err Y'
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filter.updated_state)[:, 3] - Y[1], 'r')
    plt.plot(T, agent.filter.updated_covs[:, 3, 3] ** .5, '--k')
    plt.plot(T, -agent.filter.updated_covs[:, 3, 3] ** .5, '--k')
    plt.legend(['Y\'', '1 Sigma envelope'])
    plt.title(f'agent {agent.id} Y velocity (update) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


# R measurement noise matrix
for agent in agents:
    ax = plt.figure().add_subplot()
    ax.plot(T, agent.filter.R_arr[:, 0, 0] ,'r')
    plt.title(f'agent {agent.id} at ({agent.position[0,0]},{agent.position[0,1]}) Measurement noise')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()



for agent in agents:
    #error Assim X
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,0] - X[0],'r' )
    plt.plot(T, agent.filter.assim_covs[:, 0, 0] ** .5,'--k')
    plt.plot(T, -agent.filter.assim_covs[:, 0, 0] ** .5,'--k')
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
    plt.fill_between(T, X[0] + agent.filter.assim_covs[:, 0, 0] ** .5, X[0] - agent.filter.assim_covs[:, 0, 0] ** .5,
                     facecolor='white', alpha=.2, edgecolor='black')
    plt.legend(['X estimation', 'X real'])
    plt.title(f'agent {agent.id} X state estimation (Assimilation) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    # plot X'
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agent.filter.assim_state)[:, 1], 'b')
    ax.plot(T, X[1], '--b')
    # ax.scatter(T[0] , agent.position[0,0] ,agent.position[0,0],'green')
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
    plt.fill_between(T, Y[0] + agent.filter.assim_covs[:, 2, 2] ** .5, Y[0] - agent.filter.assim_covs[:, 2, 2] ** .5,
                     facecolor='yellow', alpha=.2, edgecolor='black')
    plt.legend(['Y estimation', 'Y\' estimation', 'Y real', 'Y\' real', 'sensor position'])
    plt.title(f'agent {agent.id} Y state estimation(Assimilation) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    # plot Y'
    plt.figure()
    plt.plot(T, squeeze(agent.filter.assim_state)[:, 3], 'b')
    plt.plot(T, Y[1], '--b')
    # plt.plot(agent.position[0,0] , agent.position[0,1] , '*g')
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

for agent in agents:
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

for agent in agents:
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

for agent in agents:
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

