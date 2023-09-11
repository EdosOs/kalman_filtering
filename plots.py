import matplotlib.pyplot as plt
from kalman_attempt_1 import agents,T,X,Y
from numpy import squeeze


for agent in agents:
    #plot X
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filt.updated_state)[:,0],'r' )
    ax.plot(T ,X[0] ,'--r')
    plt.legend(['X estimation' ,  'X real'])
    plt.title(f'agent {agent.id} X state estimation (updated) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    #plot X'
    ax = plt.figure().add_subplot()
    ax.plot(T , squeeze(agent.filt.updated_state)[:,1], 'b')
    ax.plot(T , X[1] ,'--b')
    # ax.scatter(T[0] , agent.position[0] ,agent.position[0],'green')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    # plt.fill_between(T , X[0] + agent.filt.updated_covs[:,0,0]**.5 , X[0] - agent.filt.updated_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T ,  X[1] + agent.filt.updated_covs[:,1,1]**.5 , X[1] - agent.filt.updated_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend([ 'X\' estimation', 'X\' real' ])
    plt.title(f'agent {agent.id} X state estimation (updated) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    #plot Y
    plt.figure()
    plt.plot(T ,squeeze(agent.filt.updated_state)[:,2],'r')
    plt.plot(T ,Y[0] ,'--r')
    # plt.plot(agent.position[0] , agent.position[1] , '*g')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    # plt.fill_between(T , Y[0] + agent.filt.updated_covs[:, 2, 2]**.5 , Y[0] - agent.filt.updated_covs[:, 2, 2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T ,  Y[1] + agent.filt.updated_covs[:, 3, 3]**.5 , Y[1] - agent.filt.updated_covs[:, 3, 3]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T,X[2] + squeeze(agent.filt.updated_covs)[:,2]**.5 , X[2] - squeeze(agent.filt.updated_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y estimation' , 'Y\' estimation', 'Y real' , 'Y\' real' , 'sensor position'])
    plt.title(f'agent {agent.id} Y state estimation(updated) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


    # plot Y'
    plt.figure()
    plt.plot(T ,squeeze(agent.filt.updated_state)[:,3],'b')
    plt.plot(T ,Y[1] ,'--b')
    # plt.plot(agent.position[0] , agent.position[1] , '*g')
    # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
    # plt.fill_between(T , Y[0] + agent.filt.updated_covs[:, 2, 2]**.5 , Y[0] - agent.filt.updated_covs[:, 2, 2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T ,  Y[1] + agent.filt.updated_covs[:, 3, 3]**.5 , Y[1] - agent.filt.updated_covs[:, 3, 3]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    # plt.fill_between(T,X[2] + squeeze(agent.filt.updated_covs)[:,2]**.5 , X[2] - squeeze(agent.filt.updated_covs)[:,2]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
    plt.legend(['Y estimation' , 'Y\' estimation', 'Y real' , 'Y\' real'])
    plt.title(f'agent {agent.id} Y state estimation(updated) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()



for agent in agents:
    #plot XY
    ax = plt.figure().add_subplot()
    ax.plot(squeeze(agent.filt.updated_state)[:,0],squeeze(agent.filt.updated_state)[:,2],'r' )
    ax.plot(X[0] , Y[0] ,'--r')
    plt.legend(['X estimation' ,  'X real'])
    plt.title(f'agent {agent.id} at ({agent.position[0]},{agent.position[1]}) XY state estimation (updated) and measurements')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    for agent in agents:
        # plot X'Y'
        ax = plt.figure().add_subplot()
        ax.plot(squeeze(agent.filt.updated_state)[:, 1], squeeze(agent.filt.updated_state)[:, 3], 'r')
        ax.plot(X[1], Y[1], '--r')
        plt.legend(['X estimation', 'X real'])
        plt.title(f'agent {agent.id} at ({agent.position[0]},{ agent.position[1]}) XY state estimation (updated) and measurements')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        
        # 3D
        # Plot XY in 3d
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(T, squeeze(agent.filt.updated_state)[:, 0], squeeze(agent.filt.updated_state)[:, 2], 'b')
        ax.plot(T, X[0], Y[0], '--r')
        for agent in agents:
            ax.scatter(T[0], agent.position[0], agent.position[1], 'green')
        # plt.plot(T ,noisy_measurements_org[0] ,'2r',T ,noisy_measurements_org[1] ,'2b',T ,noisy_measurements_org[2] ,'2g',  linewidth = 0.5)
        # plt.fill_between(T , X[0] + agent.filt.assim_covs[:,0,0]**.5 , X[0] - agent.filt.assim_covs[:,0,0]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        # plt.fill_between(T ,  X[1] + agent.filt.assim_covs[:,1,1]**.5 , X[1] - agent.filt.assim_covs[:,1,1]**.5 ,facecolor = 'yellow' , alpha = .2 , edgecolor = 'black')
        plt.legend(['estimation', 'real', 'sensor position'])
        plt.title(f'agent {agent.id} position estimation (update) and measurements')
        ax.set_xlabel('Time')
        ax.set_ylabel('$X$')
        ax.set_zlabel(r'$Y$')
        plt.show()
