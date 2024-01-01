import matplotlib.pyplot as plt
from kalman_attempt_1 import agents,T,X,Y
from numpy import squeeze

for agent in agents:
    #plot X
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agent.filter.assim_state)[:,0],'r' )
    ax.plot(T ,X[0] ,'--r')
    plt.legend(['X estimation' ,  'X real'])
    plt.title(f'agent {agent.id} at position ({agent.position[0]},{agent.position[1]}) X state estimation (assim) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    #plot X'
    ax = plt.figure().add_subplot()
    ax.plot(T , squeeze(agent.filter.assim_state)[:,1], 'b')
    ax.plot(T , X[1] ,'--b')
    plt.legend([ 'X\' estimation', 'X\' real' ])
    plt.title(f'agent {agent.id} at position ({agent.position[0]},{agent.position[1]}) X\' state estimation (assim) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent in agents:
    #plot Y
    plt.figure()
    plt.plot(T ,squeeze(agent.filter.assim_state)[:,2],'r')
    plt.plot(T ,Y[0] ,'--r')
    plt.legend(['Y estimation'  ,'Y real'])
    plt.title(f'agent {agent.id} at position ({agent.position[0]},{agent.position[1]}) Y state estimation(assim) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()


    # plot Y'
    plt.figure()
    plt.plot(T ,squeeze(agent.filter.assim_state)[:,3],'b')
    plt.plot(T ,Y[1] ,'--b')
    plt.legend(['Y estimation' , 'Y\' estimation', 'Y real' , 'Y\' real'])
    plt.title(f'agent {agent.id} at position ({agent.position[0]},{agent.position[1]}) Y state estimation(assim) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

