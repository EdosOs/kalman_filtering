import matplotlib.pyplot as plt
from main_EKF import agents,T,X,Y , agents_mc
from numpy import squeeze
run_number = 0

for agent_idx in range(len(agents)):
    #plot X
    ax = plt.figure().add_subplot()
    ax.plot(T ,squeeze(agents_mc[run_number][agent_idx].filter.predicted_state)[:,0],'r' )
    ax.plot(T ,X[0] ,'--r')
    plt.legend(['X estimation' ,  'X real'])
    plt.title(f'agent {agents_mc[run_number][agent_idx].id} at position ({agents_mc[run_number][agent_idx].position[0,0]},{agents_mc[run_number][agent_idx].position[0,1]}) X state estimation (updated) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

for agent_idx in range(len(agents)):
    # plot X'
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agents_mc[run_number][agent_idx].filter.predicted_state)[:, 1], 'b')
    ax.plot(T, X[1], '--b')
    plt.legend(['X\' estimation', 'X\' real'])
    plt.title(
        f'agent {agents_mc[run_number][agent_idx].id} at position ({agents_mc[run_number][agent_idx].position[0, 0]},{agents_mc[run_number][agent_idx].position[0, 1]}) X\' state estimation (updated) and measurements')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()
