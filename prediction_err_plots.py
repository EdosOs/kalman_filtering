import matplotlib.pyplot as plt
from main_EKF import agents,T,X,Y , agents_mc
from numpy import squeeze
run_number = 0
for agent_idx in range(len(agents)):
    # error X
    # plot X
    ax = plt.figure().add_subplot()
    ax.plot(T, squeeze(agents_mc[run_number][agent_idx].filter.predicted_state)[:, 0] - X[0], 'r')
    plt.plot(T, agents_mc[run_number][agent_idx].filter.predicted_covs[:, 0, 0] ** .5, '--k')
    plt.plot(T, -agents_mc[run_number][agent_idx].filter.predicted_covs[:, 0, 0] ** .5, '--k')
    plt.legend(['X', '1 Sigma envelope'])
    plt.title(f'agent {agents_mc[run_number][agent_idx].id} at ({agents_mc[run_number][agent_idx].position[0,0]},{agents_mc[run_number][agent_idx].position[0,1]}) X state estimation (predicted) Errors')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()