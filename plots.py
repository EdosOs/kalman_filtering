import matplotlib.pyplot as plt
import numpy as np
# from main_EKF import agents,T,X,Y , agents_mc
from numpy import squeeze


# plot noised measurements vs original measurements
def print_measurement_comparison(agents_mc, mc_number, agent_range, simulation_time):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time, agents_mc[mc_number][agent_idx].measurements, 'r')
        ax.plot(simulation_time, agents_mc[mc_number][agent_idx].measurements_clean, 'b')
        plt.legend(['noised simulation', 'clean simulation'])
        plt.title(f'agent {agents_mc[mc_number][agent_idx].id} measurements vs real range in time')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()


def print_residual(agents_mc, mc_number, agent_range, simulation_time):
    # plot residual (z - Hx)
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time, agents_mc[mc_number][agent_idx].filter.residual_arr, 'r')
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) Residual error')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()


def print_updated_state(mc_number, state_index, agent_range, agents_mc, simulation_time, simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot X
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, state_index], 'r')
        ax.plot(simulation_time, simulation_measurement, '--r')
        plt.legend(['X estimation', 'X real'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (updated) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()


def print_updated_covariance(mc_number, state_index, agent_range, agents_mc, simulation_time,
                            simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time,simulation_measurement-
                squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, state_index] , 'r')
        plt.plot(simulation_time,
                 agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
        plt.plot(simulation_time,
                 -agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
        plt.legend(['X', '1 Sigma envelope'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
            f'{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (updated) Errors')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

def print_predicted_state(mc_number, state_index, agent_range, agents_mc, simulation_time, simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot X
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.predicted_state)[:, state_index], 'r')
        ax.plot(simulation_time, simulation_measurement, '--r')
        plt.legend(['X estimation', 'X real'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (predicted) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()


def print_predicted_covariance(mc_number, state_index, agent_range, agents_mc, simulation_time
                               ,simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time,
                squeeze(agents_mc[mc_number][agent_idx].filter.predicted_state)[:, state_index] - simulation_measurement, 'r')
        plt.plot(simulation_time,
                 agents_mc[mc_number][agent_idx].filter.predicted_covs[:, state_index, state_index] ** .5, '--k')
        plt.plot(simulation_time,
                 -agents_mc[mc_number][agent_idx].filter.predicted_covs[:, state_index, state_index] ** .5, '--k')
        plt.legend(['X', '1 Sigma envelope'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
            f'{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (predicted) Errors')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

def print_assimilated_state(mc_number, state_index, agent_range, agents_mc, simulation_time, simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot X
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, state_index], 'r')
        ax.plot(simulation_time, simulation_measurement, '--r')
        plt.legend(['X estimation', 'X real'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (assim) and measurements')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()
def print_assimilated_covariance(mc_number, state_index, agent_range, agents_mc, simulation_time,
                            simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time,
                squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, state_index] - simulation_measurement[
                    0], 'r')
        plt.plot(simulation_time,
                 agents_mc[mc_number][agent_idx].filter.assim_covs[:, state_index, state_index] ** .5, '--k')
        plt.plot(simulation_time,
                 -agents_mc[mc_number][agent_idx].filter.assim_covs[:, state_index, state_index] ** .5, '--k')
        plt.legend(['X', '1 Sigma envelope'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
            f'{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (assim) Errors')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()
def print_xy(mc_number, agent_range, agents_mc, simulation_measurement_x,
                            simulation_measurement_y):
    for agent_idx in range(agent_range[0], agent_range[1]):

            # plot XY
            ax = plt.figure().add_subplot()
            ax.plot(squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 0], squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')
            ax.plot(simulation_measurement_x, simulation_measurement_y, '--r')
            ax.plot()
            plt.legend(['X estimation', 'X real'])
            plt.title(
                f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) XY state estimation (updated) and measurements')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

def print_3d(mc_number, agent_range, agents_mc, simulation_time,
                            simulation_measurement_x,simulation_measurement_y):
    for agent_idx in range(agent_range[0], agent_range[1]):

        # 3D
        # Plot XY in 3d
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 0], squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'b')
        ax.plot(simulation_time, simulation_measurement_x, simulation_measurement_y, '--r')
        # for agent_idx in range(agent_range[0], agent_range[1]):
        ax.scatter(simulation_time, agents_mc[mc_number][agent_idx].position[0, 0], agents_mc[mc_number][agent_idx].position[0, 1], 'green')
        plt.legend(['estimation', 'real', 'sensor position'])
        plt.title(f'agent {agents_mc[mc_number][agent_idx].id} X , Y in time')
        ax.set_xlabel('Time')
        ax.set_ylabel('$X$')
        ax.set_zlabel(r'$Y$')
        plt.show()

def print_3d_assimilation(mc_number, agent_range, agents_mc, simulation_time,
             simulation_measurement_x, simulation_measurement_y):
    for agent_idx in range(agent_range[0], agent_range[1]):

        # 3D
        # Plot XY in 3d
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 0],
                squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'b')
        ax.plot(simulation_time, simulation_measurement_x, simulation_measurement_y, '--r')
        for agent_idx in range(agent_range[0], agent_range[1]):
            ax.scatter(simulation_time, agents_mc[mc_number][agent_idx].position[0, 0],
                       agents_mc[mc_number][agent_idx].position[0, 1], 'green')
        plt.legend(['estimation', 'real', 'sensor position'])
        plt.title(f'agent {agents_mc[mc_number][agent_idx].id} X , Y in time')
        ax.set_xlabel('Time')
        ax.set_ylabel('$X$')
        ax.set_zlabel(r'$Y$')
        plt.show()
def plot_R_Q(mc_number, state_index, agent_range, agents_mc, simulation_time):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time,agents_mc[mc_number][agent_idx].filter.R_arr[:,0,0] ** .5, 'r')
        plt.plot(simulation_time,
                 np.ones_like(simulation_time) * agents_mc[mc_number][agent_idx].filter.Q[state_index, state_index] ** .5, '--k')
        plt.legend(['Measurement noise magnitude', 'Process noise magnitude'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
            f'{agents_mc[mc_number][agent_idx].position[0, 1]}) measurement noise vs. process noise')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()
def plot_estimation_error(mc_number, state_index, agent_range, agents_mc, simulation_time
                               , simulation_measurement):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time,squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:,state_index]
                - simulation_measurement, 'r')
        plt.plot(simulation_time,
                 agents_mc[mc_number][agent_idx].filter.assim_covs[:, state_index, state_index] ** .5, '--k')
        plt.plot(simulation_time,
                 -agents_mc[mc_number][agent_idx].filter.assim_covs[:, state_index, state_index] ** .5, '--k')
        plt.legend(['X', '1 Sigma envelope'])
        plt.legend(['mean estimation error'])
        plt.title(
            f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
            f'{agents_mc[mc_number][agent_idx].position[0, 1]}) Estimation error')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

def plot_mc_estimation_error(mc_number, state_index, agent_idx, agents_mc, simulation_time
                                 , simulation_measurement, number_of_mc_runs):
    mean_state = 0
    theoretical_P_arr = []
    estimation_error_arr = []
    for run in range(number_of_mc_runs):
        mean_state += (squeeze(agents_mc[run][agent_idx].filter.updated_state)[:,
                       state_index]) / number_of_mc_runs
        theoretical_P_arr.append(
            agents_mc[run][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5)
        estimation_error_arr.append(
            squeeze(agents_mc[run][agent_idx].filter.updated_state)[:, state_index] - simulation_measurement)

    ax = plt.figure().add_subplot()
    ax.plot(simulation_time, mean_state - simulation_measurement, 'r')
    ax.plot(simulation_time,
            squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:,
            state_index] - simulation_measurement, 'b')
    plt.plot(simulation_time,
             agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.plot(simulation_time,
             -agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.legend(['estimation error MC', 'estimation error single', 'i sigma envalope'])
    plt.title(
        f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'
        f'{agents_mc[mc_number][agent_idx].position[0, 1]}) Estimation Error Vs. MC Mean Estimation Error')
    plt.show()

    theoretical_P_mean = np.mean(theoretical_P_arr, axis=0)
    ax = plt.figure().add_subplot()
    ax.plot(simulation_time, theoretical_P_mean, 'b')
    plt.plot(simulation_time,
             agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.plot(simulation_time,
             -agents_mc[mc_number][agent_idx].filter.updated_covs[:, state_index, state_index] ** .5, '--k')
    plt.legend(['MC calculated STD', '1 Sigma envelope'])
    MC_P_std = np.std(estimation_error_arr, axis=0)
    plt.show
    return MC_P_std, theoretical_P_mean
def plot_name(state_index):
    if state_index == 0:
        return 'X'

    elif state_index == 1:
        return 'X_dot'

    elif state_index == 2:
        return 'Y'

    elif state_index == 3:
        return 'Y_dot'
