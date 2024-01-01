import matplotlib.pyplot as plt
import numpy as np
import os
# from main_EKF import agents,T,X,Y , agents_mc
from numpy import squeeze
from scipy.interpolate import make_interp_spline


unit_arr= ['Position[M]' , 'Velocity[M/sec]' , 'Acceleration[M/sec^2]']
# plot noised measurements vs original measurements
def print_measurement_comparison(agents_mc, mc_number, agent_range, simulation_time):
    for agent_idx in range(agent_range[0], agent_range[1]):
        ax = plt.figure().add_subplot()
        ax.plot(simulation_time, agents_mc[mc_number][agent_idx].measurements, 'r')
        ax.plot(simulation_time, agents_mc[mc_number][agent_idx].measurements_clean, 'b')
        plt.legend(['Measured distance', 'Real distance'])
        plt.title(f'Distance effect on measurement noise')
        plt.xlabel('Time[sec]')
        plt.ylabel('Distance [M]')
        plt.grid(color='k', linestyle='--', linewidth=.2)
        plt.show()



def print_residual(agents_mc, mc_number, agent_range, simulation_time, measurement_dim):
    # plot residual (z - Hx)
    for agent_idx in range(agent_range[0], agent_range[1]):
        fig, axs = plt.subplots(measurement_dim)
        for index in range(measurement_dim):
            axs[index].plot(simulation_time, agents_mc[mc_number][agent_idx].filter.residual_arr[index::measurement_dim], 'r')
            axs[index].set_xlabel('Time [sec]')
            axs[index].set_ylabel(unit_arr[index])
            fig.suptitle(
                f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) Residual error')
            axs[index].grid(color='k', linestyle='--', linewidth=.2)
            fig.show()

def print_updated_state(mc_number, agent_range, agents_mc, simulation_time, simulation_measurement , mode , save_figs , fig_save_path):
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot X
        fig, axs = plt.subplots(3)
        axs[0].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 0], 'r')
        axs[1].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 1], 'r')
        if mode == 'acceleration' :axs[2].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')

        axs[0].plot(simulation_time, simulation_measurement[0], '--b')
        axs[1].plot(simulation_time, simulation_measurement[1], '--b')
        axs[2].plot(simulation_time, simulation_measurement[2], '--b')

        axs[0].legend(['position estimation', 'position real'])
        axs[1].legend(['velocity estimation', 'velocity real'])
        axs[2].legend(['acceleration real'])
        axs[0].set_xlabel('Time[Sec]')
        axs[0].set_ylabel(unit_arr[0])
        axs[1].set_xlabel('Time[Sec]')
        axs[1].set_ylabel(unit_arr[1])
        axs[2].set_xlabel('Time[Sec]')
        axs[2].set_ylabel(unit_arr[2])
        # if mode == 'acceleration' :axs[2].set_xlabel('Time[Sec]')
        # if mode == 'acceleration' :axs[2].set_ylabel(unit_arr[2])

        axs[0].set_title('Position')
        axs[1].set_title('Velocity')
        axs[2].set_title('Acceleration')

        axs[0].grid(color='k', linestyle='--', linewidth=.2)
        axs[1].grid(color='k', linestyle='--', linewidth=.2)
        axs[2].grid(color='k', linestyle='--', linewidth=.2)
        plt.subplots_adjust(hspace=0.75)
        fig.set_size_inches((8.5, 11), forward=False)

        # fig.suptitle(f'Sensor {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) state estimation vs simulation data')
        fig.suptitle(f'Sensor at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) state estimation')
        if save_figs == 1:
            # manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            fig.savefig(os.path.join(fig_save_path , f'agent{agent_idx}') , bbox_inches='tight' , dpi = 500)
def print_updated_covariance(mc_number, agent_range, agents_mc, simulation_time,
                            simulation_measurement , mode):
    for agent_idx in range(agent_range[0], agent_range[1]):
        num_of_subplots = 3 if mode == 'acceleration' else 2
        fig, axs = plt.subplots(num_of_subplots)
        for index in range(num_of_subplots):
            axs[index].plot(simulation_time,simulation_measurement[index]-squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, index] , 'r')
            axs[index].plot(simulation_time,agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,3*agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,-agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,-3*agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--k')
            axs[index].legend([f'X {index+1}state','1 Sigma envelope','3 Sigma envelope'])
            axs[index].set_xlabel('Time[Sec]')
            axs[index].set_ylabel(unit_arr[index])
            axs[index].set_title(f'X {index+1} state estimation error in 1,3 Sigma envelopes')
            axs[index].grid(color='k', linestyle='--', linewidth=.2)
            # axs[index].set_ylim([-1, 1])


        fig.suptitle(f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'f'{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (updated) Errors')

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
        plt.grid(color='k', linestyle='--', linewidth=.2)
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
        plt.grid(color='k', linestyle='--', linewidth=.2)
        plt.show()

def print_assimilated_state(mc_number, state_index, agent_range, agents_mc, simulation_time, simulation_measurement , mode, save_figs , fig_save_path):
    # for agent_idx in range(agent_range[0], agent_range[1]):
    #     # plot X
    #     ax = plt.figure().add_subplot()
    #     ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, state_index], 'r')
    #     ax.plot(simulation_time, simulation_measurement, '--r')
    #     plt.legend(['X estimation', 'X real'])
    #     plt.title(
    #         f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (assim) and measurements')
    #     plt.xlabel('time')
    #     plt.ylabel('amplitude')
    #     plt.show()
    # for agent_idx in range(agent_range[0], agent_range[1]):
    #     # plot X
    #     num_of_subplots = 3 if mode == 'acceleration' else 2
    #     fig, axs = plt.subplots(num_of_subplots)
    #     for index in range(num_of_subplots):
    #
    #         axs[index].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, index], 'r')
    #         # axs[index].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, index], 'r')
    #         # if mode == 'acceleration' :axs[2].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')
    #
    #         axs[index].plot(simulation_time, simulation_measurement[index], '--r')
    #
    #
    #         if index == 0:
    #             axs[0].legend(['X position estimation', 'X position real'])
    #         if index == 1:
    #             axs[1].legend(['X velocity estimation', 'X velocity real'])
    #         if index == 2:
    #             axs[2].legend(['X acceleration estimation', 'X acceleration real'])
    #         axs[index].set_xlabel('Time[Sec]')
    #         axs[index].set_ylabel(unit_arr[index])
    #         # if mode == 'acceleration' :axs[2].set_xlabel('Time[Sec]')
    #         # if mode == 'acceleration' :axs[2].set_ylabel(unit_arr[2])
    #
    #         axs[index].set_title('Position estimation vs simulation')
    #
    #
    #         axs[index].grid(color='k', linestyle='--', linewidth=.2)

        # fig.suptitle(f'agent {agents_mc[mc_number][agent_idx].id} state estimation vs simulation data')

    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot X
        fig, axs = plt.subplots(3)
        axs[0].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, 0], 'r')
        axs[1].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, 1], 'r')
        if mode == 'acceleration' :axs[2].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')

        axs[0].plot(simulation_time, simulation_measurement[0], '--r')
        axs[1].plot(simulation_time, simulation_measurement[1], '--r')
        axs[2].plot(simulation_time, simulation_measurement[2], '--r')

        axs[0].legend(['X position estimation', 'X position real'])
        axs[1].legend(['X velocity estimation', 'X velocity real'])
        axs[2].legend([ 'X acceleration real'])
        axs[0].set_xlabel('Time[Sec]')
        axs[0].set_ylabel(unit_arr[0])
        axs[1].set_xlabel('Time[Sec]')
        axs[1].set_ylabel(unit_arr[1])
        axs[2].set_xlabel('Time[Sec]')
        axs[2].set_ylabel(unit_arr[2])
        # if mode == 'acceleration' :axs[2].set_xlabel('Time[Sec]')
        # if mode == 'acceleration' :axs[2].set_ylabel(unit_arr[2])

        axs[0].set_title('Position estimation vs simulation')
        axs[1].set_title('Velocity estimation vs simulation')
        axs[2].set_title('Acceleration estimation vs simulation')

        axs[0].grid(color='k', linestyle='--', linewidth=.2)
        axs[1].grid(color='k', linestyle='--', linewidth=.2)
        axs[2].grid(color='k', linestyle='--', linewidth=.2)
        plt.subplots_adjust(hspace=0.5)
        fig.set_size_inches((18.5, 11), forward=False)

        fig.suptitle(f'agent {agents_mc[mc_number][agent_idx].id} at position ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) state estimation vs simulation data')
        if save_figs == 1:
            # manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            fig.savefig(os.path.join(fig_save_path , f'agent{agent_idx}') , bbox_inches='tight' , dpi = 500)

def print_assimilated_covariance(mc_number, agent_range, agents_mc, simulation_time,
                            simulation_measurement , mode):
    for agent_idx in range(agent_range[0], agent_range[1]):
        num_of_subplots = 3 if mode == 'acceleration' else 2
        fig, axs = plt.subplots(num_of_subplots)
        for index in range(num_of_subplots):
            axs[index].plot(simulation_time,simulation_measurement[index]-squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, index] , 'r')
            axs[index].plot(simulation_time,agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,3*agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,-agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,-3*agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--k')
            axs[index].legend([f'X {index+1}state','1 Sigma envelope','3 Sigma envelope'])
            axs[index].set_xlabel('Time[Sec]')
            axs[index].set_ylabel(unit_arr[index])
            axs[index].set_title(f'X {index+1} state estimation error in 1,3 Sigma envelopes')
            axs[index].grid(color='k', linestyle='--', linewidth=.2)
            axs[index].set_ylim([-1, 1])


        fig.suptitle(f'agent {agents_mc[mc_number][agent_idx].id} at ({agents_mc[mc_number][agent_idx].position[0, 0]},'f'{agents_mc[mc_number][agent_idx].position[0, 1]}) X state estimation (assim) Errors')

def print_xy(mc_number, agent_range, agents_mc, simulation_measurement_x,
                            simulation_measurement_y , start_idx,mode):
    y_position_state = 2 if mode == 'velocity' else 3
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot XY
            ax = plt.figure().add_subplot()
            ax.plot(squeeze(agents_mc[mc_number][agent_idx].filter.predicted_state)[start_idx:, 0], squeeze(agents_mc[mc_number][agent_idx].filter.predicted_state)[start_idx:, 2], 'r')
            ax.plot(simulation_measurement_x, simulation_measurement_y, '--b')
            ax.plot()
            plt.legend(['X estimation', 'X real'])
            plt.title(
                f'sensor at ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) XY state estimation')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(color='k', linestyle='--', linewidth=.2)
            plt.show()




def print_3d(mc_number, agent_range, agents_mc, simulation_time,
                            simulation_measurement_x,simulation_measurement_y,start_index):
    for agent_idx in range(agent_range[0], agent_range[1]):

        # 3D
        # Plot XY in 3d
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[start_index:, 0], squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[start_index:, 2], 'b')
        ax.plot(simulation_time, simulation_measurement_x, simulation_measurement_y, '--r')
        # for agent_idx in range(agent_range[0], agent_range[1]):
        ax.scatter(simulation_time, agents_mc[mc_number][agent_idx].position[0, 0], agents_mc[mc_number][agent_idx].position[0, 1], 'green')
        plt.legend(['estimation', 'real', 'sensor position'])
        plt.title(f'agent {agents_mc[mc_number][agent_idx].id} X , Y in time')
        ax.set_xlabel('Time')
        ax.set_ylabel('$X$')
        ax.set_zlabel(r'$Y$')
        plt.grid(color='k', linestyle='--', linewidth=.2)
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
        plt.grid(color='k', linestyle='--', linewidth=.2)
        plt.show()


def plot_mc_estimation_error(mc_number, state_index, agent_idx, simulation_time
                             , simulation_measurement, number_of_mc_runs, Q_arr, experiments, start_index):
    mean_state = 0
    theoretical_P_arr = []
    estimation_error_arr = []
    for experiment in range(len(experiments)):

        # mean_state += (squeeze(experiments[experiment][run][agent_idx].filter.predicted_state)[:,state_index])/number_of_mc_runs
        theoretical_P_arr.append(np.array(
            [experiments[experiment][run][agent_idx].filter.predicted_covs[:, state_index, state_index] ** .5 for run in
             range(number_of_mc_runs)]))
        estimation_error_arr.append(np.array([squeeze(experiments[experiment][run][agent_idx].filter.predicted_state)[:,
                                              state_index] - simulation_measurement for run in
                                              range(number_of_mc_runs)]))

    theoretical_P_mean = [np.mean(theoretical_P, axis=0) for theoretical_P in theoretical_P_arr]
    MC_P_std = [np.std(estimation_error, axis=0) for estimation_error in estimation_error_arr]
    ax = plt.figure().add_subplot()
    ax.plot(simulation_time, MC_P_std[mc_number], 'b')
    plt.plot(simulation_time,
             theoretical_P_mean[mc_number], '--k')
    plt.plot(simulation_time,
             -theoretical_P_mean[mc_number], '--k')

    plt.legend(['MC calculated STD', '1 Sigma envelope'])
    plt.title(f'agent MC calculated estimation error vs. theoretically calculated estimation error')
    plt.grid(color='k', linestyle='--', linewidth=.2)

    plt.show()

    MSE = [np.mean((theoretical_P_mean[experiment][start_index:] - MC_P_std[experiment][start_index:]) ** 2) for
           experiment in range(len(experiments))]
    ax = plt.figure().add_subplot()
    ax.plot(Q_arr, MSE, 'b')
    plt.legend(['MC calculated STD'])
    plt.title(f'MSE error as a function of process noise intensity')
    plt.grid(color='k', linestyle='--', linewidth=.2)
    plt.show()


def plot_name(state_index):
    if state_index == 0:
        return 'X'

    elif state_index == 1:
        return 'X_dot'

    elif state_index == 2:
        return 'Y'

    elif state_index == 3:
        return 'Y_dot'


def plot_mc_estimation_error_all(mc_number, agent_idx, simulation_time, simulation_measurement,
                             number_of_mc_runs, Q_arr, experiments, start_index , mode, is_assim):
    mean_state = 0
    theoretical_P_arr = []
    estimation_error_arr = []
    num_of_subplots = 3 if mode == 'acceleration' else 2
    fig, axs = plt.subplots(num_of_subplots)
    fig1, axs1 = plt.subplots(num_of_subplots)
    fig2 , axs2 = plt.subplots(1)
    for index in range(num_of_subplots):
        for experiment in range(len(experiments)):
            if is_assim:
                estimation_error_arr.append(np.array([squeeze(experiments[experiment][run][agent_idx].filter.assim_state)[:,
                                                      index] - simulation_measurement[index] for run in range(number_of_mc_runs)]))
                theoretical_P_arr.append(
                    np.array([experiments[experiment][run][agent_idx].filter.assim_covs[:, index, index]
                              ** .5 for run in range(number_of_mc_runs)]))



            else:
                theoretical_P_arr.append(np.array([experiments[experiment][run][agent_idx].filter.updated_covs[:, index, index]
                                                   ** .5 for run in range(number_of_mc_runs)]))
                estimation_error_arr.append(np.array([squeeze(experiments[experiment][run][agent_idx].filter.updated_state)[:,
                                                      index] - simulation_measurement[index] for run in range(number_of_mc_runs)]))
        theoretical_P_mean = [np.mean(theoretical_P, axis=0) for theoretical_P in theoretical_P_arr]
        MC_err_mean = np.mean(np.expand_dims(estimation_error_arr , 1) , axis=2)
        MC_err_mean1 = MC_err_mean[0,0,start_index:]
        RMSE = np.mean((MC_err_mean1**2)**.5)
        if index ==0:
            axs2.grid(color='k', linestyle='--', linewidth=.2)
            axs2.set_title('mean estimation error')
            axs2.set_xlabel('Time [Sec]')
            axs2.set_ylabel('Error [M]')
            axs2.plot(simulation_time , MC_err_mean[0 ,0, :])

        # axs2.plot(simulation_time , squeeze(experiments[experiment][0][agent_idx].filter.updated_covs[: ,0  ,0]),'--k')
        # axs2.plot(simulation_time , -squeeze(experiments[experiment][0][agent_idx].filter.updated_covs[: ,0  ,0]),'--k')
        MC_P_std = [np.std(estimation_error, axis=0) for estimation_error in estimation_error_arr]
        axs[index].plot(simulation_time, MC_P_std[mc_number], 'b')
        axs[index].plot(simulation_time,theoretical_P_mean[mc_number], '--k')
        axs[index].plot(simulation_time,-theoretical_P_mean[mc_number], '--k')
        axs[index].set_ylabel('unit_arr[index]')
        axs[index].set_xlabel('Time[Sec]')
        if index == 0:
            axs[index].set_title('MC Position estimation error')
        elif index == 1:
            axs[index].set_title('MC Velocity estimation error')
        elif index == 2:
            axs[index].set_title('MC Acceleration estimation error')

        axs[index].grid(color='k', linestyle='--', linewidth=.2)
        axs[index].legend(['MC calculated STD', '1 Sigma envelope'])
        fig.suptitle(f'agent MC calculated estimation error vs. theoretically calculated estimation error')

        # plt.suptitle('MC Estimation errors')

        MSE = [np.mean((theoretical_P_mean[experiment][start_index:] - MC_P_std[experiment][start_index:]) ** 2) for
               experiment in range(len(experiments))]

        # axs[index] = plt.figure().add_subplot()
        axs1[0].plot(Q_arr, MSE)
        fig1.suptitle(f'MSE of STD: filter computatuions with actual results')
        axs1[0].set_ylabel('MSE')
        axs1[0].set_xlabel('Q - process noise intensity')
        axs1[0].legend(['state 1 - position' , 'state 2 - velocity'])
        axs1[0].grid(color='k', linestyle='--', linewidth=.2)

        # plt.suptitle(f'MSE error as a function of process noise intensity')
        theoretical_P_arr = []
        estimation_error_arr = []
        return RMSE
y =np.array([0.245718385,
0.137999378,
0.091338511,
0.047106525,
0.036996647,
0.026253146,
])
x = np.array([
1,
2,
4,
8,
16,
32,

])
X_Y_Spline = make_interp_spline(x, y)
X_ = np.linspace(x[0], x[-1], 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_)
plt.plot(x, y)
plt.title('Effect of sensor count on estimation quality')
plt.xlabel('Sensor count')
plt.ylabel('Estimation RMSE')
plt.grid()
plt.show()