import matplotlib.pyplot as plt
import numpy as np
import os
import  alg_utils
import pandas as pd
# from main_EKF import agents,T,X,Y , agents_mc
from numpy import squeeze
from scipy.interpolate import make_interp_spline

units = ['Position', 'Velocity', 'Acceleration']

unit_arr= ['Position[M]' , 'Velocity[M/sec]' , 'Acceleration[M/sec^2]']
# plot noised measurements vs original measurements
def Fprint_measurement_comparison(agents_mc, mc_number, agent_range, simulation_time):
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
        # fig, axs = plt.subplots(3)
        fig1, axs1 = plt.subplots(3,2)
        # axs[0].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 0], 'r')
        # axs[1].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 1], 'r')
        # if mode == 'acceleration' :axs[2].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')
        #
        # axs[0].plot(simulation_time, simulation_measurement[0], '--b')
        # axs[1].plot(simulation_time, simulation_measurement[1], '--b')
        # axs[2].plot(simulation_time, simulation_measurement[2], '--b')
        #
        # axs[0].legend([' Estimation', ' Real'])
        # axs[1].legend([' Estimation', ' Real'])
        # axs[2].legend([' Real'])
        # axs[0].set_xlabel('Time[Sec]')
        # axs[0].set_ylabel(unit_arr[0])
        # axs[1].set_xlabel('Time[Sec]')
        # axs[1].set_ylabel(unit_arr[1])
        # axs[2].set_xlabel('Time[Sec]')
        # axs[2].set_ylabel(unit_arr[2])
        #
        # axs[0].set_title('Position')
        # axs[1].set_title('Velocity')
        # axs[2].set_title('Acceleration')
        #
        # axs[0].grid(color='k', linestyle='--', linewidth=.2)
        # axs[1].grid(color='k', linestyle='--', linewidth=.2)
        # axs[2].grid(color='k', linestyle='--', linewidth=.2)
        # plt.subplots_adjust(hspace=0.75)
        # fig.set_size_inches((8.5, 11), forward=False)
        # fig.suptitle(f'State estimation')

        axs1[0,0].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 0], 'r')
        axs1[1,0].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 1], 'r')

        axs1[0,0].plot(simulation_time, simulation_measurement[0], '--b')
        axs1[1,0].plot(simulation_time, simulation_measurement[1], '--b')

        axs1[0,1].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')
        axs1[1,1].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 3], 'r')

        axs1[0,1].plot(simulation_time, simulation_measurement[0], '--b')
        axs1[1,1].plot(simulation_time, simulation_measurement[1], '--b')

        axs1[2,0].plot(simulation_time, simulation_measurement[2], '--b')
        axs1[2,1].plot(simulation_time, simulation_measurement[2], '--b')

        axs1[0,0].legend([' Estimation', ' Real'])
        axs1[1,0].legend([' Estimation', ' Real'])
        axs1[0,1].legend([' Estimation', ' Real'])
        axs1[1,1].legend([' Estimation', ' Real'])

        axs1[0,0].set_title('Position - X')
        axs1[1,0].set_title('Velocity - X')
        axs1[0,1].set_title('Position - Y')
        axs1[1,1].set_title('Velocity - Y')
        axs1[2,0].set_title('Acceleration - X')
        axs1[2,1].set_title('Acceleration - Y')

        axs1[0,0].set_xlabel('Time[Sec]')
        axs1[0,0].set_ylabel(unit_arr[0])
        axs1[1,0].set_xlabel('Time[Sec]')
        axs1[1,0].set_ylabel(unit_arr[1])
        axs1[2,0].set_xlabel('Time[Sec]')
        axs1[2,1].set_ylabel(unit_arr[2])

        axs1[0,1].set_xlabel('Time[Sec]')
        axs1[0,1].set_ylabel(unit_arr[0])
        axs1[1,1].set_xlabel('Time[Sec]')
        axs1[1,1].set_ylabel(unit_arr[1])
        axs1[2,0].set_xlabel('Time[Sec]')
        axs1[2,1].set_ylabel(unit_arr[2])

        axs1[0,0].grid(color='k', linestyle='--', linewidth=.2)
        axs1[1,0].grid(color='k', linestyle='--', linewidth=.2)
        axs1[0,1].grid(color='k', linestyle='--', linewidth=.2)
        axs1[1,1].grid(color='k', linestyle='--', linewidth=.2)
        axs1[2,0].grid(color='k', linestyle='--', linewidth=.2)
        axs1[2,1].grid(color='k', linestyle='--', linewidth=.2)

        fig1.suptitle(f'State estimation')

        if save_figs == 1:
            # manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            fig.savefig(os.path.join(fig_save_path , f'agent{agent_idx}') , bbox_inches='tight' , dpi = 500)
def print_updated_covariance(mc_number, agent_range, agents_mc, simulation_time,
                            simulation_measurement , mode):
    for agent_idx in range(agent_range[0], agent_range[1]):
        num_of_subplots = 3 if mode == 'acceleration' else 2
        # fig, axs = plt.subplots(num_of_subplots)
        fig1,axs1 = plt.subplots(num_of_subplots,num_of_subplots)
        # for index in range(num_of_subplots):
        #     axs[index].plot(simulation_time,-simulation_measurement[index]+squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, index] , 'r')
        #     axs[index].plot(simulation_time,agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--k')
        #     axs[index].plot(simulation_time,3*agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--b')
        #     axs[index].plot(simulation_time,-agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--k')
        #     axs[index].plot(simulation_time,-3*agents_mc[mc_number][agent_idx].filter.updated_covs[:, index, index] ** .5, '--b')
        #     axs[0].legend([f'Estimation','1 Sigma envelope','3 Sigma envelope'])
        #     axs[index].set_xlabel('Time[Sec]')
        #     axs[index].set_ylabel(unit_arr[index])
        #     axs[index].set_title(f'{units[index]} estimation error')
        #     axs[index].grid(color='k', linestyle='--', linewidth=.2)
        #
        # fig.suptitle(f'Sensor estimation  Errors - X coordinates')
        for index in range(num_of_subplots):
            for index1 in range(num_of_subplots):
                idx = index+index1
                if index1 == 1 : idx += 1
                if idx>3:idx = 3
                axs1[index,index1].plot(simulation_time,simulation_measurement[idx]-squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, idx] , 'r')
                axs1[index,index1].plot(simulation_time,agents_mc[mc_number][agent_idx].filter.updated_covs[:, idx, idx] ** .5, '--k')
                axs1[index,index1].plot(simulation_time,3*agents_mc[mc_number][agent_idx].filter.updated_covs[:, idx, idx] ** .5, '--b')
                axs1[index,index1].plot(simulation_time,-agents_mc[mc_number][agent_idx].filter.updated_covs[:, idx, idx] ** .5, '--k')
                axs1[index,index1].plot(simulation_time,-3*agents_mc[mc_number][agent_idx].filter.updated_covs[:, idx, idx] ** .5, '--b')
                axs1[index,index1].legend([f'Estimation','1 Sigma envelope','3 Sigma envelope'])
                axs1[index,index1].set_xlabel('Time[Sec]')
                axs1[index,index1].set_ylabel(unit_arr[index])
                axs1[index,index1].set_title(f'{units[index]} estimation error')
                axs1[index,index1].grid(color='k', linestyle='--', linewidth=.2)

        fig1.suptitle(f'Sensor estimation  Errors')

        # fig.suptitle(f'sonsor at position ({agents_mc[mc_number][agent_idx].position[0, 0]},'f'{agents_mc[mc_number][agent_idx].position[0, 1]}) estimation  Errors')

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

    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot X
        fig, axs = plt.subplots(2)
        axs[0].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, 0], 'r')
        axs[1].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, 1], 'r')
        # if mode == 'acceleration' :axs[2].plot(simulation_time, squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[:, 2], 'r')

        axs[0].plot(simulation_time, simulation_measurement[0], '--b')
        axs[1].plot(simulation_time, simulation_measurement[1], '--b')
        # axs[2].plot(simulation_time, simulation_measurement[2], '--b')

        axs[0].legend([' estimation', ' real'])
        axs[1].legend([' estimation', ' real'])
        # axs[2].legend([ 'acceleration real'])
        axs[0].set_xlabel('Time[Sec]')
        axs[0].set_ylabel(unit_arr[0])
        axs[1].set_xlabel('Time[Sec]')
        axs[1].set_ylabel(unit_arr[1])
        # axs[2].set_xlabel('Time[Sec]')
        # axs[2].set_ylabel(unit_arr[2])
        # if mode == 'acceleration' :axs[2].set_xlabel('Time[Sec]')
        # if mode == 'acceleration' :axs[2].set_ylabel(unit_arr[2])

        axs[0].set_title('Position estimation - X')
        axs[1].set_title('Velocity estimation - X')
        # axs[2].set_title('Acceleration')

        axs[0].grid(color='k', linestyle='--', linewidth=.2)
        axs[1].grid(color='k', linestyle='--', linewidth=.2)
        # axs[2].grid(color='k', linestyle='--', linewidth=.2)
        plt.subplots_adjust(hspace=0.5)
        fig.set_size_inches((18.5, 11), forward=False)

        fig.suptitle(f'Assimilation state estimation ')
        if save_figs == 1:
            # manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            fig.savefig(os.path.join(fig_save_path , f'agent{agent_idx}') , bbox_inches='tight' , dpi = 500)

def print_assimilated_covariance(mc_number, agent_range, agents_mc, simulation_time,
                            simulation_measurement , mode):
    units = ['Position', 'Velocity', 'Acceleration']

    for agent_idx in range(agent_range[0], agent_range[1]):
        num_of_subplots = 3 if mode == 'acceleration' else 2
        fig, axs = plt.subplots(num_of_subplots)
        for index in range(num_of_subplots):
            axs[index].plot(simulation_time,simulation_measurement[index]-squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[:, index] , 'r')
            axs[index].plot(simulation_time,agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,3*agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--b')
            axs[index].plot(simulation_time,-agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--k')
            axs[index].plot(simulation_time,-3*agents_mc[mc_number][agent_idx].filter.assim_covs[:, index, index] ** .5, '--b')
            axs[index].legend([f'Estimation error','1 Sigma envelope','3 Sigma envelope'])
            axs[index].set_xlabel('Time[Sec]')
            axs[index].set_ylabel(unit_arr[index])
            axs[index].set_title(f'{units[index]} estimation error - X')
            axs[index].grid(color='k', linestyle='--', linewidth=.2)
            # axs[index].set_ylim([-1, 1])


        fig.suptitle(f'Assimilation state estimation Errors')

def print_xy(mc_number, agent_range, agents_mc, simulation_measurement_x,
                            simulation_measurement_y , start_idx,mode):
    y_position_state = 2 if mode == 'velocity' else 3
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot XY
            ax = plt.figure().add_subplot()
            ax.plot(squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[start_idx:, 0], squeeze(agents_mc[mc_number][agent_idx].filter.updated_state)[start_idx:, 2], 'or')
            ax.plot(simulation_measurement_x, simulation_measurement_y, '--b')
            for agent_idx2 in range(agent_range[0], agent_range[1]):
                try:
                    ax.plot(agents_mc[mc_number][agent_idx2].position[0,0] , agents_mc[mc_number][agent_idx2].position[0,1] , 'ok')
                except:
                    pass
            plt.legend(['Estimated position', 'Real position' , 'Sensor location'])
            plt.title(
                f'sensor at ({agents_mc[mc_number][agent_idx].position[0, 0]},{agents_mc[mc_number][agent_idx].position[0, 1]}) XY position estimation')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(color='k', linestyle='--', linewidth=.2)
            plt.show()

def print_xy_assim(mc_number, agent_range, agents_mc, simulation_measurement_x,
                            simulation_measurement_y , start_idx,mode):
    y_position_state = 2 if mode == 'velocity' else 3
    for agent_idx in range(agent_range[0], agent_range[1]):
        # plot XY
        ax = plt.figure().add_subplot()
        ax.plot(squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[start_idx:, 0], squeeze(agents_mc[mc_number][agent_idx].filter.assim_state)[start_idx:, 2], 'or')
        ax.plot(simulation_measurement_x, simulation_measurement_y, '--b')
        for agent_idx2 in range(agent_range[0], agent_range[1]):
            ax.plot(agents_mc[mc_number][agent_idx2].position[0,0] , agents_mc[mc_number][agent_idx2].position[0,1] , 'ok')
        plt.legend(['Estimated position', 'Real position' , 'Sensor location'])
        plt.title(
            f'XY Position estimation')
        plt.xlabel('X [M]')
        plt.ylabel('Y [M]')
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
        # ax = plt.figure().add_subplot()
        plt.figure()
        plt.plot(simulation_time,agents_mc[mc_number][agent_idx].filter.R_arr[:,0,0] ** .5, 'r')
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



        MC_P_std = [np.std(estimation_error, axis=0) for estimation_error in estimation_error_arr]
        axs[index].plot(simulation_time, MC_P_std[mc_number], 'b')
        axs[index].plot(simulation_time,theoretical_P_mean[mc_number], '--k')
        # if index == 1:
            # pd.DataFrame(MC_P_std[mc_number]).to_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\STD_OF_MC_five_sensors_vel_assim.csv')
            # pd.DataFrame(theoretical_P_mean[mc_number]).to_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\P_five_sensors_vel_assim.csv')
        # axs[index].plot(simulation_time,-theoretical_P_mean[mc_number], '--k')
        axs[index].set_ylabel(f'{unit_arr[index]}')
        axs[index].set_xlabel('Time[Sec]')
        if index == 0:
            axs[index].set_title('MC Position estimation error')
        elif index == 1:
            axs[index].set_title('MC Velocity estimation error')
        elif index == 2:
            axs[index].set_title('MC Acceleration estimation error')

        axs[index].grid(color='k', linestyle='--', linewidth=.2)
        axs[index].legend(['MC calculated STD', '1 Sigma envelope'])
        fig.suptitle(f'Estimation statistics: MC estimation error')

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

def agents_mean_vs_agents_assim(mc_number,number_of_agents, simulation_time, simulation_measurement,
                             number_of_mc_runs, experiments, start_index , mode, is_assim , path , state_index ,
                                angle_flag , distance_flag , assim_type , line_distance):
    index = state_index
    state_arr = []
    err_arr = []
    for experiment in range(len(experiments)):
        for agent_idx in range(number_of_agents):
            if is_assim:
                state_arr.append(sum(np.array([experiments[experiment][run][agent_idx].filter.assim_state[:, index]
                           for run in range(number_of_mc_runs)]))/number_of_mc_runs)
            else:
                state_arr.append(sum(np.array([experiments[experiment][run][agent_idx].filter.updated_state[:, index]
                                           for run in range(number_of_mc_runs)]))/number_of_mc_runs)

    mean_state = sum(state_arr) / (len(experiments)*number_of_agents)
    path = r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures'
    pd.DataFrame(mean_state).to_csv(os.path.
                join(path , f'mean_state_assim_{is_assim}_agents_{number_of_agents}_num_of_mc_{number_of_mc_runs}_angle_{angle_flag}_distance_{distance_flag}_assim_type_{assim_type}_state_index_{index}_line_distance_{line_distance}.csv'))
    plt.figure()
    plt.plot(simulation_time , simulation_measurement[index ,: ] , '--b')
    plt.plot(simulation_time , mean_state , 'r')
    plt.grid()
    plt.xlabel('Time [S]')
    plt.ylabel('Position [M]')
    plt.title('Sensors assimilation Vs. mean ')
    plt.legend(['Real' , 'Estimation'])

    plt.figure()
    plt.plot(simulation_time , mean_state - simulation_measurement[index ,: ] , 'r')
    plt.grid()
    plt.xlabel('Time [S]')
    plt.ylabel('Position [M]')
    plt.title('Sensors estimation error, assimilation Vs. mean ')
    plt.legend(['Real' , 'Estimation'])

def print_sensors_error(T,X):
    fig, axs = plt.subplots(2)
    fig1 , axs1 = plt.subplots(2)

    no_angle_dist_05_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_0.5.csv')
    no_angle_dist_06_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_0.6.csv')
    no_angle_dist_07_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_0.7.csv')
    no_angle_dist_08_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_0.8.csv')
    no_angle_dist_09_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_0.9.csv')
    no_angle_dist_1_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_1.csv')
    no_angle_dist_2_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_2.csv')
    no_angle_dist_3_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_3.csv')
    no_angle_dist_4_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_4.csv')
    no_angle_dist_5_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_5.csv')
    no_angle_dist_10_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_10.csv')
    no_angle_dist_15_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_15.csv')
    no_angle_dist_20_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_20.csv')
    no_angle_dist_25_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_25.csv')
    no_angle_dist_30_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_30.csv')
    no_angle_dist_35_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_35.csv')
    no_angle_dist_40_pos =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_0_line_distance_40.csv')

    no_angle_dist_05_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_1.csv')
    no_angle_dist_06_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_1.csv')
    no_angle_dist_07_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_1.csv')
    no_angle_dist_08_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_1.csv')
    no_angle_dist_09_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_1.csv')
    no_angle_dist_1_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_1.csv')
    no_angle_dist_2_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_2.csv')
    no_angle_dist_3_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_3.csv')
    no_angle_dist_4_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_4.csv')
    no_angle_dist_5_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_5.csv')
    no_angle_dist_10_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_10.csv')
    no_angle_dist_15_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_15.csv')
    no_angle_dist_20_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_20.csv')
    no_angle_dist_25_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_25.csv')
    no_angle_dist_30_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_30.csv')
    no_angle_dist_35_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_35.csv')
    no_angle_dist_40_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_0_assim_type_article_state_index_1_line_distance_40.csv')


    runs_1_agents_2_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_2_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_2_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_2_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_3_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_3_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_3_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_3_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_3_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_3_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_3_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_3_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_4_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_4_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_4_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_4_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_5_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_5_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_5_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_5_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_5_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_6_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_6_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_6_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_6_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_7_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_7_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_7_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_7_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_7_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_7_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_7_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_7_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_8_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_8_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_8_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_8_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_9_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_9_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_9_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_9_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_9_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_9_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_9_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_9_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_10_article =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_article.csv')
    runs_1_agents_10_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    runs_1_agents_10_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    runs_1_agents_10_mean_no_assim =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_0.csv')

    runs_1_agents_2_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_3_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_3_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_4_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_5_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_5_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_6_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_7_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_7_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_8_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_9_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_9_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')
    runs_1_agents_10_mean_no_covs =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_0_no_covs.csv')



    runs_1_agents_2_article_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_article_state_index_1.csv')
    runs_1_agents_4_article_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_article_state_index_1.csv')
    runs_1_agents_6_article_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_article_state_index_1.csv')
    runs_1_agents_8_article_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_article_state_index_1.csv')
    runs_1_agents_10_article_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_article_state_index_1.csv')

    runs_1_agents_2_mean_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1.csv')
    runs_1_agents_4_mean_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1.csv')
    runs_1_agents_6_mean_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1.csv')
    runs_1_agents_8_mean_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1.csv')
    runs_1_agents_10_mean_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1.csv')

    runs_1_agents_2_min_P_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_min_P_state_index_1.csv')
    runs_1_agents_4_min_P_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_min_P_state_index_1.csv')
    runs_1_agents_6_min_P_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_min_P_state_index_1.csv')
    runs_1_agents_8_min_P_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_min_P_state_index_1.csv')
    runs_1_agents_10_min_P_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_min_P_state_index_1.csv')

    runs_1_agents_2_mean_no_assim_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_1.csv')
    runs_1_agents_4_mean_no_assim_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_1.csv')
    runs_1_agents_6_mean_no_assim_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_1.csv')
    runs_1_agents_8_mean_no_assim_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_1.csv')
    runs_1_agents_10_mean_no_assim_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_0_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_none_state_index_1.csv')

    runs_1_agents_2_mean_no_covs_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_2_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1_no_covs.csv')
    runs_1_agents_4_mean_no_covs_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_4_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1_no_covs.csv')
    runs_1_agents_6_mean_no_covs_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_6_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1_no_covs.csv')
    runs_1_agents_8_mean_no_covs_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_8_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1_no_covs.csv')
    runs_1_agents_10_mean_no_covs_vel =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_mean_state_index_1_no_covs.csv')



    # runs_1_agents_10_mean =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_mean.csv')
    # runs_1_agents_10_min_P =pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\mean_state_assim_1_agents_10_num_of_mc_1_angle_1_distance_1_assim_type_min_P.csv')
    article_MSE_vel = [
                 alg_utils.calc_MSE(runs_1_agents_2_article_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_article_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_article_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_article_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_article_vel , X[1,:-1])]
    mean_MSE_no_covs_vel =[
                 alg_utils.calc_MSE(runs_1_agents_2_mean_no_covs_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_mean_no_covs_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_mean_no_covs_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_mean_no_covs_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_mean_no_covs_vel , X[1,:-1])]
    mean_MSE_vel = [
             alg_utils.calc_MSE(runs_1_agents_2_mean_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_4_mean_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_6_mean_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_8_mean_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_10_mean_vel , X[1,:-1])]
    mean_MSE_no_assim_vel = [
             alg_utils.calc_MSE(runs_1_agents_2_mean_no_assim_vel , X[1,:]) ,
             alg_utils.calc_MSE(runs_1_agents_4_mean_no_assim_vel , X[1,:]) ,
             alg_utils.calc_MSE(runs_1_agents_6_mean_no_assim_vel , X[1,:]) ,
             alg_utils.calc_MSE(runs_1_agents_8_mean_no_assim_vel , X[1,:]) ,
             alg_utils.calc_MSE(runs_1_agents_10_mean_no_assim_vel , X[1,:])]

    min_P_MSE_vel = [
             alg_utils.calc_MSE(runs_1_agents_2_min_P_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_4_min_P_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_6_min_P_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_8_min_P_vel , X[1,:-1]) ,
             alg_utils.calc_MSE(runs_1_agents_10_min_P_vel , X[1,:-1])]


    article_MSE =[alg_utils.calc_MSE(runs_1_agents_2_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_3_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_5_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_7_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_9_article , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_article , X[0,:-1])]

    mean_MSE_no_covs = [alg_utils.calc_MSE(runs_1_agents_2_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_3_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_5_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_7_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_9_mean_no_covs , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_mean_no_covs , X[0,:-1])]

    MSE_no_angle_pos = [
                 alg_utils.calc_MSE(no_angle_dist_05_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_06_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_07_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_08_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_09_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_1_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_2_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_3_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_4_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_5_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_10_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_15_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_20_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_25_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_30_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_35_pos , X[0,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_40_pos , X[0,:-1]) ,
                ]
    MSE_no_angle_vel = [
                 alg_utils.calc_MSE(no_angle_dist_05_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_06_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_07_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_08_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_09_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_1_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_2_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_3_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_4_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_5_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_10_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_15_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_20_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_25_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_30_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_35_vel , X[1,:-1]) ,
                 alg_utils.calc_MSE(no_angle_dist_40_vel , X[1,:-1]) ,
                ]

    mean_MSE = [alg_utils.calc_MSE(runs_1_agents_2_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_3_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_5_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_7_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_9_mean , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_mean , X[0,:-1])]


    mean_no_assim_MSE = [alg_utils.calc_MSE(runs_1_agents_2_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_3_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_5_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_7_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_9_mean_no_assim , X[0,:]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_mean_no_assim , X[0,:])]

    min_P_MSE = [alg_utils.calc_MSE(runs_1_agents_2_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_3_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_4_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_5_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_6_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_7_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_8_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_9_min_P , X[0,:-1]) ,
                 alg_utils.calc_MSE(runs_1_agents_10_min_P , X[0,:-1])]



    axs[0].plot([2,3,4,5,6,7,8,9,10],article_MSE,'b')
    axs[0].plot([2,3,4,5,6,7,8,9,10],mean_MSE,'r')
    axs[0].plot([2,3,4,5,6,7,8,9,10],min_P_MSE,'k')
    # axs[0].plot([2,3,4,5,6,7,8,9,10],mean_no_assim_MSE,'g')
    axs[0].plot([2,3,4,5,6,7,8,9,10],mean_MSE_no_covs,'g')

    axs[1].plot([2,4,6,8,10],article_MSE_vel,'b')
    axs[1].plot([2,4,6,8,10],mean_MSE_vel,'r')
    axs[1].plot([2,4,6,8,10],min_P_MSE_vel,'k')
    axs[1].plot([2,4,6,8,10],mean_MSE_no_covs_vel,'g')
    # axs[1].plot([2,4,6,8,10],mean_MSE_no_assim_vel,'g')

    axs[0].grid()
    axs[0].legend(['Sensors assimilation', 'Sensors mean with covs','Sensors min covariance','Sensors mean no covs'])
    axs[0].set_title('MSE (Position)')
    axs[0].set_xlabel('Sensors count')
    # axs[0].set_ylabel('Amplitude')

    axs[1].grid()
    axs[1].legend(['Sensors assimilation', 'Sensors mean with covs','Sensors min covariance' , 'Sensors mean no covs'])
    axs[1].set_title('MSE (Velocity)')
    axs[1].set_xlabel('Sensors count')
    # axs[1].set_ylabel('Amplitude')

    axs1[0].plot([0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,10,15,20,25,30,35,40],MSE_no_angle_pos,'b')
    axs1[1].plot([0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,10,15,20,25,30,35,40],MSE_no_angle_vel,'b')

    axs1[0].grid()
    axs1[0].legend(['Sensors assimilation', 'Sensors mean with covs','Sensors min covariance','Sensors mean no covs'])
    axs1[0].set_title('MSE (Position)')
    axs1[0].set_xlabel('Distance from')
    # axs[0].set_ylabel('Amplitude')

    axs1[1].grid()
    axs1[1].legend(['Sensors assimilation', 'Sensors mean with covs','Sensors min covariance' , 'Sensors mean no covs'])
    axs1[1].set_title('MSE (Velocity)')
    axs1[1].set_xlabel('Sensors count')
    # axs[1].set_ylabel('Amplitude')

def two_STD_graphs(T):
    fig,ax = plt.subplots(2,2)
    five_sensors_P_vel = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\P_five_sensors_vel_assim.csv')
    five_sensors_STD_vel = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\STD_OF_MC_five_sensors_vel_assim.csv')
    five_sensors_P_pos = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\P_five_sensors_pos_assim.csv')
    five_sensors_STD_pos = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\STD_OF_MC_five_sensors_pos_assim.csv')
    one_sensors_P_pos = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\P_one_sensors_pos.csv')
    one_sensors_STD_pos = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\STD_OF_MC_one_sensors_pos.csv')
    one_sensors_P_vel = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\P_one_sensors_vel.csv')
    one_sensors_STD_vel = pd.read_csv(r'C:\Users\gilim\Desktop\kalman_filtering-projectDevelopment\figures\distance_R_01_RF_005_Q10_pos01 - another one\MSE\STD_OF_MC_one_sensors_vel.csv')

    ax[0, 0].plot(T, one_sensors_P_pos['0'])
    ax[0, 0].plot(T, one_sensors_STD_pos['0'])

    ax[1, 0].plot(T, one_sensors_P_vel['0'])
    ax[1, 0].plot(T, one_sensors_STD_vel['0'])

    ax[0,1].plot(T[:-1] , five_sensors_P_pos['0'])
    ax[0,1].plot(T[:-1] , five_sensors_STD_pos['0'])

    ax[1,1].plot(T[:-1] , five_sensors_P_vel['0'])
    ax[1,1].plot(T[:-1] , five_sensors_STD_vel['0'])

    ax[0,0].set_xlabel('Time [sec]')
    ax[0,1].set_xlabel('Time [sec]')
    ax[1,0].set_xlabel('Time [sec]')
    ax[1,1].set_xlabel('Time [sec]')

    ax[0,0].set_ylabel('Position [M]')
    ax[0,1].set_ylabel('Position [M]')
    ax[1,0].set_ylabel('Velocity [M/s]')
    ax[1,1].set_ylabel('Velocity [M/s]')


    ax[0,0].legend(['Estimator STD','MC STD'])
    ax[0,1].legend(['Estimator STD','MC STD'])
    ax[1,0].legend(['Estimator STD','MC STD'])
    ax[1,1].legend(['Estimator STD','MC STD'])


    ax[0,0].grid()
    ax[0,1].grid()
    ax[1,0].grid()
    ax[1,1].grid()

    ax[0,0].set_title('One Sensor STD of MC - Position')
    ax[0,1].set_title('Five assimilated Sensors STD of MC - Position')
    ax[1,0].set_title('One Sensor STD of MC - Velocity')
    ax[1,1].set_title('Five assimilated Sensors STD of MC - Velocity')



