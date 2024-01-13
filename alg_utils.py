import numpy as np
import pandas as pd

def check_matrix_arr_validity(matrix_arr):
    if len(matrix_arr) == 0 :print('matrix array length must be at least 1')

    try:p = matrix_arr[0].shape
    except:print("invalid input")
    rows = matrix_arr[0].shape[0]
    try:
        cols = matrix_arr[0].shape[1]
    except: cols = 1
    return rows,cols

def matrix_arr_mean(matrix_arr):
    rows , cols = check_matrix_arr_validity(matrix_arr)
    if len(matrix_arr) == 1 : return matrix_arr
    mean_matrix = np.zeros_like(matrix_arr[0])
    for row in range(rows):
        for col in range(cols):
            mean_matrix[row,col] = np.mean(matrix_arr[:, row, col])
    return mean_matrix

def matrix_arr_min_indexes(matrix_arr):
    rows , cols = check_matrix_arr_validity(matrix_arr)
    if len(matrix_arr) == 1 : return matrix_arr
    reshaped_matrix_arr = np.array([np.reshape(matrix ,(rows*cols)) for matrix in matrix_arr])
    matrix_df = pd.DataFrame(reshaped_matrix_arr , index=None)
    min_idx = matrix_df.idxmin(axis=0)
    min_mat = [matrix_df.iloc[min_idx[x],x] for x in range(len(min_idx))] # check

    return np.reshape(min_mat,(int((len(min_mat))**0.5),int((len(min_mat))**0.5))) , np.array(min_idx)

def select_by_index_dfs(df , index_df):
    df = np.reshape(df , [len(df) , len(df[0])])
    return np.array([df[index_df[x] , x] for x in range(len(index_df))])
def select_min_P_position_sensor_scenario(agents_arr):
    #state , cov df
    agents_cov_arr = []
    agents_state_arr = []
    for agent in agents_arr:
        agents_cov_arr.append(agent.filter.updated_covs[:, 0, 0])
        agents_state_arr.append(agent.filter.updated_state[:, 0])
    # get min cov agent
    covs_df = pd.DataFrame(np.array(agents_cov_arr) , index = None)
    state_df = pd.DataFrame(np.array(agents_state_arr) , index = None)
    min_agent_per_step = covs_df.idxmin(axis=0)
    min_P_state =[state_df.iloc[min_agent_per_step[x],x] for x in range(len(agent.filter.updated_covs))]
    return min_P_state

def calc_MSE(data1 , data2):
    return sum(((data1['0'] - data2)**2)) / data1.shape[0]

def calc_RMSE(data1, data2):
    return sum(((data1['0'] - data2[0, :-1])**2)**0.5) / data1.shape[0]
