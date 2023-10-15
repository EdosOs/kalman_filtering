import numpy as np
def acceleration_model_noise(var , dt):
    process_mat = np.array([[dt**4/4 , dt**3/2 , dt**2/2], [dt**3/2  , dt**2 , dt] , [dt**2 , dt , 1]])
    return var *  np.block ([
                    [process_mat , np.zeros([3,3]) , np.zeros([3,3])],
                    [np.zeros([3,3])  , process_mat , np.zeros([3,3])],
                    [np.zeros([3,3])  , process_mat , np.zeros([3,3])]
    ])


def velocity_model_noise(var , dt):
    process_mat = np.array([[dt,1, dt,1]])
    return process_mat.T * var @ process_mat

def velocity_model_noise_3x3(var , dt):
    process_mat = np.array([[dt**2/2 , dt , 1]])
    return process_mat.T * var @ process_mat

def velocity_model_noise_2x2(var , dt):
    process_mat = np.array([[dt , 1]])
    return process_mat.T * var @ process_mat
'''Note that if the system is known (for example an asteroid floating in space , the process might just a known factor
 (0 in the asteroid example) this is a generic way to model unknown noise for certain order of physical system.'''