import numpy as np
def acceleration_model_noise(noise_vec , var , dt):
    process_mat = np.array([[dt**4/4 , dt**3/2 , dt**2/2], [dt**3/2  , dt**2 , dt] , [dt**2 , dt , 1]])
    return np.block ([
                    [process_mat , np.zeros(3) , np.zeros(3)],
                    [np.zeros(3)  , process_mat , np.zeros(3)],
                    [np.zeros(3)  , process_mat , np.zeros(3)]
    ])

def velocity_model_noise(noise_vec , var , dt):
    process_mat = np.array([[dt**4/4 , dt**3/2], [dt**3/2  , dt**2]])
    return np.block ([
                    [process_mat , np.zeros(2)],
                    [np.zeros(2)  , process_mat]
                    ])

'''Note that if the system is known (for example an asteroid floating in space , the process might just a known factor
 (0 in the asteroid example) this is a generic way to model unknown noise for certain order of physical system.'''