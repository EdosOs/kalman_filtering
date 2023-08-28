import numpy as np

def input(inp_type , t , amplitude):
    if inp_type == 'sin':
        return np.array([amplitude * np.sin(t)])
    elif inp_type == 'step':
        return np.array([amplitude * 1])
    elif inp_type == 'ramp':
        return np.array([amplitude * t])
    else:
        return 0