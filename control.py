from numpy import sin , array
def input(inp_type , t , amplitude , impulse_time = 0 , dt = 1e-2):
    if inp_type == 'sin':
        return array([amplitude * sin(t)])
    elif inp_type == 'step':
        return array([amplitude * 1])
    elif inp_type == 'ramp':
        return array([amplitude * t])
    elif inp_type == 'dirac':
        return amplitude * 1 if (t < impulse_time+dt and t> impulse_time) else 0
    else:
        return 0