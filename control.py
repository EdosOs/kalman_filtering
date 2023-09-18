from numpy import sin , array
def input(inp_type , t , amplitude , pulse_time = 2 , dt = 2 , delay = 0):
    if inp_type == 'sin':
        return array([amplitude * sin(t)])
    elif inp_type == 'step':
        return array([amplitude * (t+1e-12)/(t+1e-12)])
    elif inp_type == 'ramp':
        return array([amplitude * t])
    elif inp_type == 'pulse':
        return amplitude * 1 if (t < pulse_time+dt and t> pulse_time) else 0
    else:
        return 0