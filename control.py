from numpy import sin , array
def input(inp_type , t , amplitude):
    if inp_type == 'sin':
        return array([amplitude * sin(t)])
    elif inp_type == 'step':
        return array([amplitude * 1])
    elif inp_type == 'ramp':
        return array([amplitude * t])
    else:
        return 0