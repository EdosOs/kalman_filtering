from numpy import sin,cos , array, linspace
from numpy.random import randn
def input(inp_type , t , amplitude , pulse_time = 0 , dt = 2 , delay = 0,t_transition = [-1,0]):
    amp = [0]
    for index in range(len(t_transition)-1) if len(t_transition)-1>0 else range(1):
        if t>=t_transition[-1]:
            amp[0] = amplitude[-1]
        elif t<=t_transition[0]:
            amp[0] = amplitude[0]
        elif (t<=t_transition[index+1] and t_transition[index]<=t):
            amp[0] = amplitude[index]
    if inp_type == 'sin':
        return array([amp[0] * sin(t)])
    elif inp_type == 'cos':
        return array([amp[0] * cos(t)])
    elif inp_type == 'step':
        return array([amp[0] * (t+1e-12)/(t+1e-12)])
    elif inp_type == 'ramp':
        return array([amp[0] * t])
    elif inp_type == 'pulse':
        return amp[0] * 1 if (t <= pulse_time+dt and t >= pulse_time) else 0
    elif inp_type == 'variable_step':
        return array([amp[0] * (t+1e-12)/(t+1e-12)* randn(*t)])
    else:
        return 0

def constant_input(inp_type , t , amplitude , pulse_time = 0 , dt = 2 , delay = 0):
    if inp_type == 'sin':
        return array([amplitude * sin(t)])
    elif inp_type == 'cos':
        return array([amplitude * cos(t)])
    elif inp_type == 'step':
        return array([amplitude * (t+1e-12)/(t+1e-12)])
    elif inp_type == 'ramp':
        return array([amplitude * t])
    elif inp_type == 'pulse':
        return amplitude * 1 if (t < pulse_time+dt and t> pulse_time) else 0
    else:
        return 0

def circle_input(t_start,t_stop , amplitude , pulse_time = 0 , dt = 2 ):
    steps = (t_stop - t_start)/dt
    return array([0 , sin(linspace(t_start,t_stop,int(steps))) , 0 , cos(linspace(t_start,t_stop,int(steps)))])
