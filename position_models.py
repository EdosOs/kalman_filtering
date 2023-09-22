
from numpy import linspace , sin , cos,pi
def circle(t_start,t_stop, dt,r=1 , inital_position = [0,0]):
    steps = (t_stop - t_start)/dt + 1
    X,Y = [[]],[[]]
    X[0] = r*sin(linspace(t_start,t_stop,int(steps)))
    Y[0] = r*cos(linspace(t_start,t_stop,int(steps)))
    X[0] += inital_position[0]
    Y[0] += inital_position[1]
    return X,Y

def single_circle(t_start,t_stop, dt,r = 1,inital_position = [0,0]):
    steps =(t_stop - t_start)/dt
    X,Y = [[]],[[]]
    X[0] = r*sin(linspace(0,2*pi,int(steps)))
    Y[0] = r*cos(linspace(0,2*pi,int(steps)))
    X[0] += inital_position[0]
    Y[0] += inital_position[1]
    return X,Y