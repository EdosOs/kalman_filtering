from scipy.integrate import solve_ivp
from numpy import  squeeze , linspace
from numpy.random import randn
from control import input



#Dynamic model
#enter equations from order 1 as parameters
def acceleration_model(t_start,t_stop,initial_cond,model_noise_var , dt , B , G , input_type = 'sin',input_amplitude = 1):
    t_span = [t_start, t_stop]
    ode_fcn = lambda T,X: [X[1]+X[2]*T , X[2] , 0] + squeeze(B , axis=0) * input(input_type , T , input_amplitude) + squeeze(G , axis=0) * randn() * model_noise_var
    sol = solve_ivp(ode_fcn, t_span=t_span  ,y0=initial_cond ,t_eval=linspace(t_start , t_stop , int((t_stop-t_start) / dt)) )
    return  sol.t ,  sol.y


def velocity_model(t_start,t_stop,model_noise_var , dt , B , G, input_type = 'sin' , input_amplitude = 1):
    t_span = [t_start, t_stop]
    ode_fcn = lambda T,X: [X[1],0] + squeeze(B , axis=1) * input(input_type , T , 5) + squeeze(G , axis=1) * randn() * model_noise_var
    sol = solve_ivp (ode_fcn, t_span=t_span  ,y0=[0, 0, 1] ,t_eval=linspace(t_start , t_stop , int((t_stop-t_start) / dt)) )
    return  [sol.t ,  sol.y]