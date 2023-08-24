from scipy.integrate import solve_ivp
import numpy as np



vdp1 = lambda T,Y: [Y[1], -Y[1]*T - Y[0]]
sol = solve_ivp (vdp1, t_span , [1, 1] ,t_eval=np.linspace(t_span[0] , t_span[1] , num_of_points))
T = sol.t
Y = sol.y