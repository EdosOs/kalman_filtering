import sympy
from numpy import eye
from sympy import (init_printing, Matrix, MatMul,
                   integrate, symbols)
from IPython.display import display
sympy.init_printing(use_latex='mathjax')

init_printing(use_latex='mathjax')
dt, phi = symbols('dt Phi')
A_dt_order_1 = Matrix([[0, dt],
              [0,  0]])
A_dt_order_2 = Matrix([[0, dt, dt**2/2],
              [0,  0,      dt],
              [0,  0,       0]])
F_k = Matrix([[1, dt, dt**2/2],
              [0,  1,      dt],
              [0,  0,       1]])
Q_c = Matrix([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 1]])*phi

Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))

# factor phi out of the matrix to make it more readable
Q = Q / phi
display(MatMul(Q, phi))

print('first order F: ')
display(eye(2) + A_dt_order_1 + A_dt_order_1@A_dt_order_1/2)

print('second order F: ')
display(eye(3) + A_dt_order_2 + A_dt_order_2@A_dt_order_2/2)