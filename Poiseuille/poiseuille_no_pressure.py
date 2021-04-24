#!/usr/bin/env python3

#%%
import os
cwd = os.path.abspath(os.getcwd())
os.chdir("../")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
import tensorflow as tf
import numpy as np

#############################################################################
#  (u * u_x + v * u_y) - (u_xx + u_yy) / Re = -p_x      in \Omega = (0, 1) x (0, 2*delta)
#  (u * v_x + v * v_y) - (v_xx + v_yy) / Re = -p_y      in \Omega = (0, 1) x (0, 2*delta)
#  u   = v   = 0                                              in (0, 1) x {0, 2*delta}
#  u_x = v_x = 0                                              in {0, 1} x (0, 2*delta)
#
# u_exact(x,y) = - Re * p_x * y * (2 - y / delta) * delta / 2
# v_exact(x,y) = 0 
#############################################################################

# %% Options
# Fluid and Flow Setup
dim = 2
rho   = 3100  # lava density
mu    = 890   # lava viscosity
Ub    = 1     # Bulk velocity
L_dim = 1     # length of the pipe
H_dim = 0.1   # heigth of the pipe
P_str = 1e6 
P_end = 0

# Adimensionalization
Re = rho * Ub * L_dim / mu
L = 1
H = H_dim / L_dim
delta = H / 2
p_str = P_str / (rho * Ub^2)
p_end = P_end / (rho * Ub^2)

# %% Forcing and Solutions
p_x = p_end - p_str
p_y = 0

forcing_x = lambda x: -p_x + 0*x[:,0]
forcing_y = lambda x: -p_y + 0*x[:,0] 

u_exact   = lambda x: - Re * p_x * x[:,1] * (2 - x[:,1] / delta) * delta / 2
v_exact   = lambda x: 0*x[:,0]
 
# %% Numerical options
num_PDE  = 200
num_BC   = 20 # points for each edge
num_hint = 50
num_test = 1000

# %% Inizialization

#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(2,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(2)
])

x_PDE   = tf.random.uniform(shape = [num_PDE,  2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_hint  = tf.random.uniform(shape = [num_hint, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,   2], minval = [0, 0],  maxval = [0, H], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   2], minval = [L, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   2], minval = [0, 0],  maxval = [L, 0], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   2], minval = [0, H],  maxval = [L, H], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())

# %% Losses creation

def create_rhs(x, force):
    if force is None:
        samples = x.shape[0]
        return tf.zeros(shape = [samples,1], dtype = ns.config.get_dtype())
    return force(x)

# k is the coordinate of the vectorial equation
# j is the direction in which the derivative is computed

def PDE_MOM(x, k, force):
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        
        u_vect = model(x)
        u = u_vect[:,0]
        v = u_vect[:,1]
        u_eq = u_vect[:,k]
        
        grad_eq = operator.gradient_scalar(tape, u_eq, x)
        deqx = grad_eq[:,0]
        deqy = grad_eq[:,1]
        lapl_eq = operator.laplacian_scalar(tape, u_eq, x, dim)
        
        rhs = create_rhs(x, force)
    return (u * deqx + v * deqy) - (lapl_eq) / Re - rhs

def BC_D(x, k, g_bc = None):
    uk = model(x)[:,k]
    rhs = create_rhs(x, g_bc)
    return uk - rhs

def BC_N(x, k, j, g_bc = None): #j is the direction in which we want the derivative
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x)
        uk = model(x)[:,k]
        uk_j = operator.gradient_scalar(tape, uk, x)[:,j]
<<<<<<< HEAD:Poiseuille/poiseuille_no_pressure.py
        rhs = create_rhs(x, g_bc)
        return uk_j - rhs

def exact_value(x, k, sol = None):
    uk = model(x)[:,k]
    rhs = create_rhs(x, sol)
    return uk - rhs

# %% Training Losses definition
PDE_losses = [ns.LossMeanSquares('PDE_MOMU', lambda: PDE_MOM(x_PDE, 0, forcing_x), normalization = 1e4, weight = 1.0),
              ns.LossMeanSquares('PDE_MOMV', lambda: PDE_MOM(x_PDE, 1, forcing_y), normalization = 1e4, weight = 1.0)]
BCD_losses = [ns.LossMeanSquares('BCD_x0_u', lambda: BC_D(x_BC_x0,0, u_exact), weight = 1.0),
              ns.LossMeanSquares('BCD_x0_v', lambda: BC_D(x_BC_x0,1), weight = 1.0),
              ns.LossMeanSquares('BCD_y0_u', lambda: BC_D(x_BC_y0,0), weight = 1.0),
              ns.LossMeanSquares('BCD_y0_v', lambda: BC_D(x_BC_y0,1), weight = 1.0),
              ns.LossMeanSquares('BCD_y1_u', lambda: BC_D(x_BC_y1,0), weight = 1.0),
              ns.LossMeanSquares('BCD_y1_v', lambda: BC_D(x_BC_y1,1), weight = 1.0)]
BCN_losses = [ns.LossMeanSquares('BCN_x1_u', lambda: BC_N(x_BC_x1,0,0), weight = 1.0),
              ns.LossMeanSquares('BCN_x1_v', lambda: BC_N(x_BC_x1,1,0), weight = 1.0)]
EXC_Losses = [ns.LossMeanSquares( 'exact_u', lambda: exact_value(x_hint, 0, u_exact), weight = 1.0),
              ns.LossMeanSquares( 'exact_v', lambda: exact_value(x_hint, 1, v_exact), weight = 1.0)]

losses = PDE_losses + BCD_losses + BCN_losses + EXC_Losses

# %% Test Losses definition
loss_test = [ns.LossMeanSquares('u_fit', lambda: exact_value(x_test, 0, u_exact)),
             ns.LossMeanSquares('v_fit', lambda: exact_value(x_test, 1, v_exact))]
=======
        return tf.math.abs(uk_j - g_bc)

def Hints():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_hint)
        u_vect = model(x_hint)
        u = u_vect[:,0]
        v = u_vect[:,1] 
    return (u - u_hint) * (u - u_hint) + (v - v_hint) * (v - v_hint)

def test_loss():
    u_vect = model(x_test)
    u = u_vect[:,0]
    v = u_vect[:,1]
    return (u - u_test) * (u - u_test) + (v - v_test) * (v - v_test)

# %% Losses definition
losses = [ns.LossMeanSquares(' PDE_U', lambda: PDE(x_PDE,0,f_1), weight = 1.0),
          ns.LossMeanSquares(' PDE_V', lambda: PDE(x_PDE,1,f_2), weight = 1.0),
          ns.LossMeanSquares('BCN_x0_x', lambda: BC_N(x_BC_x0,0,0), weight = 5.0),
          ns.LossMeanSquares('BCD_x0_y', lambda: BC_D(x_BC_x0,1), weight = 5.0),
          ns.LossMeanSquares('BCD_y0', lambda: BC_D(x_BC_y0,0) + BC_D(x_BC_y0,1), 
                             weight = 10.0),
          ns.LossMeanSquares('BCD_y1', lambda: BC_D(x_BC_y1,0) + BC_D(x_BC_y1,1), 
                             weight = 10.0),
          ns.LossMeanSquares( 'BC_N',  lambda: BC_N(x_BC_x1,0,0) 
                             + BC_N(x_BC_x1,1,0), weight = 10.0),
          #ns.LossMeanSquares('Hints', lambda: Hints(), weight = 15.0)
          ]
loss_test = ns.LossMeanSquares('fit', test_loss, normalization = num_test)
>>>>>>> 37adad19eb165efbb1e83a80305b1f487a61fa31:Poiseuille/poiseuille_adimensional.py

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 250)

# %% Saving Loss History

problem_name = "Poiseuille_Adimensional"
history_file = os.path.join(cwd, "{}_history_loss.json".format(problem_name))
pb.save_history(history_file)
ns.utils.plot_history(history_file)
history = ns.utils.load_json(history_file)

# %% Post-processing
u_test = u_exact(x_test)
v_test = v_exact(x_test)

import matplotlib.pyplot as plt
fig_1 = plt.figure(1)
ax_1 = fig_1.add_subplot(projection='3d')
ax_1.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax_1.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,0].numpy(), label = 'numerical solution')
ax_1.legend()
ax_1.set_xlabel('x')
ax_1.set_ylabel('y')
ax_1.set_zlabel('velocity u')

fig_2 = plt.figure(2)
ax_2 = fig_2.add_subplot(projection='3d')
ax_2.scatter(x_test[:,0], x_test[:,1], v_test, label = 'exact solution')
ax_2.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,1].numpy(), label = 'numerical solution')
ax_2.legend()
ax_2.set_xlabel('x')
ax_2.set_ylabel('y')
ax_2.set_zlabel('velocity v')

plt.show(block = True)
print("Reynolds Number -> {}".format(Re))