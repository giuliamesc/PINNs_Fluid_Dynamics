#!/usr/bin/env python3

#%%
import sys
import os
cwd = os.path.abspath(os.getcwd())
sys.path.append(os.path.join(cwd,"nisaba"))
import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import numpy as np

#############################################################################
#  rho * (u * u_x + v * u_y) - mu * (u_xx + u_yy) = -p_x      in \Omega = (0, L) x (0, 2*delta)
#  rho * (u * v_x + v * v_y) - mu * (v_xx + v_yy) = -p_y      in \Omega = (0, L) x (0, 2*delta)
#  u   = v   = 0                                              in (0, L) x {0, 2*delta}
#  u_x = v_x = 0                                              in {0, L} x (0, 2*delta)
#
# u_exact(x,y) = - p_x * y * (2 - y / delta) * delta / (2*mu)
# v_exact(x,y) = 0 
#############################################################################

# %% Options
# Domain Setup
L      = 1
delta  = 0.05
H = 2 * delta
# Experiment setup
P_str = 1000000
P_end = 0
# Fluid Setup (lava)
rho = 3100
mu  = 890

# Numerical options
num_PDE  = 200
num_BC   = 20 # points for each edge
num_test = 1000

# %% Forcing and Solutions
dim = 2
p_x = (P_end - P_str) / L
p_y = 0
forcing_x = lambda x: -p_x + 0*x[:,0]
forcing_y = lambda x: -p_y + 0*x[:,0] 

u_exact   = lambda x: -p_x * x[:,1] * (2 - x[:,1] / delta) * delta / (2*mu)
v_exact   = lambda x: 0*x[:,0]
 

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

x_PDE   = tf.random.uniform(shape = [num_PDE, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,  2], minval = [0, 0],  maxval = [0, H], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,  2], minval = [L, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,  2], minval = [0, 0],  maxval = [L, 0], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,  2], minval = [0, H],  maxval = [L, H], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test,2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_D = tf.concat([x_BC_y0, x_BC_y1], axis = 0)
x_BC_N = tf.concat([x_BC_x0, x_BC_x1], axis = 0)


u_test = u_exact(x_test)[:, None]
v_test = v_exact(x_test)[:, None]
f_1 = forcing_x(x_PDE)
f_2 = forcing_y(x_PDE)

# %% Losses creation

# def PDE_U():
#     return model(x_PDE)

def PDE_U():
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x_PDE)
        u_vect = model(x_PDE)
        u = u_vect[:,0]
        v = u_vect[:,1]
        grad_u = nse.physics.tens_style.gradient_scalar(tape, u, x_PDE)
        dux = grad_u[:,0]
        duy = grad_u[:,1]
        lapl_u = nse.physics.tens_style.laplacian_scalar(tape, u, x_PDE, dim)
    return rho * (u * dux + v * duy) - mu * (lapl_u) - f_1


def PDE_V():
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x_PDE)
        u_vect = model(x_PDE)
        u = u_vect[:,0]
        v = u_vect[:,1]
        grad_v = nse.physics.tens_style.gradient_scalar(tape, v, x_PDE)
        dvx = grad_v[:,0]
        dvy = grad_v[:,1]
        lapl_v = nse.physics.tens_style.laplacian_scalar(tape, v, x_PDE, dim)
    return rho * (u * dvx + v * dvy) - mu * (lapl_v) - f_2

def BC_D():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_D)
        u_vect = model(x_BC_D)
    return u_vect[:,0] + u_vect[:,1]

def BC_N():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_N)
        u_vect = model(x_BC_N)
        grad_u_vect = nse.physics.tens_style.gradient_vector(tape, u_vect, x_BC_N, dim)
        u_x = grad_u_vect[:,0,0]
        v_x = grad_u_vect[:,1,0]
    return u_x + v_x

def test_loss():
    u_vect = model(x_test)
    u = u_vect[:,0]
    v = u_vect[:,1]
    return (u - u_test) * (u - u_test) + (v - v_test) * (v - v_test)

# %% Losses definition
losses = [ns.LossMeanSquares('PDE_U', PDE_U, weight = 2.0),
          ns.LossMeanSquares('PDE_V', PDE_V, weight = 2.0),
          ns.LossMeanSquares( 'BC_D',  BC_D),
          ns.LossMeanSquares( 'BC_N',  BC_N)]

loss_test = ns.LossMeanSquares('fit', test_loss)

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 500)

# %% Post-processing
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,0].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('velocity u')
ax = fig.add_subplot(122, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], v_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,1].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('velocity v')

plt.show(block = True)