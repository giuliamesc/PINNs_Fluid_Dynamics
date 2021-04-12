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
#  - mu * (u_xx + u_yy) = -p_x      in \Omega = (0, L) x (0, 2*delta)
#  u   = 0                          in (0, L) x {0, 2*delta}
#  u_x = 0                          in {0, L} x (0, 2*delta)
#
# u_exact(x,y) = - p_x * y * (2 - y / delta) * delta / (2*mu)
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
num_PDE  = 1000
num_BC   = 80 # points for each edge
num_test = 1000

# %% Forcing and Solutions
dim = 2
p_x = (P_end - P_str) / L

forcing_x = lambda x: -p_x + 0*x[:,1]

u_exact   = lambda x: -p_x * x[:,1] * (2 - x[:,1] / delta) * delta / (2*mu)
 

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


u_test = u_exact(x_test)

f_1 = forcing_x(x_PDE)

# %% Losses creation

def PDE_U():
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x_PDE)
        u = model(x_PDE)
        u_y = nse.physics.tens_style.gradient_scalar(tape, u, x_PDE)[:,1]
        u_yy = nse.physics.tens_style.gradient_scalar(tape, u, x_PDE)[:,1]
    return - mu * (u_yy) - f_1

def BC_N():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_N)
        u = model(x_BC_N)
        grad_u = nse.physics.tens_style.gradient_scalar(tape, u, x_BC_N)
        u_x = grad_u[:,0]
    return u_x 

def test_loss():
    u = model(x_test)
    return (u - u_test) * (u - u_test)


# %% Losses definition
losses = [ns.LossMeanSquares('PDE_U', PDE_U, weight = 2.0),
          ns.LossMeanSquares( 'BC_D', lambda: model(x_BC_D)),
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


plt.show(block = True)
