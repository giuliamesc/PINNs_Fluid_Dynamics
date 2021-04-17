#!/usr/bin/env python3

#%%
import sys
import os
cwd = os.path.abspath(os.getcwd())
sys.path.append(os.path.join(cwd,"nisaba"))
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
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
num_hint = 50
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

x_PDE   = tf.random.uniform(shape = [num_PDE,  2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_hint  = tf.random.uniform(shape = [num_hint, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,   2], minval = [0, 0],  maxval = [0, H], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   2], minval = [L, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   2], minval = [0, 0],  maxval = [L, 0], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   2], minval = [0, H],  maxval = [L, H], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())


u_test = u_exact(x_test)[:, None]
v_test = v_exact(x_test)[:, None]
u_hint = u_exact(x_hint)[:, None]
v_hint = v_exact(x_hint)[:, None]
inlet = u_exact(x_BC_x0)[:, None]
f_1 = forcing_x(x_PDE)
f_2 = forcing_y(x_PDE)

# %% Losses creation

def PDE(x, k, force):  # k is the coordinate of the vectorial equation
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
    return tf.math.abs(rho * (u * deqx + v * deqy) - mu * (lapl_eq) - force)

def BC_D(x, k, g_bc = None):
    with ns.GradientTape(persistent = True) as tape:
        if g_bc is None:
            samples = x.shape[0]
            g_bc = tf.zeros(shape = [samples,1], dtype = ns.config.get_dtype())
        tape.watch(x)
        u = model(x)[:,k]
        return tf.math.abs(u - g_bc)
    

def BC_N():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_x1)
        u_vect = model(x_BC_x1)
        grad_u_vect = operator.gradient_vector(tape, u_vect, x_BC_x1, dim)
        u_x = grad_u_vect[:,0,0]
        v_x = grad_u_vect[:,1,0]
    return tf.math.abs(u_x) + tf.math.abs(v_x)

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
losses = [ns.LossMeanSquares(' PDE_U', lambda: PDE(x_PDE, 0, f_1), weight = 2.0, normalization = num_PDE),
          ns.LossMeanSquares(' PDE_V', lambda: PDE(x_PDE, 1, f_2), weight = 2.0, normalization = num_PDE),
          ns.LossMeanSquares('BCD_x0', lambda: BC_D(x_BC_x0, 0, inlet) + BC_D(x_BC_x0, 1) , weight = 10.0, normalization = num_BC),
          ns.LossMeanSquares('BCD_y0', lambda: BC_D(x_BC_y0, 0) + BC_D(x_BC_y0, 1) , weight = 10.0, normalization = num_BC),
          ns.LossMeanSquares('BCD_y1', lambda: BC_D(x_BC_y1, 0) + BC_D(x_BC_y1, 1) , weight = 10.0, normalization = num_BC),
          ns.LossMeanSquares( 'BC_N',  BC_N, weight = 10.0, normalization = num_BC),
          ns.LossMeanSquares('Hints', Hints, weight = 15.0, normalization = num_hint)
          ]
loss_test = ns.LossMeanSquares('fit', test_loss, normalization = num_test)

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 500)


# %% Saving Loss History

problem_name = "Poiseuille"
history_file = os.path.join(cwd, "{}_history_loss.json".format(problem_name))
pb.save_history(history_file)
ns.utils.plot_history(history_file)
history = ns.utils.load_json(history_file)
# %% Post-processing
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
#ax.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,0].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('velocity u')
ax = fig.add_subplot(122, projection='3d')
#ax.scatter(x_test[:,0], x_test[:,1], v_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,1].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('velocity v')

plt.show(block = True)