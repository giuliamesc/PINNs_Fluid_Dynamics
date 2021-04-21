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
#  (u * u_x + v * u_y) - (u_xx + u_yy) / Re + p_x = 0      in \Omega = (0, 1) x (0, 2*delta)
#  (u * v_x + v * v_y) - (v_xx + v_yy) / Re + p_y = 0      in \Omega = (0, 1) x (0, 2*delta)
#  u   = v   = 0                                              in (0, 1) x {0, 2*delta}
#  u_x = v_x = 0                                              in {0, 1} x (0, 2*delta)
#  p = p_str                                                  in {0} x (0, 2*delta)
#  p = p_end                                                  in {1} x (0, 2*delta)
#  p_y = 0                                                    in (0, 1) x {0, 2*delta}
#
#  p_exact(x,y) = (p_end-p_str)/L * x + p_str
#  u_exact(x,y) = - Re * p_x * y * (2 - y / delta) * delta / 2
#  v_exact(x,y) = 0 
#############################################################################

# %% Options
# Fluid and Flow Setup
dim   = 2
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

forcing_x = lambda x: 0*x[:,0]
forcing_y = lambda x: 0*x[:,0] 

p_exact   = lambda x: (p_end-p_str)/L * x[:,0] + p_str
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
    tf.keras.layers.Dense(3)
])

x_PDE   = tf.random.uniform(shape = [num_PDE,  2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_hint  = tf.random.uniform(shape = [num_hint, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,   2], minval = [0, 0],  maxval = [0, H], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   2], minval = [L, 0],  maxval = [L, H], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   2], minval = [0, 0],  maxval = [L, 0], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   2], minval = [0, H],  maxval = [L, H], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test, 2], minval = [0, 0],  maxval = [L, H], dtype = ns.config.get_dtype())

p_test   = p_exact(x_test)[:, None]
u_test   = u_exact(x_test)[:, None]
v_test   = v_exact(x_test)[:, None]
u_hint   = u_exact(x_hint)[:, None]
v_hint   = v_exact(x_hint)[:, None]
p_hint   = p_exact(x_hint)[:, None]
inlet    = u_exact(x_BC_x0)[:, None]
p_inlet  = p_exact(x_BC_x0)[:, None]
p_outlet = p_exact(x_BC_x1)[:, None]
f_1 = forcing_x(x_PDE)
f_2 = forcing_y(x_PDE)

# %% Losses creation

def PDE(x, k, force):  # k is the coordinate of the vectorial equation
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        u_vect = model(x)
        u = u_vect[:,0]
        v = u_vect[:,1]
        p = u_vect[:,2]
        u_eq = u_vect[:,k]
        grad_eq = operator.gradient_scalar(tape, u_eq, x)
        dp   = operator.gradient_scalar(tape, p, x)[:,k]
        deqx = grad_eq[:,0]
        deqy = grad_eq[:,1]
        lapl_eq = operator.laplacian_scalar(tape, u_eq, x, dim)
    return (u * deqx + v * deqy) - (lapl_eq) / Re + dp - force

def BC_D(x, k, g_bc = None):
    with ns.GradientTape(persistent = True) as tape:
        if g_bc is None:
            samples = x.shape[0]
            g_bc = tf.zeros(shape = [samples,1], dtype = ns.config.get_dtype())
        tape.watch(x)
        uk = model(x)[:,k]
        return tf.math.abs(uk - g_bc)

def BC_N(x, k, j, g_bc = None):
    with ns.GradientTape(persistent = True) as tape:
        if g_bc is None:
            samples = x.shape[0]
            g_bc = tf.zeros(shape = [samples,1], dtype = ns.config.get_dtype())
        tape.watch(x)
        uk = model(x)[:,k]
        uk_j = operator.gradient_scalar(tape, uk, x)[:,j]
        return tf.math.abs(uk_j - g_bc)

def Hints():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_hint)
        u_vect = model(x_hint)
        u = u_vect[:,0]
        v = u_vect[:,1] 
        p = u_vect[:,2]
    return (u - u_hint) * (u - u_hint) + (v - v_hint) * (v - v_hint) + (p - p_test) * (p - p_test)

def test_loss():
    u_vect = model(x_test)
    u = u_vect[:,0]
    v = u_vect[:,1]
    p = u_vect[:,2]
    return (u - u_test) * (u - u_test) + (v - v_test) * (v - v_test) + (p - p_test) * (p - p_test)

# %% Losses definition
losses = [ns.LossMeanSquares(' PDE_U', lambda: PDE(x_PDE, 0, f_1), weight = 100.0),
          ns.LossMeanSquares(' PDE_V', lambda: PDE(x_PDE, 1, f_2), weight = 100.0),
          ns.LossMeanSquares( 'BC_x0', lambda: BC_D(x_BC_x0,0, inlet) + BC_D(x_BC_x0,1) + BC_D(x_BC_x0,2, p_inlet), weight = 10.0),
          ns.LossMeanSquares( 'BC_x1', lambda: BC_N(x_BC_x1,0,0) + BC_N(x_BC_x1,1,0) + BC_D(x_BC_x1,2, p_outlet), weight = 10.0),
          ns.LossMeanSquares( 'BC_y0', lambda: BC_D(x_BC_y0,0  ) + BC_D(x_BC_y0,1  ) + BC_N(x_BC_y0, 2, 1), weight = 10.0),
          ns.LossMeanSquares( 'BC_y1', lambda: BC_D(x_BC_y1,0  ) + BC_D(x_BC_y1,1  ) + BC_N(x_BC_y1, 2, 1), weight = 10.0),
          #ns.LossMeanSquares('Hints', Hints, weight = 15.0)
          ]
loss_test = ns.LossMeanSquares('fit', test_loss, normalization = num_test)

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 500)

# %% Saving Loss History

problem_name = "Poiseuille_Adimensional_Pressure"
history_file = os.path.join(cwd, "{}_history_loss.json".format(problem_name))
pb.save_history(history_file)
ns.utils.plot_history(history_file)
history = ns.utils.load_json(history_file)

# %% Post-processing
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,0].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('velocity u')
ax = fig.add_subplot(132, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], v_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,1].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('velocity v')
ax = fig.add_subplot(133, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], v_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test)[:,2].numpy(), label = 'numerical solution')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('pressure')

plt.show(block = True)
print("Reynolds Number -> {}".format(Re))