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
# - u_xx - u_yy = 2 * sin(x) * sin(y)        in \Omega = (0, 2 pi)^2
# u(x,y) = 0                                 on (0, 2 pi) x {0, 2 pi}
# u_x(x,y) = sin(y)                          on {0, 2 pi} x (0, 2 pi)
#
# u_exact(x,y) = sin(x) * sin(y)
#############################################################################

# %% Options
# Problem setup
domain_W1  = 2*np.pi
domain_W2  = 2*np.pi

dim = 2
u_exact = lambda x: np.sin(x[:,0]) * np.sin(x[:,1])
forcing = lambda x: 2 * np.sin(x[:,0]) * np.sin(x[:,1])

# Numerical options
num_PDE  = 200
num_BC   = 20 # points for each edge
num_test = 1000

# %% Inizialization

#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(2,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

x_PDE   = tf.random.uniform(shape = [num_PDE, 2], minval = [0, 0],         maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,  2], minval = [0, 0],         maxval = [0, domain_W2],         dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,  2], minval = [domain_W1, 0], maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,  2], minval = [0, 0],         maxval = [domain_W1, 0],         dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,  2], minval = [0, domain_W2], maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test,2], minval = [0, 0],         maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_BC_D  = tf.concat([x_BC_y0, x_BC_y1], axis = 0)
x_BC_N  = tf.concat([x_BC_x0, x_BC_x1], axis = 0)

u_test = u_exact(x_test)[:, None]
f = forcing(x_PDE)
g = np.sin(x_BC_N[:,1])

def PDE():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_PDE)
        u = model(x_PDE)
        laplacian = nse.physics.tens_style.laplacian_scalar(tape, u, x_PDE, dim)
    return - laplacian - f

def BC_N():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_N)
        u = model(x_BC_N)
        grad_u = nse.physics.tens_style.gradient_scalar(tape, u, x_BC_N)
        dux = grad_u @ tf.constant([[1],[0]], dtype = tf.float64)
    return dux - g

# %% Losses definition
losses = [ns.LossMeanSquares( 'PDE', PDE, weight = 2.0),
          ns.LossMeanSquares('BC_D', lambda: model(x_BC_D)),
          ns.LossMeanSquares('BC_N', BC_N, weight = 5.0)]

loss_test = ns.LossMeanSquares('fit', lambda: model(x_test) - u_test)

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 10)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 500)

# %% Post-processing
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test).numpy(), label = 'numerical solution')
ax.legend()

plt.show(block = True)

# %% 
def compute_gradient(x_set):
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_set)
        u = model(x_set)
        grad_u = nse.physics.tens_style.gradient_scalar(tape, u, x_set)
    return grad_u

x_set = x_BC_N
u = model(x_set)[:,0]
grad_u = compute_gradient(x_set)
u_x = grad_u[:,0]
print(u.shape)
li = ["[x: {}, y: {}] -> u: {}, u_x: {}".format(x[0],x[1],u,u_x) for x,u,u_x in zip(x_BC_N,u,u_x-g)]
for el in li:
    print(el)
