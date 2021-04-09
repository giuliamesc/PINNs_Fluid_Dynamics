#!/usr/bin/env python3
import sys
import os
sys.path.append('C:\\Users\\giuli\\OneDrive\\Desktop\\Numerical Analysis for PDEs\\Project\\nisaba')
import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf
import numpy as np

#############################################################################
# - mu (u_xx + u_yy) = 2 * sin(x) * sin(y)        in \Omega = (0, 2 pi)^2
# u(x,y) = 0                                      on \partial\Omega
#
# u_exact(x,y) = ( sin(x) * sin(y) ) / mu
#############################################################################

# %% Options
# Problem setup
dim = 2

domain_W1  = 2*np.pi
domain_W2  = 2*np.pi

mu_exact   = 1.2
mu_guess   = 0.0

u_exact = lambda x: ( np.sin(x[:,0]) * np.sin(x[:,1]) ) / mu_exact
forcing = lambda x: 2 * np.sin(x[:,0]) * np.sin(x[:,1])

# Numerical options
num_PDE  = 500
num_BC   = 20 # points for each edge
num_fit  = 100
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

mu = ns.Variable(mu_guess, name = 'mu')

x_PDE   = tf.random.uniform(shape = [num_PDE, 2], minval = [0, 0],         maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_fit   = tf.random.uniform(shape = [num_fit, 2], minval = [0, 0],         maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test,2], minval = [0, 0],         maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,  2], minval = [0, 0],         maxval = [0, domain_W2],         dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,  2], minval = [domain_W1, 0], maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,  2], minval = [0, 0],         maxval = [domain_W1, 0],         dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,  2], minval = [0, domain_W2], maxval = [domain_W1, domain_W2], dtype = ns.config.get_dtype())
x_BC = tf.concat([x_BC_x0, x_BC_x1, x_BC_y0, x_BC_y1], axis = 0)

u_fit  = u_exact(x_fit)[:, None]
u_test = u_exact(x_test)[:, None]
f = forcing(x_PDE)

def PDE():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_PDE)
        u = model(x_PDE)
        laplacian = nse.physics.tens_style.laplacian_scalar(tape, u, x_PDE, dim)
    return - mu * laplacian - f

# %% Losses definition
u_norm = tf.reduce_mean(tf.square(u_fit))

losses = [ns.LossMeanSquares('PDE', PDE),
          ns.LossMeanSquares('fit', lambda: model(x_fit) - u_fit, normalization = u_norm),
          ns.LossMeanSquares('BC', lambda: model(x_BC), weight = 1e1)]

loss_test = [ns.LossMeanSquares('fit', lambda: model(x_test) - u_test, normalization = u_norm),
             ns.LossMeanSquares('mu_error', lambda: (mu - mu_exact)/abs(mu_exact)),
             ns.Loss('mu', lambda: mu)]

# %% Training
pltcb = ns.utils.HistoryPlotCallback()
pb = ns.OptimizationProblem(model.variables + [mu], losses, loss_test, callbacks = pltcb)

pb.compile()

#ns.minimize(pb, 'nisaba', 'lm', num_epochs = 100)
ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 500)

print('mu (exact)     : %f' % mu_exact)
print('mu (identified): %f' % mu.numpy())
print('relative error : %1.3e' % (abs(mu.numpy() - mu_exact)/mu_exact))

pltcb.finalize(pb, block = False)
#%% Post-processing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test[:,0], x_test[:,1], u_test, label = 'exact solution')
ax.scatter(x_test[:,0], x_test[:,1], model(x_test).numpy(), label = 'numerical solution')
ax.legend()

plt.show(block = True)