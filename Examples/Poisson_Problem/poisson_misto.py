#%%
import os
cwd = os.path.abspath(os.getcwd())
os.chdir("../")
os.chdir("../")
os.chdir("../")
os.chdir("nisaba")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
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
g_neum  = lambda x: np.sin(x_BC_N[:,1])

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
g = g_neum(x_BC_N)

def PDE():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_PDE)
        u = model(x_PDE)
        laplacian = operator.laplacian_scalar(tape, u, x_PDE, dim)
    return - laplacian - f

def BC_D():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_D)
        u = model(x_BC_D)
    return u

def BC_N():
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x_BC_N)
        u = model(x_BC_N)
        u_x = operator.gradient_scalar(tape, u, x_BC_N)[:,0]
    return u_x - g

# %% Losses definition
losses = [ns.LossMeanSquares( 'PDE', PDE, weight = 1e2),
          ns.LossMeanSquares('BC_D', BC_D),
          ns.LossMeanSquares('BC_N', BC_N)]

loss_test = ns.LossMeanSquares('fit', lambda: model(x_test) - u_test)

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 10)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 7500)

# %% Saving Loss History

problem_name = "Poisson_Misto"
history_file = os.path.join(cwd, "{}_history_loss.json".format(problem_name))
pb.save_history(history_file)
ns.utils.plot_history(history_file)
history = ns.utils.load_json(history_file)

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
