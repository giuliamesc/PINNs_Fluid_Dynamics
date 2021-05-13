# %% Import libraries and working directory settings
import os
cwd = os.path.abspath(os.getcwd())
os.chdir("../")
os.chdir("../")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
import tensorflow as tf
import numpy as np

problem_name = "Colliding_Flows"

#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# %% Case Study
#############################################################################
#  u_x + v_y = 0                                           in \Omega = (-1, 1) x (-1, 1)
#  - (u_xx + u_yy) + p_x = 0                               in \Omega = (-1, 1) x (-1, 1)
#  - (v_xx + v_yy) + p_y = 0                               in \Omega = (-1, 1) x (-1, 1)
#  u = 20*x*y^3                                            on \partial\Omega
#  v = 5*x^4-5*y^4                                         on \partial\Omega
#
#  p_exact(x,y) = 60*x^2*y-20*y^3+const
#  u_exact(x,y) = 20*x*y^3
#  v_exact(x,y) = 5*x^4-5*y^4 
#############################################################################

# %% Physical Options

# Fluid and Flow Setup
dim   = 2     # set 2D or 3D for operators
a     = -1    # Lower extremum 
b     = +1    # Upper extremum

# Adimensionalization
Re = 1
L = 2

# %% Exact Solution and Forcing Terms

forcing_x = lambda x: 0*x[:,0]
forcing_y = lambda x: 0*x[:,0] 

p_exact   = lambda x: (60*x[:,0]*x[:,0]*x[:,1]-20*x[:,1]*x[:,1]*x[:,1])
u_exact   = lambda x: 20*x[:,0]*x[:,1]*x[:,1]*x[:,1]
v_exact   = lambda x: 5*x[:,0]*x[:,0]*x[:,0]*x[:,0]-5*x[:,1]*x[:,1]*x[:,1]*x[:,1]
 

# %% Numerical options

num_PDE  = 50
num_BC   = 20
num_col  = 4
num_test = 1000
num_pres = 20

# %% Simulation Options

use_noise = False
collocation = False

# %% Domain Tensors

x_PDE   = tf.random.uniform(shape = [num_PDE,  2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_col   = tf.random.uniform(shape = [num_col,  2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,   2], minval = [a, a],  maxval = [a, b], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   2], minval = [b, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   2], minval = [a, a],  maxval = [b, a], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   2], minval = [a, b],  maxval = [b, b], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test, 2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_pres = tf.random.normal(  shape = [num_pres, 2], mean = 0.0, stddev = 0.1, dtype = ns.config.get_dtype())


# %% Setting Boundary Conditions

# Dirichlet
dc_bound_cond = []
dc_bound_cond.append(x_BC_x0)
dc_bound_cond.append(x_BC_x1)
dc_bound_cond.append(x_BC_y0)
dc_bound_cond.append(x_BC_y1)
if dc_bound_cond:
    x_BCD = tf.concat(dc_bound_cond, axis = 0) 

# Neumann
ne_bound_cond = []
if ne_bound_cond:
    x_BCN = tf.concat(ne_bound_cond, axis = 0) 

# %% Normalization Costants

u_max = np.max(np.abs(u_exact(x_BCD)))
v_max = np.max(np.abs(v_exact(x_BCD)))
p_max = np.max(np.abs(p_exact(x_BCD)))
vel_max = max([u_max, v_max])

# %% Model Creation

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(2,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(3)
])


# %% Right-hand side generation

def generate_noise(x, factor = 0, sd = 1.0, mn = 0.0): 
    shape = x.shape[0]
    noise = tf.random.normal([shape], mean=mn, stddev=sd, dtype= ns.config.get_dtype())
    return noise * factor

def create_rhs(x, force, noise = None):
    samples = x.shape[0]
    if noise is None:
        noise = tf.zeros(shape = [samples], dtype = ns.config.get_dtype())
    if force is None:
        return tf.zeros(shape = [samples], dtype = ns.config.get_dtype()) + noise
    if type(force) == float:
        return tf.zeros(shape = [samples], dtype = ns.config.get_dtype()) + force + noise
    return force(x) + noise
 
# %% Noise Creation

if use_noise:
    BCD_noise_x = generate_noise(x_BCD, factor = 1e-1)
    BCD_noise_y = generate_noise(x_BCD, factor = 1e-1)
else:
    BCD_noise_x = None
    BCD_noise_y = None    
 
# %% PDE Losses creation

# k is the coordinate of the vectorial equation
# j is the direction in which the derivative is computed

def PDE_MASS(x):
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x_PDE)
        u_vect = model(x_PDE)[:,0:2] * vel_max
        div = operator.divergence_vector(tape, u_vect, x_PDE, dim)
    return div

def PDE_MOM(x, k, force):
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        
        u_vect = model(x)
        p = u_vect[:,2] * p_max
        u_eq = u_vect[:,k] * vel_max
        
        dp   = operator.gradient_scalar(tape, p, x)[:,k]
        lapl_eq = operator.laplacian_scalar(tape, u_eq, x, dim)
        
        rhs = create_rhs(x, force)
    return - (lapl_eq) + dp - rhs

# %% Boundary Losses creation

def BC_D(x, k, g_bc = None, norm = 1, noise = None):     
    uk = model(x)[:,k]
    rhs = create_rhs(x, g_bc, noise)
    return uk - rhs / norm

# %% Collocation and Test Losses

def exact_value(x, k, sol = None, norm = 1):
    uk = model(x)[:,k]
    rhs = create_rhs(x, sol)
    return uk - rhs / norm

# %% Other Losses

def PRESS_0(x):
    uk = model(x)[:,2]
    uk_mean = tf.abs(tf.math.reduce_mean(uk))
    return uk_mean

# %% Training Losses definition

PDE_losses = [ns.LossMeanSquares('PDE_MASS', lambda: PDE_MASS(x_PDE), normalization = 1e4, weight = 1e0),
              ns.LossMeanSquares('PDE_MOMU', lambda: PDE_MOM(x_PDE, 0, forcing_x), normalization = 1e4, weight = 1e-2),
              ns.LossMeanSquares('PDE_MOMV', lambda: PDE_MOM(x_PDE, 1, forcing_y), normalization = 1e4, weight = 1e-2),
              ns.Loss('PRESS_0',  lambda: PRESS_0(x_pres), normalization = 1e0, weight = 1e-2, non_negative = True)
              ]
BCD_losses = [ns.LossMeanSquares('BCD_u', lambda: BC_D(x_BCD, 0, u_exact, vel_max, BCD_noise_x), weight = 1e0),
              ns.LossMeanSquares('BCD_v', lambda: BC_D(x_BCD, 1, v_exact, vel_max, BCD_noise_y), weight = 1e0)
              ]
COL_Losses = [ns.LossMeanSquares('col_u', lambda: exact_value(x_col, 0, u_exact, vel_max), weight = 1e0),
              ns.LossMeanSquares('col_v', lambda: exact_value(x_col, 1, v_exact, vel_max), weight = 1e0)
              ]

losses = []
losses += PDE_losses 
losses += BCD_losses 
if collocation:
    losses += COL_Losses

# %% Test Losses definition
loss_test = [ns.LossMeanSquares('u_fit', lambda: exact_value(x_test, 0, u_exact, vel_max)),
             ns.LossMeanSquares('v_fit', lambda: exact_value(x_test, 1, v_exact, vel_max)),
             ns.LossMeanSquares('p_fit', lambda: exact_value(x_test, 2, p_exact, p_max))
             ]

# %% Training
pb = ns.OptimizationProblem(model.variables, losses, loss_test)
#pb.compile(optimizers  = 'scipy')

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'L-BFGS-B', num_epochs = 2000)

# %% Saving Loss History

history_file = os.path.join(cwd, "Images\\{}_history_loss.json".format(problem_name))
pb.save_history(history_file)
ns.utils.plot_history(history_file)
history = ns.utils.load_json(history_file)

# %% Post-processing

import matplotlib.pyplot as plt

def plot_image(fig_counter, title, exact, numerical):
    fig_counter += 1
    fig = plt.figure(fig_counter)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_test[:,0], x_test[:,1], exact, label = 'exact solution')
    ax.scatter(x_test[:,0], x_test[:,1], numerical, label = 'numerical solution')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(title)
    image_file = os.path.join(cwd, "Images\\{}_{}.png".format(problem_name, title))
    plt.savefig(image_file)

# Image 1 is "Loss History"
plot_image(2, "velocity_u", u_exact(x_test)[:, None], model(x_test)[:,0].numpy() * vel_max)
plot_image(3, "velocity_v", v_exact(x_test)[:, None], model(x_test)[:,1].numpy() * vel_max)
plot_image(4,   "pressure", p_exact(x_test)[:, None], model(x_test)[:,2].numpy() * p_max)

plt.show(block = False)

# %% Final recap

print("\nSIMULATION OPTIONS RECAP...")
print("\tPressure mean ->", np.mean( model(x_test)[:,2].numpy()))
print("\tData Noise    ->", use_noise)
print("\tCollocation   ->", collocation)
print("\tDirichlet bc  ->", len(dc_bound_cond), "edges")
print("\tNeumann   bc  ->", len(ne_bound_cond), "edges")

