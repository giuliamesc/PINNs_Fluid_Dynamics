# %% Import libraries and working directory settings
import os
cwd = os.path.abspath(os.getcwd())
os.chdir("../../../nisaba")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
os.chdir(cwd)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

problem_name = "Lid Driven Cavity - Steady"

#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Reading the CSV file with the numerical solutions
df = pd.read_csv (r'../../DataGeneration/data/navier-stokes_cavity_steady.csv')


# %% Case Study
#############################################################################
#  u_x + v_y = 0                                        in \Omega = (0, 1) x (0, 1)
#  - (u_xx + u_yy) + u * u_x + v * u_y + p_x = 0        in \Omega = (0, 1) x (0, 1)
#  - (v_xx + v_yy) + u * v_x + v * v_y + p_y = 0        in \Omega = (0, 1) x (0, 1)
#  u = v = 0                                            on {0,1} x (0,1), (0,1) x {0}
#  u = 500                                              on (0,1) x {1}
#
#############################################################################

# %% Physical Options

# Fluid and Flow Setup
dim   = 2    # set 2D or 3D for operators
a     = 0    # Lower extremum 
b     = 1    # Upper extremum
U     = 500  # x-velocity on the upper boundary

# %% Forcing Terms and Extraction of Numerical Solutions

forcing_x = lambda x: 0*x[:,0]
forcing_y = lambda x: 0*x[:,0] 

p_num   = pd.DataFrame(df, columns= ['p']).to_numpy()
u_num   = pd.DataFrame(df, columns= ['ux']).to_numpy()
v_num   = pd.DataFrame(df, columns= ['uy']).to_numpy()

#Points for the numerical solution
x_num   = pd.DataFrame(df, columns= ['x','y']).to_numpy()

# %% Numerical options

num_PDE  = 50
num_BC   = 50
num_col  = 50
num_test = 2000
num_pres = 100

# %% Simulation Options

epochs      = 7500
use_noise   = False
collocation = True
press_mode  = "Collocation" # Options -> "Collocation", "Mean", "None"

# %% Domain Tensors

x_PDE   = tf.convert_to_tensor(x_num[:num_PDE,:])
x_col   = tf.convert_to_tensor(x_num[num_PDE:num_PDE+num_col,:])
x_BC_x0 = tf.random.uniform(shape = [num_BC,   2], minval = [a, a],  maxval = [a, b], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   2], minval = [b, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   2], minval = [a, a],  maxval = [b, a], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   2], minval = [a, b],  maxval = [b, b], dtype = ns.config.get_dtype())
x_test  = tf.convert_to_tensor(x_num[num_PDE+num_col:num_PDE+num_col+num_test,:])
x_pres  = tf.convert_to_tensor(x_num[num_PDE+num_col+num_test:num_PDE+num_col+num_test+num_pres,:])


# %% Setting Boundary Conditions

# Dirichlet
dc_bound_cond = []
dc_bound_cond.append(x_BC_x0)
dc_bound_cond.append(x_BC_x1)
dc_bound_cond.append(x_BC_y0)
x_BCD_0 = tf.concat(dc_bound_cond, axis = 0) 

# %% Normalization Costants

u_max = np.max(u_num)-np.min(u_num)
v_max = np.max(v_num)-np.min(v_num)
p_max = np.max(p_num)-np.min(p_num)
vel_max = max([u_max, v_max])

# Mean of the pressure 
p_mean = np.mean(p_num[num_PDE+num_col+num_test:num_PDE+num_col+num_test+num_pres,:])

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
    if force is None or force == 0:
        return tf.zeros(shape = [samples], dtype = ns.config.get_dtype()) + noise
    if type(force) == np.float64 or type(force) == int:
        return tf.zeros(shape = [samples], dtype = ns.config.get_dtype()) + force + noise
    return force(x) + noise
 
# %% Noise Creation

if use_noise:
    BCD_noise_x = generate_noise(x_BCD_0, factor = 1e-1)
    BCD_noise_y = generate_noise(x_BCD_0, factor = 1e-1)
    BCD_noise_x_up = generate_noise(x_BC_y1, factor = 1e-1)
    BCD_noise_y_up = generate_noise(x_BC_y1, factor = 1e-1)
else:
    BCD_noise_x = None
    BCD_noise_y = None    
    BCD_noise_x_up = None
    BCD_noise_y_up = None
# %% PDE Losses creation

# k is the coordinate of the vectorial equation
# j is the direction in which the derivative is computed

def PDE_MASS(x):
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        u_vect = model(x)[:,0:2] * vel_max
        div = operator.divergence_vector(tape, u_vect, x, dim)
    return div

def PDE_MOM(x, k, force):
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        
        u_vect = model(x)
        p = u_vect[:,2] * p_max
        u_eq = u_vect[:,k] * vel_max
        
        dp   = operator.gradient_scalar(tape, p, x)[:,k]
        lapl_eq = operator.laplacian_scalar(tape, u_eq, x, dim)
        
        du_x = operator.gradient_scalar(tape, u_eq, x)[:,0]
        du_y = operator.gradient_scalar(tape, u_eq, x)[:,1]
        
        conv1 = tf.math.multiply(vel_max * u_vect[:,0], du_x)
        conv2 = tf.math.multiply(vel_max * u_vect[:,1], du_y)
        
        rhs = create_rhs(x, force)
        
    return - (lapl_eq) + dp + conv1 + conv2 - rhs

# %% Boundary Losses creation

def BC_D(x, k, f, norm = 1, noise = None): 
    uk = model(x)[:,k]
    rhs = create_rhs(x, f, noise)
    norm_rhs = rhs/norm
    return uk - norm_rhs

# %% Collocation and Test Losses

def col_pressure(x, sol, norm = 1):
    p = model(x)[:,2]
    samples = sol[num_PDE+num_col+num_test:num_PDE+num_col+num_test+num_pres,:]
    norm_rhs = tf.squeeze(tf.convert_to_tensor(samples/norm))
    return p - norm_rhs

def col_velocity(x, k, sol, norm = 1):
    u = model(x)[:,k]
    samples = sol[num_PDE:num_PDE+num_col,:]
    norm_rhs = tf.squeeze(tf.convert_to_tensor(samples/norm))
    return u - norm_rhs

# %% Other Losses

def exact_value(x, k, sol, norm = 1):
    uk = model(x)[:,k]
    samples = sol[num_PDE+num_col:num_PDE+num_col+num_test,:]
    norm_rhs = tf.squeeze(tf.convert_to_tensor(samples/norm))
    return uk - norm_rhs

def PRESS_MEAN(x, p, norm = 1):
    uk = model(x)[:,2]
    uk_mean = tf.abs(tf.math.reduce_mean(uk))
    rhs = create_rhs(x, p/norm)
    return uk_mean - rhs

# %% Training Losses definition

PDE_losses = [ns.LossMeanSquares('PDE_MASS', lambda: PDE_MASS(x_PDE), normalization = 1e4, weight = 1e-2),
              ns.LossMeanSquares('PDE_MOMU', lambda: PDE_MOM(x_PDE, 0, forcing_x), normalization = 1e4, weight = 1e-2),
              ns.LossMeanSquares('PDE_MOMV', lambda: PDE_MOM(x_PDE, 1, forcing_y), normalization = 1e4, weight = 1e-2)
              ]
              
BCD_losses = [ns.LossMeanSquares('BCD_u_x0', lambda: BC_D(x_BC_x0, 0, 0, vel_max, BCD_noise_x), weight = 1e0),
              ns.LossMeanSquares('BCD_v_x0', lambda: BC_D(x_BC_x0, 1, 0, vel_max, BCD_noise_y), weight = 1e0),
              ns.LossMeanSquares('BCD_u_x1', lambda: BC_D(x_BC_x1, 0, 0, vel_max, BCD_noise_x), weight = 1e0),
              ns.LossMeanSquares('BCD_v_x1', lambda: BC_D(x_BC_x1, 1, 0, vel_max, BCD_noise_y), weight = 1e0),
              ns.LossMeanSquares('BCD_u_y0', lambda: BC_D(x_BC_y0, 0, 0, vel_max, BCD_noise_x), weight = 1e0),
              ns.LossMeanSquares('BCD_v_y0', lambda: BC_D(x_BC_y0, 1, 0, vel_max, BCD_noise_y), weight = 1e0),
              ns.LossMeanSquares('BCD_u_y1', lambda: BC_D(x_BC_y1, 0, U, vel_max, BCD_noise_x_up), weight = 1e0),
              ns.LossMeanSquares('BCD_v_y1', lambda: BC_D(x_BC_y1, 1, 0, vel_max, BCD_noise_y_up), weight = 1e0)
              ]
COL_Losses = [ns.LossMeanSquares('COL_u', lambda: col_velocity(x_col, 0, u_num, vel_max), weight = 1e0),
              ns.LossMeanSquares('COL_v', lambda: col_velocity(x_col, 1, v_num, vel_max), weight = 1e0)
              ]
COL_P_Loss = [ns.LossMeanSquares('COL_p', lambda: col_pressure(x_pres, p_num, p_max), weight = 1e0)]

MEAN_P_Loss = [ns.LossMeanSquares('MEAN_p', lambda: PRESS_MEAN(x_pres, p_mean, p_max), weight = 1e-6)]

losses = []
losses += PDE_losses 
losses += BCD_losses 

if collocation:
    losses += COL_Losses
if press_mode == "Collocation":
    losses += COL_P_Loss
if press_mode == "Mean":
    losses += MEAN_P_Loss

# %% Test Losses definition
loss_test = [ns.LossMeanSquares('u_fit', lambda: exact_value(x_test, 0, u_num, vel_max)),
             ns.LossMeanSquares('v_fit', lambda: exact_value(x_test, 1, v_num, vel_max)),
             ns.LossMeanSquares('p_fit', lambda: exact_value(x_test, 2, p_num, p_max))
             ]

# %% Training

loss_image_file = os.path.join(cwd, "Images//{}_LossTrend.png".format(problem_name))
history_file    = os.path.join(cwd, "Images//{}_history_loss.json".format(problem_name))

pb = ns.OptimizationProblem(model.variables, losses, loss_test, callbacks=[])
pb.callbacks.append(ns.utils.HistoryPlotCallback(frequency=100, gui=False, 
                                                 filename=loss_image_file,
                                                 filename_history=history_file))

ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'BFGS', num_epochs = epochs)

# # %% Loss Post-processing
# def plot_loss(history, first_key, second_key, ax, style, label):
#     value_tot = 0
#     count = 0
#     for key in second_key:
#         count += 1
#         log_np = np.array(history[first_key][key]["log"])
#         value_tot += history[first_key][key]["weight"] * log_np
#     value_tot /= count
#     ax.plot(history['log']['iter'], \
#             value_tot, \
#             style, linewidth = 1.5, \
#             label = label)
#     ax.set_xscale('symlog')
#     ax.set_yscale('log')

# history = ns.utils.load_json(history_file)
# fig = plt.figure(5, figsize = (10, 8))
# ax = fig.add_subplot()
# #ax.loglog(history['log']['iter'], history['log']['loss_global'], 'k-', linewidth = 2)

# plot_loss(history, "losses", ["PDE_MASS", "PDE_MOMU", "PDE_MOMV"], ax, 'b-', 'Equations_Residuals')
# plot_loss(history, "losses", ["BCD_u", "BCD_v", "COL_p"], ax, 'g-', 'Boundary_Conditions')
# plot_loss(history, "losses_test", ["u_fit", "v_fit", "p_fit"], ax, 'm--', 'Test_Loss')
# plt.axvline(100, 0, 1, c = "r")
# plt.axvline(0, 0, 1, c = "r")    

# ax.legend(loc = 3, fontsize = 20)
# ax.grid()
# ax.set_xlabel('# iterations')

# graph_file = os.path.join(cwd, "Images//{}_LossTrend_simplified.png".format(problem_name))
# plt.savefig(graph_file)

# %% Images Post-processing

def plot_image(fig_counter, title, exact, numerical, x):
    fig = plt.figure(fig_counter)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:,0], x[:,1], exact, label = 'exact solution')
    ax.scatter(x[:,0], x[:,1], numerical, label = 'numerical solution')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(title)
    image_file = os.path.join(cwd, "Images//{}_{}.png".format(problem_name, title))
    plt.savefig(image_file)

# Image 1 is "Loss History"
plot_image(2, "velocity_u", u_num[num_PDE+num_col:num_PDE+num_col+num_test,:], model(x_test)[:,0].numpy() * vel_max, x_test)
plot_image(3, "velocity_v", v_num[num_PDE+num_col:num_PDE+num_col+num_test,:], model(x_test)[:,1].numpy() * vel_max, x_test)
plot_image(4,   "pressure", p_num[num_PDE+num_col:num_PDE+num_col+num_test,:], model(x_test)[:,2].numpy() * p_max, x_test)

plt.show(block = False)

# %% Final recap

print("\nSIMULATION OPTIONS RECAP...")
print("\tEpochs        ->", epochs)
print("\tPressure mean -> {:e}".format(np.mean(model(x_test)[:,2].numpy())))
print("\tData Noise    ->", use_noise)
print("\tCollocation   ->", collocation)
print("\tPressure Mode ->", press_mode)


# %% Countour Plots

grid_x, grid_y = np.meshgrid(np.linspace(a, b , 100), np.linspace(a, b, 100))


grid_x_flatten = np.reshape(grid_x, (-1,))
grid_y_flatten = np.reshape(grid_y, (-1,))
grid = tf.stack([grid_x_flatten, grid_y_flatten], axis = -1)


u = model(grid)[:,0].numpy().reshape(grid_x.shape) * v_max
v = model(grid)[:,1].numpy().reshape(grid_x.shape) * v_max
p = model(grid)[:,2].numpy().reshape(grid_x.shape) * p_max

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
 
fig.suptitle('Left: PINN, Right: Numerical')

ax1.contourf(grid_x, grid_y, u)
ax3.contourf(grid_x, grid_y, v)
ax5.contourf(grid_x, grid_y, p)

df2 = pd.read_csv (r'../../DataGeneration/data/navier-stokes_cavity_steady_r.csv')

my_p_num   = pd.DataFrame(df2, columns= ['p']).to_numpy().reshape(grid_x.shape)
my_u_num   = pd.DataFrame(df2, columns= ['ux']).to_numpy().reshape(grid_x.shape)
my_v_num   = pd.DataFrame(df2, columns= ['uy']).to_numpy().reshape(grid_x.shape)

ax2.contourf(grid_x, grid_y, my_u_num)
ax4.contourf(grid_x, grid_y, my_v_num)
ax6.contourf(grid_x, grid_y, my_p_num)
