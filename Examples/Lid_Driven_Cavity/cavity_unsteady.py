# %% Import libraries and working directory settings
import os
import time
cwd = os.path.abspath(os.getcwd())
os.chdir("../../../nisaba")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
os.chdir(cwd)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py 
from random import sample


#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

problem_name = "Lid Driven Cavity - Unsteady"


# %% Settings for saving and loading

model_name_load = None
model_name_save = None
load_mode = False
save_mode = False

# %% Case Study
######################################################################################
#
#  \Omega = (0, 1) x (0, 1) T = 1e-2
#  u_x + v_y = 0                                        in \Omega x (0, T)
#  u_t - (u_xx + u_yy) + u * u_x + v * u_y + p_x = 0    in \Omega x (0, T)
#  u_t - (v_xx + v_yy) + u * v_x + v * v_y + p_y = 0    in \Omega x (0, T)
#  u = v = 0                                            on {0,1} x (0,1) x (0, T), (0,1) x {0} x [0, T)
#  u = 1                                                on (0,1) x {1} x [0, T)
#  v = p = 0                                            in \Omega x {0}
#  u = 0                                                in \Omega x {0}
#
######################################################################################


# %% Physical Options

# Fluid and Flow Setup
dim   = 3    # set 2D or 3D for operators
a     = 0    # Lower extremum
b     = 1    # Upper extremum
U     = 1    # x-velocity on the upper boundary
T     = 1e-2 # Temporal Horizon

# %% Numerical options

num_PDE  = 50000
num_BC   = 5000
num_CI   = 5000
num_col  = 10000
num_pres = 10000
num_test = 10000

# %% Simulation Options

epochs        = 100

use_pdelosses = False
use_boundaryc = False
use_initialco = False
coll_velocity = True
coll_pressure = True

use_noise     = False

# %% Grid of time and geometry

n1 = 100
n2 = 100

n = (n1+1)*(n2+1)
dt = 1e-4

num_times = int(T/dt)

N = (n1+1)*(n2+1)*num_times

time_vector = np.arange(0.0, T, step = dt)
x = np.linspace(a, b, n1+1)
y = np.linspace(a, b, n2+1)

var_np = np.zeros((N, dim), dtype ='float')
count = 0
for t in time_vector:
    for j in y:
        for i in x:
            var_np[count, :] = (t, i, j)
            count = count + 1 
            
var = tf.convert_to_tensor(var_np)

# %% Reading the h5 file with the numerical solutions

count = 0 #number of the time instant

u = []
v = []
p = []
p_mean = []

for count in range(100):
    if count < 10 :
        file_name = r'../../DataGeneration/data/UnsteadyCase/navier-stokes_SI_cavity_unsteady_0000%s.h5' % count
    else :
        file_name = r'../../DataGeneration/data/UnsteadyCase/navier-stokes_SI_cavity_unsteady_000%s.h5' % count
    
    hf = h5py.File(file_name,'r')
    dset = hf['VisualisationVector']
    vel = dset['0']
    vel_data = np.zeros((n,dim), dtype='<f8')
    vel.read_direct(vel_data)

    u.append(vel_data[:,0])
    v.append(vel_data[:,1])

    p_data = dset['1']
    pp = np.zeros((n,1), dtype='<f8')
    p_data.read_direct(pp)
    
    # Pressure Mean Equal to 0
    p.append(pp-np.mean(pp))
    
    count = count + 1

u_num = tf.convert_to_tensor(np.concatenate(u, axis = 0))
v_num = tf.convert_to_tensor(np.concatenate(v, axis = 0))
p_num = tf.convert_to_tensor(np.concatenate(p, axis = 0))


# %% Forcing Terms 

forcing_x = lambda x: 0*x[:,0]
forcing_y = lambda x: 0*x[:,0]

# %% Domain Tensors

sequence = [i for i in range(N)] 

subset_PDE = sample(sequence, num_PDE) #extracting a random sample of num_PDE indexes
subset_col = sample(sequence, num_col) #extracting a random sample of num_col indexes
subset_pres = sample(sequence, num_pres) #extracting a random sample of num_pres indexes
subset_test = sample(sequence, num_test) #extracting a random sample of num_test indexes
x_PDE = tf.gather(var, subset_PDE)

x_BC_x0 = tf.random.uniform(shape = [num_BC,   3], minval = [0, a, a],  maxval = [T, a, b], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   3], minval = [0, b, a],  maxval = [T, b, b], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   3], minval = [0, a, a],  maxval = [T, b, a], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   3], minval = [0, a, b],  maxval = [T, b, b], dtype = ns.config.get_dtype())

x_CI   = tf.random.uniform(shape = [num_CI,  3], minval = [0, a, a], maxval = [0, b, b], dtype = ns.config.get_dtype())

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

# %% Model Creation

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(dim,), activation=tf.nn.tanh),
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
        du_x = operator.gradient_scalar(tape, u_vect[:,0], x)[:,1]
        dv_y = operator.gradient_scalar(tape, u_vect[:,1], x)[:,2]
    return du_x + dv_y

def PDE_MOM(x, k, force):
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)

        u_vect = model(x)
        p = u_vect[:,2] * p_max
        u_eq = u_vect[:,k] * vel_max

        dp   = operator.gradient_scalar(tape, p, x)[:,k+1]
        
        du_t = operator.gradient_scalar(tape, u_eq, x)[:,0]
        du_x = operator.gradient_scalar(tape, u_eq, x)[:,1]
        du_y = operator.gradient_scalar(tape, u_eq, x)[:,2]
        du_xx = operator.gradient_scalar(tape, du_x, x)[:,1]
        du_yy = operator.gradient_scalar(tape, du_y, x)[:,2]

        conv1 = tf.math.multiply(vel_max * u_vect[:,0], du_x)
        conv2 = tf.math.multiply(vel_max * u_vect[:,1], du_y)

        rhs = create_rhs(x, force)

    return du_t - du_xx - du_yy + dp + conv1 + conv2 - rhs

# %% Boundary Losses creation

def BC_D(x, k, f, norm = 1, noise = None):
    uk = model(x)[:,k]
    rhs = create_rhs(x, f, noise)
    norm_rhs = rhs/norm
    return uk - norm_rhs

# %% Initial Conditions Losses creation

def BC_IN(x, k, f, norm = 1, noise = None):
    uk = model(x)[:,k]
    rhs = create_rhs(x, f, noise)
    norm_rhs = rhs/norm
    return uk - norm_rhs

# %% Collocation and Test Losses

def col_pressure(idx, sol, norm = 1):
    p = model(tf.gather(var,idx))[:,2]
    samples = tf.gather(sol,idx)
    norm_rhs = tf.squeeze(tf.convert_to_tensor(samples/norm))
    return p - norm_rhs

def col_velocity(idx, k, sol, norm = 1):
    u = model(tf.gather(var,idx))[:,k]
    samples = tf.gather(sol,idx)
    norm_rhs = tf.squeeze(tf.convert_to_tensor(samples/norm))
    return u - norm_rhs

# %% Other Losses

def exact_value(idx, k, sol, norm = 1):
    uk = model(tf.gather(var,idx))[:,k]
    samples = tf.gather(sol,idx)
    norm_rhs = tf.squeeze(tf.convert_to_tensor(samples/norm))
    return uk - norm_rhs

# %% Training Losses definition

PDE_losses = [ns.LossMeanSquares('PDE_MASS', lambda: PDE_MASS(x_PDE), normalization = 1e0, weight = 1e-2),
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
IN_losses = [ns.LossMeanSquares('CI_u', lambda: BC_IN(x_CI, 0, 0, vel_max), weight = 1e0),
             ns.LossMeanSquares('CI_v', lambda: BC_IN(x_CI, 1, 0, vel_max), weight = 1e0),
             ns.LossMeanSquares('CI_p', lambda: BC_IN(x_CI, 2, 0, p_max), weight = 1e0)]

COL_Losses = [ns.LossMeanSquares('COL_u', lambda: col_velocity(subset_col, 0, u_num, vel_max), weight = 1e0),
              ns.LossMeanSquares('COL_v', lambda: col_velocity(subset_col, 1, v_num, vel_max), weight = 1e0)
              ]
COL_P_Loss = [ns.LossMeanSquares('COL_p', lambda: col_pressure(subset_col, p_num, p_max), weight = 1e0)]

losses = []
if use_pdelosses: losses += PDE_losses
if use_boundaryc: losses += BCD_losses
if use_initialco: losses += IN_losses
if coll_velocity: losses += COL_Losses
if coll_pressure: losses += COL_P_Loss

# %% Test Losses definition

loss_test = [ns.LossMeanSquares('u_fit', lambda: exact_value(subset_test, 0, u_num, vel_max)),
             ns.LossMeanSquares('v_fit', lambda: exact_value(subset_test, 1, v_num, vel_max)),
             ns.LossMeanSquares('p_fit', lambda: exact_value(subset_test, 2, p_num, p_max))
             ]

# %% Training

loss_image_file = os.path.join(cwd, "Images//{}_LossTrend.png".format(problem_name))
history_file    = os.path.join(cwd, "Images//{}_history_loss.json".format(problem_name))

if not load_mode:
    pb = ns.OptimizationProblem(model.variables, losses, loss_test, callbacks=[])
    pb.callbacks.append(ns.utils.HistoryPlotCallback(frequency=100, gui=False,
                                                     filename=loss_image_file,
                                                     filename_history=history_file))
    ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
    ns.minimize(pb, 'scipy', 'BFGS', num_epochs = epochs)

# %% Model Loading

if load_mode and model_name_load is not None:
    os.chdir("Saved_Model")
    json_file = open('{}.json'.format(model_name_load), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights('{}.h5'.format(model_name_load))
    os.chdir("..")

# %% Model Saving

if save_mode and model_name_save is not None:
    os.chdir("Saved_Model")
    model_json = model.to_json()
    with open('{}.json'.format(model_name_save), "w") as json_file:
        json_file.write(model_json)
    model.save_weights('{}.h5'.format(model_name_save))
    os.chdir("..")

# %% Solutions on Regular Grid

n_time_stamp = 4  # must divide num_times = int(T/dt)
time_steps = np.linspace(0, T, n_time_stamp+1)
p_num_list = []
u_num_list = []
v_num_list = []

# Regular Grid
grid_x, grid_y = np.meshgrid(np.linspace(a, b , 100), np.linspace(a, b, 100))

# Numerical Solutions
regular_mesh_file = r'../../DataGeneration/data/UnsteadyCase/navier-stokes_SI_cavity_unsteady_r.csv'
df2 = pd.read_csv (regular_mesh_file)

for t in time_steps:
    if t == T: t = T - dt
    temp_df = df2[df2["t"] == t]
    p_temp = pd.DataFrame(temp_df, columns = [ 'p']).to_numpy().reshape(grid_x.shape)
    p_num_list.append(p_temp-np.mean(p_temp))
    u_num_list.append(pd.DataFrame(temp_df, columns = ['ux']).to_numpy().reshape(grid_x.shape))
    v_num_list.append(pd.DataFrame(temp_df, columns = ['uy']).to_numpy().reshape(grid_x.shape))

# %% PINN Solutions 
grid_x_flatten = np.reshape(grid_x, (-1,))
grid_y_flatten = np.reshape(grid_y, (-1,))
grid_t0 = np.zeros(grid_x_flatten.shape) 

u_list = []
v_list = []
p_list = []

for t in time_steps:
    if t == T: t = T - dt
    grid_t = grid_t0 + t
    grid = tf.stack([grid_t, grid_x_flatten, grid_y_flatten], axis = -1)
    u_list.append(model(grid)[:,0].numpy().reshape(grid_x.shape) * v_max)
    v_list.append(model(grid)[:,1].numpy().reshape(grid_x.shape) * v_max)
    p_list.append(model(grid)[:,2].numpy().reshape(grid_x.shape) * p_max)

# %% Contour Levels
def approx_scale(x, up):
    factor = np.floor(np.log10(abs(x)))-1
    if up: x =  np.ceil(x/(np.power(10,factor))/5)
    else : x = np.floor(x/(np.power(10,factor))/5)
    return x*5*np.power(10,factor)

u_min, u_max = [], []
v_min, v_max = [], []
p_min, p_max = [], []

for i, _ in enumerate(time_steps):
    u_min.append(min(np.min(u_list[i]),np.min(u_num_list[i])))
    u_max.append(max(np.max(u_list[i]),np.max(u_num_list[i])))
    v_min.append(min(np.min(v_list[i]),np.min(v_num_list[i])))
    v_max.append(max(np.max(v_list[i]),np.max(v_num_list[i])))
    p_min.append(min(np.min(p_list[i]),np.min(p_num_list[i])))
    p_max.append(max(np.max(p_list[i]),np.max(p_num_list[i])))

lev_u_min, lev_u_max = (min(u_min), max(u_max))
lev_v_min, lev_v_max = (min(v_min), max(v_max))
lev_p_min, lev_p_max = (min(p_min), max(p_max))

num_levels = 11 
level_u = np.linspace(approx_scale(lev_u_min, False), approx_scale(lev_u_max, True), num_levels)
level_v = np.linspace(approx_scale(lev_v_min, False), approx_scale(lev_v_max, True), num_levels)
level_p = np.linspace(approx_scale(lev_p_min, False), approx_scale(lev_p_max, True), num_levels)


# %% Countour Plots

for i,t in enumerate(time_steps):
    graph_title = "Solutions when t = {0:.4f}".format(t)
    graph_title += ", time step #{}/{}".format(int(i*(num_times/n_time_stamp)), num_times)
    
    # Figure Creation
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,6))
    fig.suptitle(graph_title , fontsize=18, y = 1.5, x = 0.55)
    plt.subplots_adjust(top = 1.4, right = 1)
    ax1.title.set_text('Numerical u-velocity')
    ax2.title.set_text('PINNS u-velocity')
    ax3.title.set_text('Numerical v-velocity')
    ax4.title.set_text('PINNS v-velocity')
    ax5.title.set_text('Numerical Pressure')
    ax6.title.set_text('PINNS Pressure')
    
    # Numerical Plots
    cs1 = ax1.contourf(grid_x, grid_y, u_num_list[i], levels = level_u)
    fig.colorbar(cs1, ax=ax1)
    cs3 = ax3.contourf(grid_x, grid_y, v_num_list[i], levels = level_v)
    fig.colorbar(cs3, ax=ax3)
    cs5 = ax5.contourf(grid_x, grid_y, p_num_list[i], levels = level_p)
    fig.colorbar(cs5, ax=ax5)

    # PINN Plots
    cs2 = ax2.contourf(grid_x, grid_y, u_list[i], levels = level_u)
    fig.colorbar(cs2, ax=ax2)
    cs4 = ax4.contourf(grid_x, grid_y, v_list[i], levels = level_v)
    fig.colorbar(cs4, ax=ax4)
    cs6 = ax6.contourf(grid_x, grid_y, p_list[i], levels = level_p)
    fig.colorbar(cs6, ax=ax6)
    

# %% Final recap

print("\nSIMULATION OPTIONS RECAP...")
print("\tEpochs             ->", epochs)
print("\tPinns points       ->", num_PDE)
print("\tBoundary points    ->", num_BC)
print("\tInitial  points    ->", num_CI)
print("\tCollocation points ->", num_col)
print("\tPressure points    ->", num_pres)
print("\tTest points        ->", num_test)
print("\nFENICS FILES RECAP")
print("\tUnstructured Mesh last change ->", time.ctime(os.path.getmtime(regular_mesh_file)))
print("\tStructured   Mesh last change ->", time.ctime(os.path.getmtime(regular_mesh_file)))
