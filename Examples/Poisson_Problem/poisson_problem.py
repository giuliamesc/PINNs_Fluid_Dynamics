# %% Setup Options --- Import Libraries

# Main Libraries
import os
import numpy as np
import tensorflow as tf

# Setting Names and Working Directory 
problem_name = "Poisson_Problem"
cwd = os.path.abspath(os.getcwd())
nisaba_wd = "../../../nisaba"

# Import Nisaba Library
os.chdir(nisaba_wd)
import nisaba as ns
os.chdir(cwd)

# %% Setup Options --- Saving Options

save_results = True

default_name    = "Test_Case_#"
default_folder  = "Last_Training" 
recap_file_name = "Test_Options.txt"

test_cases = [x for x in os.listdir() if x.startswith(default_name)]
naming_idx = 1 if not test_cases else (max([int(x[len(default_name):]) for x in test_cases])+1)
saved_test_folder = f"{default_name}{naming_idx:03d}"

saving_folder  = saved_test_folder if save_results else default_folder  
if save_results: os.mkdir(saving_folder)
else: 
    if default_folder not in os.listdir(): os.mkdir(default_folder)

# %% Setup Options --- Setting Simulation Options

options_file_name = "simulation_options.txt"
options_file_path = os.path.join(cwd,options_file_name)
with open(options_file_path) as options_file:
    simulation_options = options_file.readlines()[0:-1:2]

epochs = int(simulation_options[1])
noise_factor_fit = float(simulation_options[2])
noise_factor_bnd = float(simulation_options[3])

n_pts = {}
n_pts["PDE"]  = int(simulation_options[4])
n_pts["BC"]   = int(simulation_options[5])
n_pts["Test"] = int(simulation_options[6])

use_collloss = True if n_pts["PDE"]  else False
use_boundary = True if n_pts["BC"]   else False


# %% Setup Options --- Setting Physical Parameters

dim = 2    # set 2D or 3D for operators

# Domain Dimensions 
Le_x  = 0         # Lower x extremum
Ue_x  = 2*np.pi   # Upper x extremum
Le_y  = 0         # Lower y extremum
Ue_y  = 2*np.pi   # Upper y extremum

# Physical Forces
bnd_val = [{},{}]
bnd_val[0]["BOT"] = 0
bnd_val[0] ["DX"] = None
bnd_val[0]["TOP"] = 0
bnd_val[0] ["SX"] = None

# %% Data Creation --- Building the Grid

n1, n2 = 100, 100
n = (n1+1)*(n2+1)

# Uniform Mesh
uniform_mesh = True
x_vec = np.linspace(Le_x, Ue_x, n1+1) if uniform_mesh else np.random.uniform(Le_x, Ue_x, n1+1)
y_vec = np.linspace(Le_y, Ue_y, n2+1) if uniform_mesh else np.random.uniform(Le_y, Ue_y, n2+1)

dom_grid = tf.convert_to_tensor([(i,j) for j in y_vec for i in x_vec])
#dom_grid = tf.cast(dom_grid, tf.float32)

key_subset = ("PDE", "Test")
val_subset = np.split(np.random.permutation(np.array([i for i in range(dom_grid.shape[0])])), 
                      np.cumsum([n_pts[x] for x in key_subset]))[:-1]
idx_set = {k : v for (k,v) in zip(key_subset,val_subset)}

# %% Data Creation --- Exact Solution 

u_f = lambda x: np.sin(x[:,0]) * np.sin(x[:,1])
f_f = lambda x: 2 * np.sin(x[:,0]) * np.sin(x[:,1])
g_f  = lambda x: np.sin(x[:,1])

bnd_val[0] ["SX"] = g_f
bnd_val[0] ["DX"] = g_f

f = tf.convert_to_tensor(f_f(tf.gather(dom_grid, idx_set["PDE"])))
u_ex = tf.convert_to_tensor(u_f(dom_grid))
# %% Data Creation --- Data Normalization 

spread = lambda vec: np.max(vec) - np.min(vec)
norm_vel = spread(u_ex)

f_norm = f / norm_vel
sol_norm  = u_ex / norm_vel

# %% Data Creation --- Boundary and Initial Conditions

bnd_pts = {}
boundary_sampling = lambda minval, maxval: tf.random.uniform(shape = [n_pts["BC"], dim], minval = minval, maxval = maxval, dtype = 'float64')

bnd_pts["BOT"] = boundary_sampling([Le_x, Le_y], [Ue_x, Le_y])
bnd_pts["DX"]  = boundary_sampling([Ue_x, Le_y], [Ue_x, Ue_y])
bnd_pts["TOP"] = boundary_sampling([Le_x, Ue_y], [Ue_x, Ue_y])
bnd_pts["SX"]  = boundary_sampling([Le_x, Le_y], [Le_x, Ue_y])

zero_base = tf.zeros(shape = [n_pts["BC"]], dtype = np.double)

for key, value in bnd_val[0].items():
    bnd_val[0][key] = zero_base + value/norm_vel if type(value) == float or type(value) == int else zero_base + value(bnd_pts[key])/norm_vel
for key, value in bnd_val[1].items():
    bnd_val[1][key] = zero_base + value/norm_vel if type(value) == float or type(value) == int else zero_base + value(bnd_pts[key])/norm_vel

# %% Data Creation --- Noise Management

def generate_noise(n_pts, factor = 0, sd = 1.0, mn = 0.0):
    noise = tf.random.normal([n_pts], mean=mn, stddev=sd, dtype= np.double)
    return noise * factor

for key, _ in bnd_val[0].items():
    bnd_val[0][key] += generate_noise(n_pts["BC"], noise_factor_bnd)


# %% Loss Building --- Differential Losses

gradient   = ns.experimental.physics.tens_style.gradient_scalar
laplacian  = ns.experimental.physics.tens_style.laplacian_scalar

def PDE():
    x = tf.gather(dom_grid, idx_set["PDE"])
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x)
        u = model(x)
    return - laplacian(tape, u, x, dim) - f_norm

def neu_loss(x, k, j, rhs = 0):
    with ns.GradientTape(persistent = True) as tape:
        tape.watch(x)
        
        uk = model(x)[:,k] * norm_vel
        uk_j = gradient(tape, uk, x)[:,j]
        
        norm_const = 1/norm_vel
        
    return norm_const * (uk_j - rhs)

    
# %% Loss Building --- Dirichlet Style Losses

def dir_loss(points, component, rhs):
    uk = model(points)[:,component]
    return uk - rhs

BC_D = lambda edge, component:              dir_loss(bnd_pts[edge], component, bnd_val[component][edge])
BC_N = lambda edge, component, direction:   neu_loss(bnd_pts[edge], component, direction, bnd_val[component][edge])

exact_value  = lambda component : dir_loss(tf.gather(dom_grid,idx_set["Test"]), component, tf.gather(sol_norm,idx_set["Test"]))

# %% Model's Setup --- Model Creation

LMS = ns.LossMeanSquares
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(dim,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(32, activation=tf.nn.tanh),
    tf.keras.layers.Dense(32, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

PDE_losses = [LMS('PDE', lambda: PDE(), weight = 1e0)]

BCD_losses = [LMS('BCD_u_y0', lambda: BC_D("BOT", 0), weight = 1e0),              
              LMS('BCD_u_y1', lambda: BC_D("TOP", 0), weight = 1e0)]
              
BCN_losses = [LMS('BCN_u_x0', lambda: BC_N("SX",0,0), weight = 1e0),
              LMS('BCN_u_x1', lambda: BC_N("DX",0,0), weight = 1e0)]


losses = []
if use_collloss: losses += PDE_losses
if use_boundary: losses += (BCD_losses + BCN_losses)

loss_test = [LMS('u_test', lambda: exact_value(0))]

# %% Model's Setup --- Training Section

loss_image_file = os.path.join(cwd, "{}//Loss_Trend_Full.png".format(saving_folder))
history_file    = os.path.join(cwd, "{}//History_Loss.json".format(saving_folder))

pb = ns.OptimizationProblem(model.variables, losses, loss_test, callbacks=[])
pb.callbacks.append(ns.utils.HistoryPlotCallback(frequency=100, gui=False,
                                                 filename=loss_image_file,
                                                 filename_history=history_file))
ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=1e-2), num_epochs = 100)
ns.minimize(pb, 'scipy', 'BFGS', num_epochs = epochs)

if save_results:
    with open('{}//Model.json'.format(saving_folder), "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights('{}//Weights.h5'.format(saving_folder))

# %% Image Process --- Solutions on Regular Grid

# Regular Grid
grid_x, grid_y = np.meshgrid(np.linspace(Le_x, Ue_x , 100), np.linspace(Le_y, Ue_y, 100))

grid_x_flatten = np.reshape(grid_x, (-1,))
grid_y_flatten = np.reshape(grid_y, (-1,)) 

grid = tf.stack([grid_x_flatten, grid_y_flatten], axis = -1)

# Numerical Solutions
p_ex_list = p_f(grid).numpy().reshape(grid_x.shape)
u_ex_list = u_f(grid).numpy().reshape(grid_x.shape)
v_ex_list = v_f(grid).numpy().reshape(grid_x.shape)

# %% Image Process --- PINN Solutions 

grid = tf.stack([grid_x_flatten, grid_y_flatten], axis = -1)
u_list = model(grid)[:,0].numpy().reshape(grid_x.shape) * norm_vel
v_list = model(grid)[:,1].numpy().reshape(grid_x.shape) * norm_vel
p_list = model(grid)[:,2].numpy().reshape(grid_x.shape) * norm_pre

# %% Image Process --- Contour Levels

def find_lims(exact_sol, pinn_sol, take_max):
    pfunc = max if take_max else min
    nfunc = np.max if take_max else np.min
    levels = pfunc(nfunc(exact_sol),nfunc(pinn_sol))
    return levels

lev_u_min, lev_u_max = (find_lims(u_ex_list, u_list, False), find_lims(u_ex_list, u_list, True))
lev_v_min, lev_v_max = (find_lims(v_ex_list, v_list, False), find_lims(v_ex_list, v_list, True))
lev_p_min, lev_p_max = (find_lims(p_ex_list, p_list, False), find_lims(p_ex_list, p_list, True))

def approx_scale(x, up):
    factor = np.floor(np.log10(abs(x)))-1
    if up: x =  np.ceil(x/(np.power(10,factor))/5)
    else : x = np.floor(x/(np.power(10,factor))/5)
    return x*5*np.power(10,factor)

num_levels = 11 
level_u = np.linspace(approx_scale(lev_u_min, False), approx_scale(lev_u_max, True), num_levels)
level_v = np.linspace(approx_scale(lev_v_min, False), approx_scale(lev_v_max, True), num_levels)
level_p = np.linspace(approx_scale(lev_p_min, False), approx_scale(lev_p_max, True), num_levels)

if level_u[-1]/level_v[-1] > 1e3: level_v *= 1e3

# %% Image Process --- Countour Plots

import matplotlib.pyplot as plt

def plot_subfig(fig, ax, function, levels, title):
    ax.title.set_text(title)
    cs = ax.contourf(grid_x, grid_y, function, levels = levels)
    fig.colorbar(cs, ax=ax)


graph_title = "Solutions of the {} problem".format(problem_name)
    
# Figure Creation
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,8))
fig.suptitle(graph_title , fontsize=18, y = 0.97, x = 0.50)
plt.subplots_adjust(top = 1.4, right = 1)
    
plot_subfig(fig, ax1, u_ex_list, level_u, 'Numerical u-velocity')
plot_subfig(fig, ax2, u_list, level_u, 'PINNS u-velocity')
plot_subfig(fig, ax3, v_ex_list, level_v, 'Numerical v-velocity')
plot_subfig(fig, ax4, v_list, level_v, 'PINNS v-velocity')
plot_subfig(fig, ax5, p_ex_list, level_p, 'Numerical Pressure')
plot_subfig(fig, ax6, p_list, level_p, 'PINNS Pressure')
    
plt.tight_layout()
saving_file = os.path.join(cwd, "{}//Graphic.jpg".format(saving_folder))
plt.savefig(saving_file)
    
# %% Image Process --- Loss Trend Graphs

from matplotlib.cm import get_cmap
cmap = get_cmap("Set1")

def plot_loss(history, ax, style, color, lwdt, label, first_key, second_key):
    values_tot = [history[first_key][key]["weight"] * np.array(history[first_key][key]["log"]) for key in second_key]
    value_tot = sum(values_tot) / len(second_key)
    hist_mod = [x for x in history['log']['iter']]
    ax.plot(hist_mod, value_tot, style, color = color, linewidth = lwdt, label = label)
    ax.set_xscale('symlog', linthresh = 100, linscale = 1)
    ax.set_yscale('log')

history = ns.utils.load_json(history_file)
fig, ax = plt.subplots(figsize = (10, 8))
ax.loglog(history['log']['iter'], history['log']['loss_global'], 'k-', linewidth = 2)

plot_loss(history, ax, '--', cmap(0), 3.0, 'Test_Loss',           "losses_test", ["u_test", "v_test", "p_test"])
plot_loss(history, ax, '-' , cmap(2), 1.5, 'Equations_Residuals', "losses", ["PDE_MASS", "PDE_MOMU", "PDE_MOMV"])
plot_loss(history, ax, '-' , cmap(1), 1.5, 'Boundary_Cond_U',     "losses", ["BCD_u_x0", "BCN_u_x1", "BCD_u_y0", "BCD_u_y1"])
plot_loss(history, ax, '-' , cmap(3), 1.5, 'Boundary_Cond_V',     "losses", ["BCD_v_x0", "BCN_v_x1", "BCD_v_y0", "BCD_v_y1"])
plot_loss(history, ax, '-' , cmap(4), 1.5, 'Fitting Loss',        "losses", ["Fit_u", "Fit_v", "Fit_p"])
plt.axvline(  0, 0, 1, c = cmap(5))
plt.axvline(100, 0, 1, c = cmap(5))

plt.text(  1, 0.3, "keras_Adam", bbox={'facecolor':'lightgray','alpha':0.7,'edgecolor':'black','pad':3}, rotation = 90)
plt.text(101, 0.3, "scipy_BFGS", bbox={'facecolor':'lightgray','alpha':0.7,'edgecolor':'black','pad':3}, rotation = 90)

ax.legend(loc = 1, fontsize = 15)
ax.grid()
ax.set_xlabel('# Iterations', fontsize = 15)
ax.set_ylabel('Losses Values', fontsize = 15)

plt.savefig(os.path.join(cwd, "{}//Loss_Trend_Reduced.png".format(saving_folder)))

# %% Final Recap

recap_info = []
recap_info.append("Problem Name    -> {}".format(problem_name))
recap_info.append("Training Epochs -> {} epochs".format(epochs))
recap_info.append("Pyhsical PDE Losses  -> {} points".format(n_pts["PDE"]))
recap_info.append("Boundary Conditions  -> {} points".format(n_pts["BC"]))
recap_info.append("Initial  Conditions  -> {} points".format(n_pts["IC"]))
recap_info.append("Fitting Velocity  -> {} points".format(n_pts["Vel"]  if fit_velocity else 0))
recap_info.append("Fitting Pressure  -> {} points".format(n_pts["Pres"] if fit_pressure else 0))
recap_info.append("Noise on Boundary -> {} times a gaussian N(0,1)".format(noise_factor_bnd))
recap_info.append("Noise on Domain   -> {} times a gaussian N(0,1)".format(noise_factor_fit))

recap_file_path = os.path.join(os.path.join(cwd, saving_folder),recap_file_name)
recap_file = open(recap_file_path, "w")                                                                          
print("\nSIMULATION OPTIONS RECAP...")
for row_string in recap_info:
    print("\t",row_string)
    recap_file.write(row_string+"\n")
recap_file.close()
