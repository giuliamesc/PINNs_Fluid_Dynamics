# %% Setup Options --- Import Libraries

# Main Libraries
import os
import numpy as np
import tensorflow as tf

# Setting Names and Working Directory 
problem_name = "Coronary_Flow"
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
n_pts["IC"]   = int(simulation_options[6])
n_pts["Vel"]  = int(simulation_options[7])
n_pts["Pres"] = int(simulation_options[8])
n_pts["Test"] = int(simulation_options[9])

use_collloss = True if n_pts["PDE"]  else False
use_boundary = True if n_pts["BC"]   else False
use_initialc = True if n_pts["IC"]   else False
fit_velocity = True if n_pts["Vel"]  else False
fit_pressure = True if n_pts["Pres"] else False

# %% Setup Options --- Setting Physical Parameters

dim = 3    # set 2D or 3D for operators

# Domain Dimensions
T     = 1e-2 # Temporal Horizon
dt    = 1e-4 # Time-step
n_times = int(T/dt)
H = np.sqrt(0.4**2+0.1**2)
U = 20
x0 = -1.4
y0 = -0.8
mu  = 1e-2   # kg/m*s
rho = 1.06e3 # kg/m^3
ni = 1e4*mu/rho
Re = (U*(H/2)/ni)

cos_theta = np.cos(np.arctan(1/4))
sin_theta = np.sin(np.arctan(1/4))
u_inf = lambda x: U * cos_theta * np.sqrt((x[:,0]-x0)**2+(x[:,1]-y0)**2) / H * (1 - np.sqrt((x[:,0]-x0)**2+(x[:,1]-y0)**2) / H)
v_inf = lambda x: U * sin_theta * np.sqrt((x[:,0]-x0)**2+(x[:,1]-y0)**2) / H * (1 - np.sqrt((x[:,0]-x0)**2+(x[:,1]-y0)**2) / H)

# Physical Forces
bnd_val = [{},{}]
bnd_val[0]["NOSL"] = 0
bnd_val[0] ["INF"] = None
bnd_val[0]["OUT1"] = 0
bnd_val[0] ["OUT2"] = 0
bnd_val[1]["NOSL"] = 0
bnd_val[1] ["INF"] = None
bnd_val[1]["OUT1"] = 0
bnd_val[1] ["OUT2"] = 0

# %% Mesh Import

import h5py
folder_h5 = "../../DataGeneration/data/Coronary"

# Mesh Loading
mesh_h5 = h5py.File(f'{folder_h5}/navier-stokes_SI_coronary_unsteady_00000.h5', "r")['Mesh']['0']['mesh']['geometry']

x_vec = mesh_h5[:,0]
y_vec = mesh_h5[:,1]

N = len(x_vec)

time_vec = np.arange(0.0, T, step = dt)
dom_grid = tf.convert_to_tensor([(t,x_vec[n],y_vec[n]) for t in time_vec for n in range(N)])

key_subset = ("PDE", "Vel", "Pres", "Test", "IC")
val_subset = np.split(np.random.permutation(np.array([i for i in range(dom_grid.shape[0])])), 
                      np.cumsum([n_pts[x] for x in key_subset]))[:-1]
idx_set = {k : v for (k,v) in zip(key_subset,val_subset)}

# %% Data Creation --- Exact Solution (Import Data or Analitical)

import h5py
folder_h5 = "../../DataGeneration/data/Coronary"
data_h5 = lambda x : h5py.File(f'{folder_h5}/navier-stokes_SI_coronary_unsteady_{x:05d}.h5', "r")['VisualisationVector']
uvel_h5 = lambda x : data_h5(x)["0"][:,0]
vvel_h5 = lambda x : data_h5(x)["0"][:,1]
pres_h5 = lambda x : data_h5(x)["1"][()] - np.mean(data_h5(x)["1"][()])

u_ex = tf.convert_to_tensor(np.concatenate([uvel_h5(time_step) for time_step in range(n_times)],  axis = 0))
v_ex = tf.convert_to_tensor(np.concatenate([vvel_h5(time_step) for time_step in range(n_times)],  axis = 0))
p_ex = tf.convert_to_tensor(np.concatenate([pres_h5(time_step) for time_step in range(n_times)],  axis = 0))

#p_ex = tf.reshape(p_ex, shape = [p_ex.shape[0],])

# Boundary conditions
bnd_val[0] ["INF"] = u_inf
bnd_val[1] ["INF"] = v_inf
# %% Data Creation --- Data Normalization 

spread = lambda vec: np.max(vec) - np.min(vec)
norm_vel = max([spread(u_ex), spread(v_ex)])
norm_pre = spread(p_ex)

u_ex_norm = u_ex / norm_vel
v_ex_norm = v_ex / norm_vel
p_ex_norm = p_ex / norm_pre
sol_norm  = [u_ex_norm , v_ex_norm, p_ex_norm]

# %% Data Creation --- Boundary and Initial Conditions

# Boundary Conditions
bnd_pts = {}

boundary_array = np.load(f'{folder_h5}/bpoints.npy')
n_bpts = len(boundary_array)
bnd_pts['NOSL'] = tf.convert_to_tensor(boundary_array[np.where(boundary_array[:,3]==0)][:,0:3])
bnd_pts['INF'] = tf.convert_to_tensor(boundary_array[np.where(boundary_array[:,3]==1)][:,0:3])
bnd_pts['OUT1'] = tf.convert_to_tensor(boundary_array[np.where(boundary_array[:,3]==2)][:,0:3])
bnd_pts['OUT2'] = tf.convert_to_tensor(boundary_array[np.where(boundary_array[:,3]==3)][:,0:3])

for key, value in bnd_val[0].items():
    zero_base = tf.zeros(len(bnd_pts[key]), dtype = np.double)
    bnd_val[0][key] = zero_base + value/norm_vel if type(value) == float or type(value) == int else zero_base + value(bnd_pts[key])/norm_vel
for key, value in bnd_val[1].items():
    zero_base = tf.zeros(len(bnd_pts[key]), dtype = np.double)
    bnd_val[1][key] = zero_base + value/norm_vel if type(value) == float or type(value) == int else zero_base + value(bnd_pts[key])/norm_vel

# %% Data Creation --- Noise Management

def generate_noise(n_pts, factor = 0, sd = 1.0, mn = 0.0):
    noise = tf.random.normal([n_pts], mean=mn, stddev=sd, dtype= np.double)
    return noise * factor

for key, _ in bnd_val[0].items():
    bnd_val[0][key] += generate_noise(len(bnd_pts[key]), noise_factor_bnd)
    bnd_val[1][key] += generate_noise(len(bnd_pts[key]), noise_factor_bnd)


u_ex_noise = tf.gather(u_ex_norm,idx_set["Vel"]) + generate_noise(n_pts[ "Vel"], noise_factor_fit)
v_ex_noise = tf.gather(v_ex_norm,idx_set["Vel"]) + generate_noise(n_pts[ "Vel"], noise_factor_fit)
p_ex_noise = tf.gather(p_ex_norm,idx_set["Pres"]) + generate_noise(n_pts["Pres"], noise_factor_fit)
sol_noise  = [u_ex_noise , v_ex_noise, p_ex_noise]

# %% Loss Building --- Differential Losses

gradient = ns.experimental.physics.tens_style.gradient_scalar

def PDE_MASS():
    x = tf.gather(dom_grid, idx_set["PDE"])
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        u_vect = model(x)[:,0:2]
        du_x = gradient(tape, u_vect[:,0], x)[:,1]
        dv_y = gradient(tape, u_vect[:,1], x)[:,2]
    return du_x + dv_y

def PDE_MOM(k):
    x = tf.gather(dom_grid, idx_set["PDE"])
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
        
        u_vect = model(x)
        p    = u_vect[:,2] * norm_pre
        u_eq = u_vect[:,k] * norm_vel

        dp    = gradient(tape, p, x)[:,k+1]
        du_t  = gradient(tape, u_eq, x)[:,0]
        du_x  = gradient(tape, u_eq, x)[:,1]
        du_y  = gradient(tape, u_eq, x)[:,2]
        du_xx = gradient(tape, du_x, x)[:,1]
        du_yy = gradient(tape, du_y, x)[:,2]

        conv1 = tf.math.multiply(norm_vel * u_vect[:,0], du_x)
        conv2 = tf.math.multiply(norm_vel * u_vect[:,1], du_y)
        unnormed_lhs = du_t - 1/Re * (du_xx + du_yy) + dp + conv1 + conv2
        norm_const = 1/max(norm_pre,norm_vel)

    return unnormed_lhs*norm_const

# %% Loss Building --- Dirichlet Style Losses

def dir_loss(points, component, rhs):
    uk = model(points)[:,component]
    return uk - rhs
def neu_loss(edge,k,rhs):
    x = bnd_pts[edge]
    if edge == 'OUT1':
        n = tf.constant([2,1], dtype='double')
        n= tf.reshape(n,[2,1])
    else : 
        n = tf.constant([1,0], dtype='double')
        n= tf.reshape(n,[2,1])
    with ns.GradientTape(persistent=True) as tape:
        tape.watch(x)
    u_vect = model(x)
    u_eq = u_vect[:,k] * norm_vel
    p_eq = u_vect[:,2] * norm_pre
    grad = gradient(tape, u_eq, x)[:,0:2]
    return ni * tf.linalg.matmul(grad,n) - p_eq * n[k] - rhs

BC_D = lambda edge, component:   dir_loss(bnd_pts[edge], component, bnd_val[component][edge])
BC_N = lambda edge, component:   neu_loss(edge, component, bnd_val[component][edge])
IN_C = lambda component:         dir_loss(tf.gather(dom_grid,idx_set["IC" ]), component, tf.zeros(shape = [n_pts["IC"]], dtype = np.double))
fit_velocity = lambda component: dir_loss(tf.gather(dom_grid,idx_set["Vel" ]), component, sol_noise[component])
fit_pressure = lambda:           dir_loss(tf.gather(dom_grid,idx_set["Pres"]), 2, sol_noise[2])
exact_value  = lambda component: dir_loss(tf.gather(dom_grid,idx_set["Test"]), component, tf.gather(sol_norm[component],idx_set["Test"]))

# %% Model's Setup --- Model Creation

LMS = ns.LossMeanSquares
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(dim,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(32, activation=tf.nn.tanh),
    tf.keras.layers.Dense(32, activation=tf.nn.tanh),
    tf.keras.layers.Dense(3)
])

PDE_losses = [LMS('PDE_MASS', lambda: PDE_MASS(), weight = 1e5),
              LMS('PDE_MOMU', lambda: PDE_MOM(0), weight = 1e4),
              LMS('PDE_MOMV', lambda: PDE_MOM(1), weight = 1e4)]
BCD_losses = [LMS('BCD_u_NS', lambda: BC_D( "NOSL", 0), weight = 1e0),
              LMS('BCD_v_NS', lambda: BC_D( "NOSL", 1), weight = 1e0),
              LMS('BCD_u_IN', lambda: BC_D( "INF", 0), weight = 1e0),
              LMS('BCD_v_IN', lambda: BC_D( "INF", 1), weight = 1e0),
              LMS('BCN_u_OUT1', lambda: BC_N("OUT1", 0), weight = 1e-3),
              LMS('BCN_v_OUT1', lambda: BC_N("OUT1", 1), weight = 1e-3),
              LMS('BCN_u_OUT2', lambda: BC_N("OUT2", 0), weight = 1e-3),
              LMS('BCN_v_OUT2', lambda: BC_N("OUT2", 1), weight = 1e-3)]
IN_losses = [LMS('IC_u', lambda: IN_C(0), weight = 1e0),
              LMS('IC_v', lambda: IN_C(1), weight = 1e0),
              LMS('IC_p', lambda: IN_C(2), weight = 1e0)]
FIT_V_Loss = [LMS('Fit_u', lambda: fit_velocity(0), weight = 1e0),
              LMS('Fit_v', lambda: fit_velocity(1), weight = 1e0)]
FIT_P_Loss = [LMS('Fit_p', lambda: fit_pressure(), weight = 1e0)]

losses = []
if use_collloss: losses += PDE_losses
if use_boundary: losses += BCD_losses
if use_initialc: losses += IN_losses
if fit_velocity: losses += FIT_V_Loss
if fit_pressure: losses += FIT_P_Loss

loss_test = [LMS('u_test', lambda: exact_value(0)),
             LMS('v_test', lambda: exact_value(1)),
             LMS('p_test', lambda: exact_value(2))]

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

# # %% Image Process --- Solutions on Regular Grid

# import pandas as pd

# # Regular Grid
# grid_x, grid_y = np.meshgrid(np.linspace(Le_x, Ue_x , 100), np.linspace(Le_y, Ue_y, 100))
# n_time_stamp = 4
# time_steps = np.linspace(0, T, n_time_stamp+1)
# p_ex_list, u_ex_list, v_ex_list = [], [], []

# # Numerical Solutions
# regular_mesh_file = r'../../DataGeneration/data/UnsteadyCase/navier-stokes_SI_cavity_unsteady_r.csv'
# dfr = pd.read_csv (regular_mesh_file)

# for t in time_steps:
#     if t == T: t = T - dt
#     temp_df = dfr.loc[(dfr["t"] > t-dt/4) & (dfr["t"] < t+dt/4)]
#     p_temp = pd.DataFrame(temp_df, columns = ['p']).to_numpy().reshape(grid_x.shape)
#     p_ex_list.append(p_temp-np.mean(p_temp))
#     u_ex_list.append(pd.DataFrame(temp_df, columns = ['ux']).to_numpy().reshape(grid_x.shape))
#     v_ex_list.append(pd.DataFrame(temp_df, columns = ['uy']).to_numpy().reshape(grid_x.shape))

# %% Image Process --- PINN Solutions 

# grid_x_flatten = np.reshape(grid_x, (-1,))
# grid_y_flatten = np.reshape(grid_y, (-1,))
# grid_t0 = np.zeros(grid_x_flatten.shape) 

# u_list, v_list, p_list = [], [], []

# for t in time_steps:
#     if t == T: t = T - dt
#     grid_t = grid_t0 + t
#     grid = tf.stack([grid_t, grid_x_flatten, grid_y_flatten], axis = -1)
#     u_list.append(model(grid)[:,0].numpy().reshape(grid_x.shape) * norm_vel)
#     v_list.append(model(grid)[:,1].numpy().reshape(grid_x.shape) * norm_vel)
#     p_list.append(model(grid)[:,2].numpy().reshape(grid_x.shape) * norm_pre)

# %% Image Process --- Contour Levels

# def find_lims(exact_sol, pinn_sol, take_max):
#     pfunc = max if take_max else min
#     nfunc = np.max if take_max else np.min
#     levels = [pfunc(nfunc(exact_sol[i]),nfunc(pinn_sol[i])) for i in range(n_time_stamp + 1)]
#     return pfunc(levels)

# lev_u_min, lev_u_max = (find_lims(u_ex_list, u_list, False), find_lims(u_ex_list, u_list, True))
# lev_v_min, lev_v_max = (find_lims(v_ex_list, v_list, False), find_lims(v_ex_list, v_list, True))
# lev_p_min, lev_p_max = (find_lims(p_ex_list, p_list, False), find_lims(p_ex_list, p_list, True))

# def approx_scale(x, up):
#     factor = np.floor(np.log10(abs(x)))-1
#     if up: x =  np.ceil(x/(np.power(10,factor))/5)
#     else : x = np.floor(x/(np.power(10,factor))/5)
#     return x*5*np.power(10,factor)

# num_levels = 11 
# level_u = np.linspace(approx_scale(lev_u_min, False), approx_scale(lev_u_max, True), num_levels)
# level_v = np.linspace(approx_scale(lev_v_min, False), approx_scale(lev_v_max, True), num_levels)
# level_p = np.linspace(approx_scale(lev_p_min, False), approx_scale(lev_p_max, True), num_levels)

# %% Image Process --- Countour Plots

import matplotlib.pyplot as plt

# def plot_subfig(fig, ax, function, levels, title):
#     ax.title.set_text(title)
#     cs = ax.contourf(grid_x, grid_y, function, levels = levels)
#     fig.colorbar(cs, ax=ax)

# for i,t in enumerate(time_steps):
#     graph_title = "Solutions when t = {0:.4f}".format(t)
#     graph_title += ", time step #{}/{}".format(int(i*(n_times/n_time_stamp)), n_times)
    
#     # Figure Creation
#     fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12,8))
#     fig.suptitle(graph_title , fontsize=18, y = 0.97, x = 0.50)
#     plt.subplots_adjust(top = 1.4, right = 1)
    
#     plot_subfig(fig, ax1, u_ex_list[i], level_u, 'Numerical u-velocity')
#     plot_subfig(fig, ax2, u_list[i], level_u, 'PINNS u-velocity')
#     plot_subfig(fig, ax3, v_ex_list[i], level_v, 'Numerical v-velocity')
#     plot_subfig(fig, ax4, v_list[i], level_v, 'PINNS v-velocity')
#     plot_subfig(fig, ax5, p_ex_list[i], level_p, 'Numerical Pressure')
#     plot_subfig(fig, ax6, p_list[i], level_p, 'PINNS Pressure')
    
#     plt.tight_layout()
#     saving_file = os.path.join(cwd, "{}//Graphic_{}_of_{}.jpg".format(saving_folder, i+1, n_time_stamp+1))
#     plt.savefig(saving_file)


# os.mkdir('PINN_sol')
for i in range(int(T/dt)):
    my_data = model(dom_grid[i:i+N])
    with h5py.File('PINN_sol/sol_pinn_%s.h5' % i,'w') as hf:
        hf.create_dataset("u_pinn",  data=my_data[:,0]*norm_vel)
        hf.create_dataset("v_pinn",  data=my_data[:,1]*norm_vel)
        hf.create_dataset("p_pinn",  data=my_data[:,2]*norm_pre)
        
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
plot_loss(history, ax, '-' , cmap(2), 1.5, 'Equations_Residuals', "losses", ["PDE_MASS", "PDE_MOMU", "PDE_MOMV"],)
plot_loss(history, ax, '-' , cmap(1), 1.5, 'Boundary_Cond_U',     "losses", ["BCD_u_x0", "BCD_u_x1", "BCD_u_y0", "BCD_u_y1", "IC_u"],)
plot_loss(history, ax, '-' , cmap(3), 1.5, 'Boundary_Cond_V',     "losses", ["BCD_v_x0", "BCD_v_x1", "BCD_v_y0", "BCD_v_y1", "IC_v"])
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