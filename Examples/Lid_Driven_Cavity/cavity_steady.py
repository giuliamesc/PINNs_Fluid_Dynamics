# %% Import libraries and working directory settings
import os
cwd = os.path.abspath(os.getcwd())
os.chdir("../")
os.chdir("../")
os.chdir("../nisaba")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
os.chdir(cwd)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.math import multiply as product
import pandas as pd


problem_name = "Lid Driven Cavity - Steady"

#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Reading the CSV file with the numerical solutions
df = pd.read_csv (r'../../DataGeneration/data/navier-stokes_cavity_steady.csv')
#print (df)

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
dim   =  2    # set 2D or 3D for operators
a     = 0    # Lower extremum 
b     = 1    # Upper extremum
U     = 500  # x-velocity on the upper boundary

# %% Forcing Terms and Extraction of Numerical Solutions

forcing_x = lambda x: 0*x[:,0]
forcing_y = lambda x: 0*x[:,0] 

p_num   = pd.DataFrame(df, columns= ['p']).to_numpy()
u_num   = pd.DataFrame(df, columns= ['ux']).to_numpy()
v_num   = pd.DataFrame(df, columns= ['uy']).to_numpy()

# %% Numerical options

num_PDE  = 50
num_BC   = 20
num_col  = 20
num_test = 500
num_pres = 20

# %% Simulation Options

epochs      = 2000
use_noise   = False
collocation = False
press_mode  = "Collocation" # Options -> "Collocation", "Mean", "None"

# %% Domain Tensors

x_PDE   = tf.random.uniform(shape = [num_PDE,  2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_col   = tf.random.uniform(shape = [num_col,  2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_BC_x0 = tf.random.uniform(shape = [num_BC,   2], minval = [a, a],  maxval = [a, b], dtype = ns.config.get_dtype())
x_BC_x1 = tf.random.uniform(shape = [num_BC,   2], minval = [b, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_BC_y0 = tf.random.uniform(shape = [num_BC,   2], minval = [a, a],  maxval = [b, a], dtype = ns.config.get_dtype())
x_BC_y1 = tf.random.uniform(shape = [num_BC,   2], minval = [a, b],  maxval = [b, b], dtype = ns.config.get_dtype())
x_test  = tf.random.uniform(shape = [num_test, 2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())
x_pres  = tf.random.uniform(shape = [num_pres, 2], minval = [a, a],  maxval = [b, b], dtype = ns.config.get_dtype())


# %% Setting Boundary Conditions

# Dirichlet
dc_bound_cond = []
dc_bound_cond.append(x_BC_x0)
dc_bound_cond.append(x_BC_x1)
dc_bound_cond.append(x_BC_y0)
x_BCD = tf.concat(dc_bound_cond, axis = 0) 

# %% Normalization Costants

u_max = np.max(np.abs(u_num))
v_max = np.max(np.abs(v_num))
p_max = np.max(np.abs(p_num))
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
        
        dux = operator.gradient_scalar(tape, u_eq, x)[:,0]
        duy = operator.gradient_scalar(tape, u_eq, x)[:,1]
        
        conv = u_vect[:,0] * dux + u_vect[:,1] * duy
        
        rhs = create_rhs(x, force)
    return - (lapl_eq) + dp + conv - rhs

# %% Boundary Losses creation

def BC_D1(x, k, norm = 1, noise = None):     
    uk = model(x)[:,k]
    return uk 

def BC_D2(x, k, U, norm = 1, noise = None):
    uk = model(x)[:,k]
    return uk - U/norm

# %% Collocation and Test Losses

def col_pressure(x, k, sol = None, norm = 1):
    uk = model(x)[:,k]

    samples = sol[:n]
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
              ns.LossMeanSquares('PDE_MOMV', lambda: PDE_MOM(x_PDE, 1, forcing_y), normalization = 1e4, weight = 1e-2)
              ]
              
BCD_losses = [ns.LossMeanSquares('BCD_u', lambda: BC_D1(x_BCD, 0, vel_max, BCD_noise_x), weight = 1e0),
              ns.LossMeanSquares('BCD_v', lambda: BC_D1(x_BCD, 1, vel_max, BCD_noise_x), weight = 1e0),
              ns.LossMeanSquares('BCD_u_up', lambda: BC_D2(x_BC_y1, 1, U, vel_max, BCD_noise_y), weight = 1e0)
              ]
#COL_Losses = [ns.LossMeanSquares('COL_u', lambda: exact_value(x_col, 0, u_num, vel_max), weight = 1e0),
#              ns.LossMeanSquares('COL_v', lambda: exact_value(x_col, 1, v_num, vel_max), weight = 1e0)
#              ]
COL_P_Loss = [ns.LossMeanSquares('COL_p', lambda: col_pressure(x_pres, 2, p_num, p_max), weight = 1e0)]
#PRESS_Loss = [ns.Loss('PRESS_0',  lambda: PRESS_0(x_pres), normalization = 1e0, weight = 1e-2, non_negative = True)]

losses = []
losses += PDE_losses 
losses += BCD_losses 

if collocation:
    losses += COL_Losses
if press_mode == "Collocation":
    losses += COL_P_Loss
if press_mode == "Mean":
    losses += PRESS_Loss
    

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

# %% Loss Post-processing
def plot_loss(history, first_key, second_key, ax, style, label):
    value_tot = 0
    count = 0
    for key in second_key:
        count += 1
        log_np = np.array(history[first_key][key]["log"])
        value_tot += history[first_key][key]["weight"] * log_np
    value_tot /= count
    ax.plot(history['log']['iter'], \
            value_tot, \
            style, linewidth = 1.5, \
            label = label)
    ax.set_xscale('symlog')
    ax.set_yscale('log')

history = ns.utils.load_json(history_file)
fig = plt.figure(5, figsize = (10, 8))
ax = fig.add_subplot()
#ax.loglog(history['log']['iter'], history['log']['loss_global'], 'k-', linewidth = 2)

plot_loss(history, "losses", ["PDE_MASS", "PDE_MOMU", "PDE_MOMV"], ax, 'b-', 'Equations_Residuals')
plot_loss(history, "losses", ["BCD_u", "BCD_v", "COL_p"], ax, 'g-', 'Boundary_Conditions')
plot_loss(history, "losses_test", ["u_fit", "v_fit", "p_fit"], ax, 'm--', 'Test_Loss')
plt.axvline(100, 0, 1, c = "r")
plt.axvline(0, 0, 1, c = "r")    

ax.legend(loc = 3, fontsize = 20)
ax.grid()
ax.set_xlabel('# iterations')

graph_file = os.path.join(cwd, "Images//{}_LossTrend_simplified.png".format(problem_name))
plt.savefig(graph_file)

# %% Images Post-processing

#Points for the numerical solution
x_num   = pd.DataFrame(df, columns= ['x']).to_numpy()
y_num   = pd.DataFrame(df, columns= ['y']).to_numpy()

def plot_image(fig_counter, title, exact, numerical, x_n, y_n):
    fig = plt.figure(fig_counter)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_test[:,0], x_test[:,1], exact, label = 'exact solution')
    ax.scatter(x_n, y_n, numerical, label = 'numerical solution')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(title)
    image_file = os.path.join(cwd, "Images//{}_{}.png".format(problem_name, title))
    plt.savefig(image_file)

# Image 1 is "Loss History"
plot_image(2, "velocity_u", u_num, model(x_test)[:,0].numpy() * vel_max, x_num, y_num)
plot_image(3, "velocity_v", v_num, model(x_test)[:,1].numpy() * vel_max, x_num, y_num)
plot_image(4,   "pressure", p_num, model(x_test)[:,2].numpy() * p_max, x_num, y_num)

plt.show(block = False)

# %% Final recap

print("\nSIMULATION OPTIONS RECAP...")
print("\tEpochs        ->", epochs)
print("\tPressure mean -> {:e}".format(np.mean(model(x_test)[:,2].numpy())))
print("\tData Noise    ->", use_noise)
print("\tCollocation   ->", collocation)
print("\tPressure Mode ->", press_mode)
print("\tDirichlet bc  ->", len(dc_bound_cond), "edges")
print("\tNeumann   bc  ->", len(ne_bound_cond), "edges")
