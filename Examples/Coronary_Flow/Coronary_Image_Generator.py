# %% Import libraries and working directory settings
import os
cwd = os.path.abspath(os.getcwd())
os.chdir("../")
os.chdir("../")
os.chdir("../")
os.chdir("nisaba")
import nisaba as ns
from nisaba.experimental.physics import tens_style as operator
os.chdir(cwd)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from tensorflow.math import multiply as product

problem_name = "Coronary_Flow"

#  Set seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

L = 40 
H = 10
eps = 0.3
c = 0.2
max_blob = (H/3)*(2/c)
r_c = np.sqrt(c)
xx_num = 1000
xx = tf.linspace(start = -10,  stop = 10, num = xx_num + 1)
blob = max_blob*c/(2*np.cosh(xx*r_c/2)*np.cosh(xx*r_c/2))
print(blob)

fig = plt.figure(1, figsize = (12, 3))
ax = fig.add_subplot()
ax.axis([-L/2, L/2, 0 - eps, H + eps])
ax.axis("equal")
plt.axvline(-L/2, 0, H, c = "r")
plt.axvline( L/2, 0, H, c = "r")
plt.axhline(0, -L/2, L/2, c = "r")
plt.axhline(H, -L/2, L/2, c = "r")
ax.plot(xx,blob, "k")
