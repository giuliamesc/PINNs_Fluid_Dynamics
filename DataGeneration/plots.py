# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:10:04 2021

@author: giuli
"""

import numpy as np
import matplotlib.pyplot as plt

vec = np.load("bpoints.npy")

plt.scatter(vec[:,0],vec[:,1],c=vec[:,3])