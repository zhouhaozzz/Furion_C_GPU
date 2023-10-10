import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import os
import sys
import time
from matplotlib.font_manager import FontProperties
import math
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rcParams['mathtext.fontset']='stix'
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
plt.rc('font', family="Times New Roman")

folder_path = os.getcwd()
file_prefix = folder_path + "/data/Furion_plot_sigma_"

file_count = 100
try:
    file_count = int(sys.argv[1])
except ValueError:
    print("Invalid integer provided")
    sys.exit(1)

X=[]
Y=[]
Phi=[]
Psi=[]

for i in range(1, file_count + 1):
    file_name = f"{file_prefix}{i}.dat"
    
    try:
        data = np.loadtxt("%s"%file_name)
        X = np.append(X,data[0])
        Y = np.append(Y,data[1])
        Phi = np.append(Phi,data[2])
        Psi = np.append(Psi,data[3])
        print(f"Read {file_name} successfully")

    except FileNotFoundError:
        break
        print(f"file {file_name} does not exist")
    except Exception as e:
        print(f"Error while processing {file_name} : {str(e)}")
print(folder_path)
PHI0 = Phi
PSI0 = np.arctan(np.tan(Psi) / np.cos(Phi))
X01 = X
Y01 = Y
X1 = X01*1e6
Y1 = Y01*1e6
PHI = PHI0*1e6 
PSI = PSI0*1e6
x_Std = np.std(X1)
y_Std = np.std(Y1)
phi_Std = np.std(PHI)
psi_Std = np.std(PSI)
x_Mean = np.mean(X1)
y_Mean = np.mean(Y1)
phi_Mean = np.mean(PHI)
psi_Mean = np.mean(PSI)
x_Scale = [x_Mean-6*x_Std, x_Mean+6*x_Std]
y_Scale = [y_Mean-6*y_Std, y_Mean+6*y_Std]
phi_Scale = [phi_Mean-6*phi_Std, phi_Mean+6*phi_Std]
psi_Scale = [psi_Mean-6*psi_Std, psi_Mean+6*psi_Std]
x_Tick = [x_Mean-6*x_Std,   x_Mean-3*x_Std,  x_Mean,  x_Mean+3*x_Std,   x_Mean+6*x_Std]
y_Tick = [y_Mean-6*y_Std,   y_Mean-3*y_Std,  y_Mean,  y_Mean+3*y_Std,   y_Mean+6*y_Std]
phi_Tick = [phi_Mean-6*phi_Std,   phi_Mean-3*phi_Std,  phi_Mean,  phi_Mean+3*phi_Std,  phi_Mean+6*phi_Std]
psi_Tick = [psi_Mean-6*psi_Std,   psi_Mean-3*psi_Std,  psi_Mean,  psi_Mean+3*psi_Std,  psi_Mean+6*psi_Std]

import matplotlib.pyplot as plt

# Creating the Figure Window
figure1 = plt.figure(figsize=(48/2.54, 20/2.54),dpi=100)
plt.axis('off')

# Create the axes axes1
axes1 = figure1.add_axes([0.1, 0.590660465116279, 0.24, 0.334339534883721])
plt.scatter(X1, Y1, s=10, marker='.', color='blue')
plt.xlim(x_Scale)
plt.ylim(y_Scale)
plt.xlabel(r'$X [\mu m]$', fontsize=18, visible=True)
plt.ylabel(r'$Y [\mu m]$', fontsize=18, visible=True)
plt.box(True)
plt.xticks(x_Tick)
plt.yticks(y_Tick)

# Create the axes axes2
axes2 = figure1.add_axes([0.4, 0.590660465116279, 0.24, 0.334339534883721])
plt.scatter(PHI, PSI, s=10, marker='.', color='blue')
plt.xlim(phi_Scale)
plt.ylim(psi_Scale)
plt.xlabel(r'$\Phi [urad]$', fontsize=22, visible=True)
plt.ylabel(r'$\Psi [urad]$', fontsize=22, visible=True)
plt.box(True)
plt.xticks(phi_Tick)
plt.yticks(psi_Tick)

# Create the axes axes3
axes3 = figure1.add_axes([0.7, 0.590660465116279, 0.24, 0.334339534883721])
plt.scatter(X1, PHI, s=10, marker='.', color='blue')
plt.xlim(x_Scale)
plt.ylim(phi_Scale)
plt.xlabel(r'$X [\mu m]$', fontsize=22, visible=True)
plt.ylabel(r'$\Phi [urad]$', fontsize=22, visible=True)
plt.box(True)
plt.xticks(x_Tick)
plt.yticks(phi_Tick)

# Create the axes axes4
axes4 = figure1.add_axes([0.1, 0.116823255813953, 0.24, 0.334339534883721])
plt.scatter(Y1, PSI, s=10, marker='.', color='blue')
plt.xlim(y_Scale)
plt.ylim(psi_Scale)
plt.xlabel(r'$Y [\mu m]$', fontsize=22, visible=True)
plt.ylabel(r'$\Psi [urad]$', fontsize=22, visible=True)
plt.box(True)
plt.xticks(y_Tick)
plt.yticks(psi_Tick)

# Create the axes axes5
axes5 = figure1.add_axes([0.4, 0.116823255813953, 0.24, 0.334339534883721])
plt.scatter(X1, PSI, s=10, marker='.', color='blue')
plt.xlim(x_Scale)
plt.ylim(psi_Scale)
plt.xlabel(r'$X [\mu m]$', fontsize=22, visible=True)
plt.ylabel(r'$\Psi [urad]$', fontsize=22, visible=True)
plt.box(True)
plt.xticks(x_Tick)
plt.yticks(psi_Tick)

# Create the axes axes6
axes6 = figure1.add_axes([0.7, 0.116823255813953, 0.24, 0.334339534883721])
plt.scatter(Y1, PHI, s=10, marker='.', color='blue')
plt.xlim(y_Scale)
plt.ylim(phi_Scale)
plt.xlabel(r'$Y [\mu m]$', fontsize=22, visible=True)
plt.ylabel(r'$\Phi [urad]$', fontsize=22, visible=True)
plt.box(True)
plt.xticks(y_Tick)
plt.yticks(phi_Tick)

plt.savefig("Furion_plot4_6sigma.png",bbox_inches="tight")

plt.show()
