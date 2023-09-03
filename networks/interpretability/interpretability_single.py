import sys
sys.path.append('../net_implementations')
sys.path.append('../..')

from hlp_fncs import *
from fc_netz import *
import torch
import numpy as np
from captum.attr import Occlusion
import matplotlib.pyplot as plt

# Creates a single heatmap which visualizes the results of the occlusion algorithm next to a matix with marked entries which belong to a ISM violation 

netzfc = torch.load('../saved_nets/saved_fc_netz.py', map_location=torch.device('cpu'))
data = read_data_tens("../../data/10x10/noise/sorted/test_inf_noise_15_sorted.txt")
netzfc.eval()

occ = Occlusion(netzfc)

mat = data[9].unsqueeze(0)
# Occlusion
class_net = int(torch.round(netzfc(mat))[0])
attr_occ = occ.attribute(mat, sliding_window_shapes=(1,1))
class_alg = int(check_inf_sites(mat[0]))

# ISM violations
alg = np.zeros((10,10))
cols, rows = check_inf_sites_arg(mat[0])
for j in range(len(cols)):
  for col in cols[j]:
    for row in rows[j]:
      alg[row, col] = -1

# Heatmaps
fig, axs = plt.subplots(nrows=1, ncols=2, figsize =(12, 5))

heatmap = axs[0].imshow(np.array(attr_occ[0]), cmap='RdGy', interpolation='nearest', vmin = -1, vmax = 1)
cbar = plt.colorbar(heatmap, ax=axs[0])
cbar.ax.tick_params(labelsize=12)
axs[0].set_title("A", fontsize = 12)
axs[0].tick_params(axis='both', labelsize=12)
heatmap2 = axs[1].imshow(alg, cmap='RdGy', interpolation='nearest', vmin = -1, vmax = 1)
axs[1].set_title("B", fontsize = 12)
axs[1].tick_params(axis='both', labelsize=12)
mat = data[9]
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        text_color1 = 'white' if np.array(attr_occ[0])[i, j] < (-0.8) else 'black'
        text_color2 = 'white' if alg[i, j] == (-1) else 'black'
        text1 = axs[0].text(j, i, int(mat[i,j]), ha='center', va='center', color='black', fontsize = 12)
        text2 = axs[1].text(j, i, int(mat[i,j]), ha='center', va='center', color=text_color2, fontsize = 12)


plt.tight_layout()
plt.savefig('../../results/explainability/heatmap_single_inf.png')
plt.show()