from hlp_fncs import *
from fc_netz import *
import torch
import numpy as np
from captum.attr import IntegratedGradients, Occlusion
import matplotlib.pyplot as plt

def check_inf_sites_arg(m):
  for i in range(np.shape(m)[1]):
    for j in range(i+1, np.shape(m)[1]):
      col1 = m[:,i]
      col2 = m[:,j]
      indices1 = [i for i, x in enumerate(col1) if x == 1]
      indices2 = [i for i, x in enumerate(col2) if x == 1]

      found_diff = ((len(list(set(indices1) - set(indices2))) != 0) & (len(list(set(indices2) - set(indices1))) != 0))
      found_same = (len(list(set(indices1) & set(indices2))) != 0)

      if found_diff & found_same: return [i, j], (list(set(indices1) - set(indices2)) + list(set(indices2) - set(indices1)) + list(set(indices1) & set(indices2)))
  return []

netzfc = torch.load('saved_fc_netz.py')
data = read_data_tens("../data/test_fin_sorted.txt")
data_noise = read_data_tens("../data/test_fin_noise_15_sorted.txt")
netzfc = netzfc.cpu() 
netzfc.eval()
print(netzfc(data[2]))
print(netzfc(data_noise[2]))
ig = IntegratedGradients(netzfc)
occ = Occlusion(netzfc)
mat = data[2].unsqueeze(0)
mat_noise = data_noise[2].unsqueeze(0)
#attr_ig = ig.attribute(data[4], target=0, n_steps=100)
attr_occ = occ.attribute(mat, sliding_window_shapes=(1,1))
attr_occ_noise = occ.attribute(mat_noise, sliding_window_shapes=(1,1))
alg = np.zeros((10,10))
cols, rows = check_inf_sites_arg(mat[0])
for col in cols:
  for row in rows:
    alg[row, col] = -1

plt.figure(figsize=(20,5))
plt.subplot(1, 3, 1)
plt.imshow(np.array(attr_occ[0]), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Network")

plt.subplot(1, 3, 2)
plt.imshow(np.array(attr_occ_noise[0]), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Network Noise")

plt.subplot(1, 3, 3)
plt.imshow(alg, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Algorithm")

plt.savefig('../results/heatmap_noise.png')
plt.show()