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
netzfc = netzfc.cpu() 
netzfc.eval()

ig = IntegratedGradients(netzfc)
#occ = Occlusion(netzfc)

attr_ig = ig.attribute(data[4], target=0, n_steps=100)
#attr_occ = occ.attribute(data[0], target=0, sliding_window_shapes=(2,))
alg = np.zeros((10,10))
cols, rows = check_inf_sites_arg(data[4])
for col in cols:
  for row in rows:
    alg[row, col] = -1

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.imshow(np.array(attr_ig), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Network")

plt.subplot(1, 2, 2)
plt.imshow(alg, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Algorithm")

plt.savefig('../results/heatmap4.png')
plt.show()