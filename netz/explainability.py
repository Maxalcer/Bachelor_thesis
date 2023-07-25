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
  return list(), list()

netzfc = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))
#data = read_data_tens("../data/test_fin_sorted.txt")
data = read_data_tens("../data/test_inf_noise_15_sorted.txt")
netzfc.eval()
#ig = IntegratedGradients(netzfc)
occ = Occlusion(netzfc)
class_net = []
attr_occ = []
class_alg = []
alg = []
i = 0
while i < 10:
  mat = data[i+40].unsqueeze(0)
  class_net.append(int(torch.round(netzfc(mat))[0]))
  attr_occ.append(occ.attribute(mat, sliding_window_shapes=(1,1)))
  class_alg.append(check_inf_sites(mat[0]))
  a = np.zeros((10,10))
  cols, rows = check_inf_sites_arg(mat[0])
  for col in cols:
    for row in rows:
      a[row, col] = -1
  alg.append(a)
  i += 1

heatmaps = []
cbars = []
fig, axs = plt.subplots(nrows=5, ncols=4, figsize =(60, 50))

inx = 0
for k in range(5):
  for l in range(2):
    heatmaps.append(axs[k, l*2].imshow(np.array(attr_occ[inx][0]), cmap='summer', interpolation='nearest'))
    cbars.append(plt.colorbar(heatmaps[-1], ax=axs[k, l*2]))
    axs[k, l*2].set_title(str(class_net[inx]))
    heatmaps.append(axs[k, l*2+1].imshow(alg[inx], cmap='gray', interpolation='nearest'))
    cbars.append(plt.colorbar(heatmaps[-1], ax=axs[k, l*2+1]))
    axs[k, l*2+1].set_title(str(class_alg[inx]))
    mat = data[inx+40]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            #text_color1 = 'white' if np.array(attr_occ[inx][0])[i, j] < (-0.6) else 'black'
            text_color2 = 'white' if alg[inx][i, j] == (-1) else 'black'
            text1 = axs[k, l*2].text(j, i, int(mat[i,j]), ha='center', va='center', color='black')
            text2 = axs[k, l*2+1].text(j, i, int(mat[i,j]), ha='center', va='center', color=text_color2)
    inx += 1


plt.tight_layout()
plt.savefig('../results/heatmaps5_inf.png')
plt.show()