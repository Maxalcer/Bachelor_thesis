import sys
sys.path.append('../networks/net_implementations')
sys.path.append('../')

import numpy as np
from hlp_fncs import *
from get_dataset import *
from fc_netz import *

# calculates the median number of ISM violations for matrices that get falsy classified by the network

netzfc = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))
data = get_train_dataset("../data/noise/sorted/test_fin_noise_15_sorted.txt", "../data/noise/sorted/test_inf_noise_15_sorted.txt", 1)
f = 0
fn = 0
fp = 0
num_f = 0
num_fn = 0
num_fp = 0
for input, target in data:
  output = netzfc(input)
  if(torch.round(output) != target):
    f += 1
    col, row = check_inf_sites_arg(np.array(input[0]))
    num_f += len(col)  

print("Violations:", num_f/f)
