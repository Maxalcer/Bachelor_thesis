from hlp_fncs import *
import numpy as np
import string
import subprocess

data_inf = read_data_mat("../data/train_inf.txt")

count = 0

for mat in data_inf:
  if(check_inf_sites(mat)): count += 1

print(count)

data_fin = read_data_mat("../data/train_fin.txt")

count = 0

for mat in data_fin:
  if(check_inf_sites(mat)): count += 1

print(count)