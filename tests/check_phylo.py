from hlp_fncs import *
import numpy as np
import string
import subprocess

data_inf = read_data_mat("../data/test_fin.txt")

count_inf = 0

for mat in data_inf:
  if(check_inf_sites(sort_cluster(mat))): count_inf += 1

print(count_inf)


