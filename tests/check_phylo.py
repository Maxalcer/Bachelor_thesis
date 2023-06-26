from hlp_fncs import *
import numpy as np
import string
import subprocess

data_inf = read_data_mat("../data/test_inf_noise_5.txt")

count_inf = 0

for mat in data_inf:
  if(check_inf_sites(mat)): count_inf += 1


data_fin = read_data_mat("../data/test_fin_noise_5.txt")

count_fin = 0

for mat in data_fin:
  if(not check_inf_sites(mat)): count_fin += 1

print("5%:", (count_fin+count_inf)/2000)

data_inf = read_data_mat("../data/test_inf_noise_15.txt")

count_inf = 0

for mat in data_inf:
  if(check_inf_sites(mat)): count_inf += 1

data_fin = read_data_mat("../data/test_fin_noise_15.txt")

count_fin = 0

for mat in data_fin:
  if(not check_inf_sites(mat)): count_fin += 1

print("15%:", (count_fin+count_inf)/2000)

data_inf = read_data_mat("../data/test_inf_noise_25.txt")

count_inf = 0

for mat in data_inf:
  if(check_inf_sites(mat)): count_inf += 1


data_fin = read_data_mat("../data/test_fin_noise_25.txt")

count_fin = 0

for mat in data_fin:
  if(not check_inf_sites(mat)): count_fin += 1

print("25%:", (count_fin+count_inf)/2000)

