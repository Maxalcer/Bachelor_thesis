import sys
sys.path.append('../')

import numpy as np
from hlp_fncs import *

# calculates the median number of ISM violations in matrices generated and not generated under the ISM with beta=15%

data_fin = read_data_mat("../data/noise/unsorted/test_fin_noise_15.txt")
data_inf = read_data_mat("../data/noise/unsorted/test_inf_noise_15.txt")

num_fin = 0
num_inf = 0
for i in range(len(data_fin)):
  col_fin, row_fin = check_inf_sites_arg(data_fin[i])
  col_inf, row_inf = check_inf_sites_arg(data_inf[i])
  num_fin += len(col_fin)
  num_inf += len(col_inf)

print("Violations FIN:", num_fin/len(data_fin))
print("Violations INF:", num_inf/len(data_inf))