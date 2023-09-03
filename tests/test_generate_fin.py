import sys
sys.path.append('../..')

from hlp_fncs import *
import random
import subprocess
import math
import numpy as np
from math import comb

# tests a different approach for generating the data with ISM violations

def generate_matrix(ncols):
  process = subprocess.Popen([r'./ms', '10', '1', '-t', '5.0', '-s', str(ncols)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=r'/buffer/ag_bsc/pmsb_23/max_alcer/bachelorarbeit/msdir')
  result = process.communicate()
  lines = result[0].split('\n')
  del lines[0:7]
  del lines[len(lines)-1]
  matrix = [list(map(int, list(line))) for line in lines]
  return np.array(matrix)

num = 0

for i in range(1000):
  c = 0
  matrix = generate_matrix(10)
  temp_matrix = matrix.copy()

  while(check_inf_sites(temp_matrix)):
    if(c > 100):
      matrix = generate_matrix(10)
      c = 0
    temp_matrix = matrix.copy()
    np.random.seed()
    col = np.random.choice(range(10), 1, replace=False)[0]
    row = np.random.choice(range(10), 1, replace=False)[0]
    if(temp_matrix[col, row] == 1): temp_matrix[col, row] = 0
    else: temp_matrix[col, row] = 1
    c += 1

  cols, rows = check_inf_sites_arg(temp_matrix)
  num += len(cols)

print(num/1000)


  








