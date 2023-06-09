from hlp_fncs import *
import random
import subprocess
import math
import numpy as np
from math import comb

def generate_matrix(ncols):
  process = subprocess.Popen([r'./ms', '12', '1', '-t', '5.0', '-s', str(ncols)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=r'/home/max/msdir')
  result = process.communicate()
  lines = result[0].split('\n')
  seeds = [int(i) for i in lines[1].split(' ')]
  del lines[0:7]
  del lines[len(lines)-1]
  matrix = [list(map(int, list(line))) for line in lines]
  return np.array(matrix), seeds

   
probs = range(1,4)

probs = [(1/math.exp(p)) for p in probs]

probs = [(p/np.sum(probs)) for p in probs]

orig_data = []

train_data = []

c = 0

# new_row = [a ^ b for a, b in zip(matrix[row], matrix[row+10])]

for i in range(500000):
  if(i % 5000 == 0): print(i/5000, "%")
  n = np.random.choice(range(1,4), 1, p=probs)
  n = n[0]
  
  temp_data = generate_matrix(12+n)

  temp_data += (n,)

  matrix = temp_data[0][:]
  break_inf_loop = 0
    
  while(check_inf_sites(matrix)):
    matrix = temp_data[0]
    if break_inf_loop > comb(12, n):
      temp_data = generate_matrix(12+n)
      temp_data += (n,)
      matrix = temp_data[0][:]
      break_inf_loop = 0

    cols = np.random.choice(range(12), n, replace=False)
    j = 0
    for col in cols:
      new_col = [a | b for a, b in zip(matrix[:,col], matrix[:,12+j])]
      for k in range(12):
        matrix[k, col] = new_col[k]
      j += 1 
    matrix = np.delete(matrix, range(12,np.shape(matrix)[1]), 1)
    break_inf_loop += 1

  train_data.append(matrix)
  orig_data.append(temp_data)

write_data("../data/train_fin_12x12.txt", train_data)

with open("../data/train_original_fin_12x12.txt", "w") as file:
  for mat, seeds, n in orig_data:
    seedstr = [str(seed) for seed in seeds]
    file.write("seeds: "+ (' '.join(seedstr)) + "\n")
    file.write("added lines: "+ str(n) + "\n")
    for line in mat:
      strl = ''
      for num in line:
        strl += str(num)
      file.write(strl+"\n")
    file.write("\n")






