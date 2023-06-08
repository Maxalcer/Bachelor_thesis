import random
import subprocess
import math
import numpy as np
from math import comb

def read_data(input):

  data = []

  file = open(input, "r")

  content = file.read()

  matrices = content.split('\n\n')

  for m in matrices:
    rows = m.split('\n')
    matrix = [list(map(int, list(row))) for row in rows]    
    data.append(np.array(matrix))

  return data

def check_inf_sites(m):

  for i in range(np.shape(m)[1]):
    for j in range(i+1, np.shape(m)[1]):
      col1 = m[:,i]
      col2 = m[:,j]
      indices1 = [i for i, x in enumerate(col1) if x == 1]
      indices2 = [i for i, x in enumerate(col2) if x == 1]

      found_diff = ((len(list(set(indices1) - set(indices2))) != 0) & (len(list(set(indices2) - set(indices1))) != 0))
      found_same = (len(list(set(indices1) & set(indices2))) != 0)

      if found_diff & found_same: return False

  return True

def generate_matrix(ncols):
  command = r"./ms 10 1 -t 5.0 -s " + str(ncols)
  process = subprocess.Popen([r'./ms', '10', '1', '-t', '5.0', '-s', str(ncols)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=r'/home/max/msdir')
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

for i in range(500):
  n = np.random.choice(range(1,4), 1, p=probs)
  n = n[0]
  
  temp_data = generate_matrix(10+n)

  temp_data += (n,)

  matrix = temp_data[0][:]
  break_inf_loop = 0
    
  while(check_inf_sites(matrix)):
    matrix = temp_data[0]
    if break_inf_loop > comb(10, n):
      temp_data = generate_matrix(10+n)
      temp_data += (n,)
      matrix = temp_data[0][:]
      break_inf_loop = 0

    cols = np.random.choice(range(10), n, replace=False)
    j = 0
    for col in cols:
      new_col = [a | b for a, b in zip(matrix[:,col], matrix[:,10+j])]
      for k in range(10):
        matrix[k, col] = new_col[k]
      j += 1 
    matrix = np.delete(matrix, range(10,np.shape(matrix)[1]), 1)
    break_inf_loop += 1

  train_data.append(matrix)
  orig_data.append(temp_data)


with open("../data/test_fin.txt", "w") as file:
  for mat in train_data:
    for line in mat:
      strl = ''
      for num in line:
        strl += str(num)
      file.write(strl+"\n")
    file.write("\n")

with open("../data/test_original_fin.txt", "w") as file:
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






