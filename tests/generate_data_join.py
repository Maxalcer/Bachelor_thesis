from hlp_fncs import *
import random
import subprocess
import math
import numpy as np
from math import comb

def generate_matrix(nrows):
  process = subprocess.Popen([r'./ms', str(nrows), '1', '-t', '5.0', '-s', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=r'/home/max/msdir')
  result = process.communicate()
  lines = result[0].split('\n')
  seeds = [int(i) for i in lines[1].split(' ')]
  del lines[0:7]
  del lines[len(lines)-1]
  matrix = [list(map(int, list(line))) for line in lines]
  return np.array(matrix), seeds

   
probs = range(1,6)

probs = [(1/math.exp(p)) for p in probs]

probs = [(p/np.sum(probs)) for p in probs]

orig_data = []

train_data = []

c = 0

# new_row = [a ^ b for a, b in zip(matrix[row], matrix[row+10])]

for i in range(1000):
  if(i % 5000 == 0): print(i/5000, "%")
  while True:    
    n = np.random.choice(range(1,6), 1, p=probs)
    n = n[0]
    temp_data = generate_matrix(10+n)
    temp_data += (n,)
    matrix = temp_data[0].copy()
    rows = np.random.choice(range(10), n, replace=False)
    j = 0
    for row in rows:
      new_row = [a | b for a, b in zip(matrix[row], matrix[10+j])]
      matrix[row] = new_row
      j += 1 
    matrix = np.delete(matrix, range(10,np.shape(matrix)[0]), 0)

    if (np.any(np.delete(temp_data[0], range(10,np.shape(temp_data[0])[0]), 0) != matrix)):
      train_data.append(matrix)
      orig_data.append(temp_data)
      break

write_data("../data/test_join.txt", train_data)

with open("../data/test_original_join.txt", "w") as file:
  for mat, seeds, n in orig_data:
    seedstr = [str(seed) for seed in seeds]
    file.write("seeds: "+ (' '.join(seedstr)) + "\n")
    file.write("number of duplets: "+ str(n) + "\n")
    for line in mat:
      strl = ''
      for num in line:
        strl += str(num)
      file.write(strl+"\n")
    file.write("\n")






