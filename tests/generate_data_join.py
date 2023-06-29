from hlp_fncs import *
import random
import subprocess
import math
import numpy as np
from math import comb

def generate_matrix():
  process = subprocess.Popen([r'./ms', '10', '1', '-t', '5.0', '-s', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=r'/home/max/msdir')
  result = process.communicate()
  lines = result[0].split('\n')
  seeds = [int(i) for i in lines[1].split(' ')]
  del lines[0:7]
  del lines[len(lines)-1]
  matrix = [list(map(int, list(line))) for line in lines]
  return np.array(matrix), seeds
  
orig_data1 = []
orig_data2 = []
train_data = []

for i in range(1000):
  if(i % 5000 == 0): print(i/5000, "%")
  
  temp_data1 = generate_matrix()
  temp_data2 = generate_matrix()
  matrix1 = temp_data1[0].copy()
  matrix2 = temp_data2[0].copy()

  for i in range(np.shape(matrix1)[0]):
    for j in range(np.shape(matrix1)[1]):
      matrix1[i,j] = matrix1[i,j] | matrix2[i,j]

  train_data.append(matrix1)
  orig_data1.append(temp_data1)
  orig_data2.append(temp_data2)

write_data("../data/test_join.txt", train_data)

with open("../data/test_original_join1.txt", "w") as file:
  for mat, seeds in orig_data1:
    seedstr = [str(seed) for seed in seeds]
    file.write("seeds: "+ (' '.join(seedstr)) + "\n")
    for line in mat:
      strl = ''
      for num in line:
        strl += str(num)
      file.write(strl+"\n")
    file.write("\n")

with open("../data/test_original_join2.txt", "w") as file:
  for mat, seeds in orig_data2:
    seedstr = [str(seed) for seed in seeds]
    file.write("seeds: "+ (' '.join(seedstr)) + "\n")
    for line in mat:
      strl = ''
      for num in line:
        strl += str(num)
      file.write(strl+"\n")
    file.write("\n")




