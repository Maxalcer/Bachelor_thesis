import sys
sys.path.append('../')

from hlp_fncs import *
import subprocess
import math
import numpy as np
from math import comb

# Scribt for adding doublets to generated data

# generate a matrix with a variable number of rows under the ISM using MS
def generate_matrix(nrows):
  ex = 0
  if nrows == 1: 
    ex = 1
    nrows = 2
  process = subprocess.Popen([r'./ms', str(nrows), '1', '-t', '5.0', '-s', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=r'/buffer/ag_bsc/pmsb_23/max_alcer/bachelorarbeit/msdir')
  result = process.communicate()
  lines = result[0].split('\n')
  #seeds = [int(i) for i in lines[1].split(' ')]
  del lines[0:7]
  del lines[len(lines)-1]
  matrix = [list(map(int, list(line))) for line in lines]
  matrix = np.array(matrix)
  if ex == 1:
    matrix = np.delete(matrix, (1), axis=0)
  return matrix #, seeds

# Adding a variable number of doublets to a matrix
def add_douplets(mat, n):
  while True:
    np.random.seed()
    matrix = mat.copy()
    douplets = generate_matrix(n)
    rows = np.random.choice(range(50), n, replace=False)
    j = 0
    for row in rows:
      new_row = [a | b for a, b in zip(matrix[row], douplets[j])]
      matrix[row] = new_row
      j += 1 

    if (np.any(mat != matrix)):
      return matrix

probs = range(4)

probs = [(1/math.exp(p)) for p in probs]

probs = [(p/np.sum(probs)) for p in probs]

training_data_inf = read_data_mat("../data/10x50/no_noise/train_inf_10x50.txt")
training_data_fin = read_data_mat("../data/10x50/no_noise/train_fin_10x50.txt")

for i in range(len(training_data_fin)):
  if(i % 5000 == 0): print(i/5000, "%")
  np.random.seed()
  ran_n = np.random.choice(range(4), 1, replace=False)[0]
  if ran_n != 0:
    training_data_inf[i] = add_douplets(training_data_inf[i], ran_n)
  ran_n = np.random.choice(range(4), 1, replace=False)[0]
  if ran_n != 0:
    training_data_fin[i] = add_douplets(training_data_fin[i], ran_n)


write_data("../data/10x50/douplets/no_noise/train_inf_10x50_douplets.txt", training_data_inf)
write_data("../data/10x50/douplets/no_noise/train_fin_10x50_douplets.txt", training_data_fin)

testing_data_inf = read_data_mat("../data/10x50/no_noise/test_inf_10x50.txt")
testing_data_fin = read_data_mat("../data/10x50/no_noise/test_fin_10x50.txt")

testing_data_inf_1 = []
testing_data_inf_2 = []
testing_data_inf_3 = []
testing_data_fin_1 = []
testing_data_fin_2 = []
testing_data_fin_3 = []

for i in range(len(testing_data_inf)):
  testing_data_inf_1.append(add_douplets(testing_data_inf[i], 1))
  testing_data_inf_2.append(add_douplets(testing_data_inf[i], 2))
  testing_data_inf_3.append(add_douplets(testing_data_inf[i], 3))
  testing_data_fin_1.append(add_douplets(testing_data_fin[i], 1))
  testing_data_fin_2.append(add_douplets(testing_data_fin[i], 2))
  testing_data_fin_3.append(add_douplets(testing_data_fin[i], 3))

write_data("../data/10x50/douplets/no_noise/test_inf_10x50_douplets_1.txt", testing_data_inf_1)
write_data("../data/10x50/douplets/no_noise/test_fin_10x50_douplets_1.txt", testing_data_fin_1)
write_data("../data/10x50/douplets/no_noise/test_inf_10x50_douplets_2.txt", testing_data_inf_2)
write_data("../data/10x50/douplets/no_noise/test_fin_10x50_douplets_2.txt", testing_data_fin_2)
write_data("../data/10x50/douplets/no_noise/test_inf_10x50_douplets_3.txt", testing_data_inf_3)
write_data("../data/10x50/douplets/no_noise/test_fin_10x50_douplets_3.txt", testing_data_fin_3)





