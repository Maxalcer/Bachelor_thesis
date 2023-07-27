from hlp_fncs import *
import random
import subprocess
import math
import numpy as np
from math import comb

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

def add_douplets(mat, n):
  while True:
    np.random.seed()
    matrix = mat.copy()
    douplets = generate_matrix(n)
    rows = np.random.choice(range(10), n, replace=False)
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

training_data_inf = read_data_mat("../data/no_noise/unsorted/train_inf.txt")
training_data_fin = read_data_mat("../data/no_noise/unsorted/train_fin.txt")

for i in range(len(training_data_fin)):
  if(i % 5000 == 0): print(i/5000, "%")
  np.random.seed()
  ran_n = np.random.choice(range(4), 1, replace=False)[0]
  if ran_n != 0:
    training_data_inf[i] = add_douplets(training_data_inf[i], ran_n)
  ran_n = np.random.choice(range(4), 1, replace=False)[0]
  if ran_n != 0:
    training_data_fin[i] = add_douplets(training_data_fin[i], ran_n)


write_data("../data/douplets/no_noise/unsorted/train_douplets_inf.txt", training_data_inf)
write_data("../data/douplets/no_noise/unsorted/train_douplets_fin.txt", training_data_fin)

testing_data_inf = read_data_mat("../data/no_noise/unsorted/test_inf.txt")
testing_data_fin = read_data_mat("../data/no_noise/unsorted/test_fin.txt")

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

write_data("../data/douplets/no_noise/unsorted/test_douplets_1_inf.txt", testing_data_inf_1)
write_data("../data/douplets/no_noise/unsorted/test_douplets_1_fin.txt", testing_data_fin_1)
write_data("../data/douplets/no_noise/unsorted/test_douplets_2_inf.txt", testing_data_inf_2)
write_data("../data/douplets/no_noise/unsorted/test_douplets_2_fin.txt", testing_data_fin_2)
write_data("../data/douplets/no_noise/unsorted/test_douplets_3_inf.txt", testing_data_inf_3)
write_data("../data/douplets/no_noise/unsorted/test_douplets_3_fin.txt", testing_data_fin_3)





