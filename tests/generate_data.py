import random
import math
import numpy as np

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

data_pre = read_data("train_fin_pre.txt")

probs = range(1,11)

probs = [(1/math.exp(p)) for p in probs]

probs = [(p/np.sum(probs)) for p in probs]

data = []

c = 0

for matrix in data_pre:
  new_matrix = matrix

  n = np.random.choice(range(1,11), 1, p=probs)
  n = n[0]
  rows = np.random.choice(range(10), n, replace=False)

  for row in rows:
    r = np.random.choice(range(2), 1)
    if(r == 0):
      new_row = [a | b for a, b in zip(matrix[row], matrix[row+10])]
    else:
      new_row = [a ^ b for a, b in zip(matrix[row], matrix[row+10])]

    for i in range(10):
      new_matrix[row, i] = new_row[i]
    
  new_matrix = np.delete(new_matrix, range(10,20), 0)
  data.append(new_matrix)


with open("train_fin.txt", "w") as file:
  for matrix in data:
    for line in matrix:
      strl = ''
      for num in line:
        strl += str(num)
      file.write(strl+"\n")
    file.write("\n")






