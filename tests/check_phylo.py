import numpy as np
import string
import subprocess

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


data_inf = read_data("../data/train_inf.txt")

count = 0

for mat in data_inf:
  if(check_inf_sites(mat)): count += 1

print(count)

data_fin = read_data("../data/train_fin.txt")

count = 0

for mat in data_fin:
  if(check_inf_sites(mat)): count += 1

print(count)