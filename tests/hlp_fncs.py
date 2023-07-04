import numpy as np
import torch

def read_data_mat(input):
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

def check_inf_sites_list(data_inf, data_fin):
  count = 0
  for inf, fin in zip(data_inf, data_fin):
    if(check_inf_sites(inf)): count += 1
    if(not check_inf_sites(fin)): count += 1
  return count/(len(data_fin)*2)

def read_data_tens(input):

  data = []
  file = open(input, "r")
  content = file.read()
  matrices = content.split('\n\n')
  for m in matrices:
    rows = m.split('\n')
    matrix = [list(map(float, list(row))) for row in rows]    
    data.append(torch.Tensor(matrix))
  return data

def write_data(path, data):
  with open(path, "w") as file:
    for mat in data:
      for line in mat:
        strl = ''
        for num in line:
          strl += str(num)
        file.write(strl+"\n")
      file.write("\n")

def convert(bitlist):
  out = 0
  for bit in bitlist:
    out = (out << 1) | int(bit)
  return(out)

def sort_tensor(batched_tens):
  for i in range(batched_tens.size()[0]):
    int_tens = torch.Tensor([convert(batched_tens[i][:,j]) for j in range(batched_tens[i].size()[1])])
    int_tens, indices = torch.sort(int_tens, descending=False)
    batched_tens[i] = batched_tens[i][:, indices]

def sort_matrix(mat):
  int_cols = [convert(mat[:,j]) for j in range(np.shape(mat)[1])]
  indices = np.argsort(int_cols)
  mat = mat[:, indices]
  return mat

def noise_matrix(mat, alpha, beta):
  np.random.seed()
  noise_mat = mat.copy()
  for i in range(np.shape(mat)[0]):
    for j in range(np.shape(mat)[1]):
      if mat[i,j] == 0:
        if np.random.random() < alpha: noise_mat[i,j] = 1
      else:
        if np.random.random() < beta: noise_mat[i,j] = 0
  return noise_mat

def accuracy(output, target):
  return float(sum(torch.round(output) == target)/output.size()[0])