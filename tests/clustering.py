import numpy as np
from scipy.cluster.hierarchy import average, leaves_list

def rotate(mat):
  return np.array(list(zip(*mat[::-1])))

def rotate_back(mat):
  return np.array(list(zip(*mat))[::-1])

def get_distance_matrix(mat):
  dist_matrix = np.zeros((np.shape(mat)[0], np.shape(mat)[0]))
  for i in range(np.shape(mat)[0]):
    for j in range(i, np.shape(mat)[0]):
      dist = 0
      for a, b in zip(mat[i], mat[j]):
        if a != b: dist += 1
      dist_matrix[i, j] = dist
  return dist_matrix

def sort_cluster(mat):
  dist = get_distance_matrix(mat)
  cluster = average(dist)
  leaves = leaves_list(cluster)
  mat = mat[leaves]
  mat = rotate(mat)
  dist = get_distance_matrix(mat)
  cluster = average(dist)
  leaves = leaves_list(cluster)
  mat = mat[leaves]
  return rotate_back(mat)

test = np.array([[1,0,1,1], [0,0,1,0], [1,1,0,0], [0,1,1,0]])
print(test, "\n")
print(sort_cluster(test))


