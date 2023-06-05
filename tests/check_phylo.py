import numpy as np
import string
import sys
sys.path.insert(1, '../perfect_phylogeny/')
from perfect_phylogeny import phylogeny as phyl

def check_phy(mat):
  # rotate M for convenient iteration
  m_prime = np.rot90(mat)

  # keep only unique combinations
  m_prime = np.unique(list(map(lambda x: '.'.join(map(str, x)), m_prime)))
  m_prime = np.array(list(map(lambda x: list(map(int, x.split('.'))), m_prime)))

  # count binary score of columns
  binary_strings = []
  for col in m_prime:
    col_string = '0b'+''.join(map(str,col))
    binary_strings.append(int(col_string,2))

  # sort by binary score
  order = np.argsort(binary_strings)[::-1]
  m_prime = m_prime[order] 

  m_prime = np.rot90(m_prime[:,:])[::-1] #rotate again

  ncol = len(m_prime[0])

  k = np.empty( [0,ncol], dtype='|S15' )
  features = np.array(list(string.ascii_lowercase[:ncol]))

  for m in m_prime:
    row_feats = features[m!=0] #features in the row
    mrow = np.zeros(ncol,dtype='|S15')
    mrow.fill('0')
    for idx,feature in enumerate(row_feats):
      mrow[idx] = feature
    n_feat = len(row_feats)    
    if n_feat < ncol: 
      mrow[n_feat]='#'
    k = np.append(k,[mrow],axis=0)

  locations = []
  for feature in features:
    present_at = set([])
    for k_i in k:
      [ present_at.add(loc_list) for loc_list in list(np.where(k_i==feature)[0]) ]
    locations.append(present_at)

  loc_test = np.array([len(loc_list)>1 for loc_list in locations])

  return not np.any(loc_test)

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

data_fin = read_data("train_fin.txt")

count = 0

s = np.array(['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10'])
c = np.array(['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'])

for mat in data_fin:
  # m_new, c_prime, k1, tree = phyl.solve_incomplete_phylogeny(mat,s,c)
  k1 = phyl.get_k1_matrix(mat,c)
  if(phyl.perfect_phylogeny_exists(k1,c)): count += 1

print(count)

data_inf = read_data("train_inf.txt")

count = 0

for mat in data_inf:
  # m_new, c_prime, k1, tree = phyl.solve_incomplete_phylogeny(mat,s,c)
  k1 = phyl.get_k1_matrix(mat,c)
  if(phyl.perfect_phylogeny_exists(k1,c)): count += 1

print(count)