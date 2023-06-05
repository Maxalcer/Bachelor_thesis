import numpy as np
import string

m = np.array([[0,1,1,0],
              [0,1,0,1],
              [0,0,0,0],
              [0,0,0,1]])

# rotate M for convenient iteration
m_prime = np.rot90(m)

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
if np.any(loc_test):
    print('No phylogeny found!')
else:    
    print('Success! Found phylogeny!')