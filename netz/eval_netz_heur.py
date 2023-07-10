from hlp_fncs import *
from get_dataset import *
from convol_netz import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import comb
import numpy as np

netzfc = torch.load('saved_fc_netz.py')

netzfc = netzfc.cpu() 


def test_var_noise():
  netzfc.eval()
  hits_fc = 0
  hits_algo = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b))
    input = torch.tensor(sort_matrix(np.array(input)))
    algo_erg = check_inf_sites(np.array(input))
    if (algo_erg == int(target[0])): hits_algo += 1
    loops = comb(input.size()[0], 3)*comb(input.size()[1], 2)
    output = 1
    for i in range(loops):
      rows = np.random.choice(range(12), 10, replace=False)
      cols = np.random.choice(range(12), 10, replace=False)
      np.sort(rows)
      np.sort(cols)
      input10 = input[rows,:][:,cols]
      output_fc = netzfc(input10)
      if(output_fc < 0.2): 
        output = 0
        break
    target = Variable(target)
    if(output == target): hits_fc += 1
  return (hits_fc/len(testing_data)), (hits_algo/len(testing_data))

acc_fc = []
acc_algo = []
noise = []

b = 0

testing_data = get_train_dataset("../data/test_fin_12x12.txt", "../data/test_inf_12x12.txt", 1)
while (b < 0.51):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  fc, algo = test_var_noise()
  #fc = test_var_noise()
  acc_fc.append(fc)
  acc_algo.append(algo)
  noise.append(b*100)
  b += 0.01

plt.plot(noise, acc_fc, label = "FCNN")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise_12x12.png')
plt.show()

