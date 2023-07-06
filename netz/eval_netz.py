from hlp_fncs import *
from get_dataset import *
from convol_netz import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

netz = torch.load('saved_netz.py')
netzfc = torch.load('saved_fc_netz.py')
# netzwon = torch.load('saved_fc_netz_won.py')
# netzfc = torch.load('saved_fc_netz_join.py')

netz = netz.cpu()
netzfc = netzfc.cpu() 
#netzwon = netzwon.cpu()

def test_var_noise():
  netz.eval()
  netzfc.eval()
  hits = 0
  hits_fc = 0
  #hits_won = 0
  hits_algo = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b))
    input = torch.tensor(sort_cluster(np.array(input[0])))
    algo_erg = check_inf_sites(np.array(input))
    if (algo_erg == int(target[0])): hits_algo += 1
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    output_fc = netzfc(input)
    #output_won = netzwon(input)
    target = Variable(target)
    if(torch.round(output) == target): hits += 1
    if(torch.round(output_fc) == target): hits_fc += 1
    #if(torch.round(output_won) == target): hits_won += 1
  return (hits/len(testing_data)), (hits_fc/len(testing_data)), (hits_algo/len(testing_data))
  #return (hits_fc/len(testing_data))

acc = []
acc_fc = []
#acc_won = []
acc_algo = []
noise = []

b = 0

testing_data = get_train_dataset("../data/test_fin.txt", "../data/test_inf.txt", 1)
while (b < 0.51):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  net, fc, algo = test_var_noise()
  #fc = test_var_noise()
  acc.append(net)
  acc_fc.append(fc)
  #acc_won.append(won)
  acc_algo.append(algo)
  noise.append(b*100)
  b += 0.005


plt.plot(noise, acc, label = "CNN")
plt.plot(noise, acc_fc, label = "FCNN")
#plt.plot(noise, acc_won, label = "FCNN without Noise")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise_Cluster.png')
plt.show()

