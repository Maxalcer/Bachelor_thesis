from hlp_fncs import *
from get_dataset import *
from convol_netz import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

#netz = torch.load('saved_conv_netz.py')
netzfc = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))
# netzwon = torch.load('saved_fc_netz_won.py')
netzuns = torch.load('saved_fc_netz_unsorted.py', map_location=torch.device('cpu'))

def test_var_noise():
  #netz.eval()
  netzfc.eval()
  netzuns.eval()
  #hits = 0
  hits_fc = 0
  hits_uns = 0
  #hits_won = 0
  hits_algo = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input_sort = sort_tensor(input)
    algo_erg = check_inf_sites(np.array(input[0]))
    if (algo_erg == int(target[0])): hits_algo += 1
    """
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    """
    # output = netz(expanded_input)
    output_fc = netzfc(input_sort)
    output_uns = netzuns(input)
    #output_won = netzwon(input)
    target = Variable(target)
    #if(torch.round(output) == target): hits += 1
    if(torch.round(output_fc) == target): hits_fc += 1
    if(torch.round(output_uns) == target): hits_uns += 1
    #if(torch.round(output_won) == target): hits_won += 1
  return (hits_uns/len(testing_data)), (hits_fc/len(testing_data)), (hits_algo/len(testing_data))
  #return (hits_fc/len(testing_data))

#acc = []
acc_fc = []
acc_uns = []
#acc_won = []
acc_algo = []
noise = []

b = 0

testing_data = get_train_dataset("../data/test_fin.txt", "../data/test_inf.txt", 1)
#testing_data_big = get_train_dataset("../data/test_fin_12x12.txt", "../data/test_inf_12x12.txt", 1)
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  uns, fc, algo = test_var_noise()
  #fc = test_var_noise()
  #acc.append(net)
  acc_fc.append(round(fc, 4))
  acc_uns.append(round(uns, 4))
  #acc_won.append(won)
  acc_algo.append(round(algo, 4))
  noise.append(b*100)
  b = round(b + 0.005, 3)


#plt.plot(noise, acc, label = "CNN")
plt.plot(noise, acc_uns, label = "FCNN")
plt.plot(noise, acc_fc, label = "FCNN sorted")
#plt.plot(noise, acc_won, label = "FCNN without Noise")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise_unsorted.png')
plt.show()

