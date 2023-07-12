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
netzbig = torch.load('saved_fc_netz_12x12.py', map_location=torch.device('cpu'))

def test_var_noise():
  #netz.eval()
  netzfc.eval()
  netzbig.eval()
  #hits = 0
  hits_fc = 0
  hits_big = 0
  #hits_won = 0
  hits_algo = 0
  for (input, target), (input_big, target_big) in zip(testing_data, testing_data_big):
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    input_big = torch.tensor(noise_matrix(np.array(input_big[0]), a, b)).unsqueeze(0)
    input_big = sort_tensor(input_big)
    algo_erg = check_inf_sites(np.array(input[0]))
    if (algo_erg == int(target[0])): hits_algo += 1
    """
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    """
    # output = netz(expanded_input)
    output_fc = netzfc(input)
    output_big = netzbig(input_big)
    #output_won = netzwon(input)
    target = Variable(target)
    #if(torch.round(output) == target): hits += 1
    if(torch.round(output_fc) == target): hits_fc += 1
    if(torch.round(output_big) == target_big): hits_big += 1
    #if(torch.round(output_won) == target): hits_won += 1
  return (hits_big/len(testing_data)), (hits_fc/len(testing_data)), (hits_algo/len(testing_data))
  #return (hits_fc/len(testing_data))

#acc = []
acc_fc = []
acc_big = []
#acc_won = []
acc_algo = []
noise = []

b = 0

testing_data = get_train_dataset("../data/test_fin.txt", "../data/test_inf.txt", 1)
testing_data_big = get_train_dataset("../data/test_fin_12x12.txt", "../data/test_inf_12x12.txt", 1)
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  big, fc, algo = test_var_noise()
  #fc = test_var_noise()
  #acc.append(net)
  acc_fc.append(fc)
  acc_big.append(big)
  #acc_won.append(won)
  acc_algo.append(algo)
  noise.append(b*100)
  b = round(b + 0.001, 3)


#plt.plot(noise, acc, label = "CNN")
plt.plot(noise, acc_fc, label = "FCNN")
plt.plot(noise, acc_big, label = "FCNN 12x12")
#plt.plot(noise, acc_won, label = "FCNN without Noise")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise_12x12.png')
plt.show()

