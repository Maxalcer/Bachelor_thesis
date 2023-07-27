from hlp_fncs import *
from get_dataset import *
from convol_netz import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

netzcnn = torch.load('saved_conv_netz.py', map_location=torch.device('cpu'))
netzfc = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))
# netzwon = torch.load('saved_fc_netz_won.py')
# netzuns = torch.load('saved_fc_netz_unsorted.py', map_location=torch.device('cpu'))

def classify(input, cutoff):
  if input > cutoff: return 1
  else: return 0

def test_var_noise():
  netzfc.eval()
  netzcnn.eval()
  hits_fc = 0
  hits_cnn = 0
  hits_algo = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    algo_erg = check_inf_sites(np.array(input[0]))
    if (algo_erg == int(target[0])): hits_algo += 1
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)

    output_fc = netzfc(input)
    output_cnn = netzcnn(expanded_input)
    #output_won = netzwon(input)
    target = Variable(target)
    #if(torch.round(output) == target): hits += 1
    if(classify(output_fc, 0.5) == target): hits_fc += 1
    if(classify(output_cnn, 0.5) == target): hits_cnn += 1
    #if(torch.round(output_won) == target): hits_won += 1
  return (hits_cnn/len(testing_data)), (hits_fc/len(testing_data)), (hits_algo/len(testing_data))
  #return (hits_fc/len(testing_data))

acc_cnn = []
acc_fc = []
acc_algo = []
noise = []

b = 0

testing_data = get_train_dataset("../data/no_noise/unsorted/test_fin.txt", "../data/no_noise/unsorted/test_inf.txt", 1)
#testing_data_big = get_train_dataset("../data/test_fin_12x12.txt", "../data/test_inf_12x12.txt", 1)
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  cnn, fc, algo = test_var_noise()
  #fc = test_var_noise()
  #acc.append(net)
  acc_fc.append(round(fc, 3))
  acc_cnn.append(round(cnn, 3))
  #acc_won.append(won)
  acc_algo.append(round(algo, 3))
  noise.append(b*100)
  b = round(b + 0.02, 3)

np.savez('../results/evaluation/Accuracy_Noise_fc_cnn.npz', cnn=acc_cnn, fcnn=acc_fc, noise=noise)


#plt.plot(noise, acc, label = "CNN")
plt.plot(noise, acc_fc, label = "FCNN")
plt.plot(noise, acc_cnn, label = "CNN")
#plt.plot(noise, acc_won, label = "FCNN without Noise")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise_cutoff_fc_cnn.png')
plt.show()

