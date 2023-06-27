from hlp_fncs import *
from get_dataset import *
from convol_netz import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

netz = torch.load('saved_netz.py')

def test_var_noise():
  netz.eval()
  hits = 0
  hits_algo = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b))
    algo_erg = check_inf_sites(np.array(input))
    if (algo_erg == int(target[0])): hits_algo += 1
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    target = Variable(target)
    if(torch.round(output) == target): hits += 1
  return (hits/len(testing_data)), (hits_algo/len(testing_data))

acc = []
acc_algo = []
noise = []

b = 0

testing_data = get_train_dataset("../data/test_fin_sorted.txt", "../data/test_inf_sorted.txt", 1)
while (b <= 0.25):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  net, algo = test_var_noise()
  acc.append(net)
  acc_algo.append(algo)
  noise.append(b)
  b += 0.01


plt.plot(noise, acc, label = "CNN")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise.png')
plt.show()

