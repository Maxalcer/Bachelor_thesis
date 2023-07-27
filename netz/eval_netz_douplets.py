from hlp_fncs import *
from get_dataset import *
from convol_netz import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

netzfc = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))

def classify(input, cutoff):
  if input > cutoff: return 1
  else: return 0

def test_var_noise():
  netzfc.eval()
  hits0 = 0
  hits1 = 0
  hits2 = 0
  hits3 = 0
  for input, target in testing_data0:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output = netzfc(input)
    target = Variable(target)
    if(classify(output, 0.5) == target): hits0 += 1
  for input, target in testing_data1:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output = netzfc(input)
    target = Variable(target)
    if(classify(output, 0.5) == target): hits1 += 1
  for input, target in testing_data2:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output = netzfc(input)
    target = Variable(target)
    if(classify(output, 0.5) == target): hits2 += 1
  for input, target in testing_data3:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output = netzfc(input)
    target = Variable(target)
    if(classify(output, 0.5) == target): hits3 += 1

  return (hits0/len(testing_data0)), (hits1/len(testing_data1)), (hits2/len(testing_data2)), (hits3/len(testing_data3))

acc_0 = []
acc_1 = []
acc_2 = []
acc_3 = []
noise = []

b = 0

testing_data0 = get_train_dataset("../data/no_noise/unsorted/test_fin.txt", "../data/no_noise/unsorted/test_inf.txt", 1)
testing_data1 = get_train_dataset("../data/douplets/no_noise/unsorted/test_douplets_1_fin.txt", "../data/douplets/no_noise/unsorted/test_douplets_1_inf.txt", 1)
testing_data2 = get_train_dataset("../data/douplets/no_noise/unsorted/test_douplets_2_fin.txt", "../data/douplets/no_noise/unsorted/test_douplets_2_inf.txt", 1)
testing_data3 = get_train_dataset("../data/douplets/no_noise/unsorted/test_douplets_3_fin.txt", "../data/douplets/no_noise/unsorted/test_douplets_3_inf.txt", 1)
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  a0, a1, a2, a3 = test_var_noise()
  acc_0.append(round(a0, 3))
  acc_1.append(round(a1, 3))
  acc_2.append(round(a2, 3))
  acc_3.append(round(a3, 3))
  noise.append(b*100)
  b = round(b + 0.005, 3)

np.savez('../results/evaluation/Accuracy_Noise_FCNN_douplets.npz', acc_0=acc_0, acc_1=acc_1, acc_2=acc_2, acc_3=acc_3, noise=noise)

plt.plot(noise, acc_0, label = "0 douplets")
plt.plot(noise, acc_1, label = "1 douplet")
plt.plot(noise, acc_2, label = "2 douplets")
plt.plot(noise, acc_3, label = "3 douplets")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/evaluation/Accuracy_Noise_FCNN_douplets.png')
plt.show()

