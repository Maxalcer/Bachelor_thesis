from hlp_fncs import *
from get_dataset import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

netz10 = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))
netz25 = torch.load('saved_fc_netz_10x25.py', map_location=torch.device('cpu'))
netz50 = torch.load('saved_fc_netz_10x50.py', map_location=torch.device('cpu'))

def classify(input, cutoff):
  if input > cutoff: return 1
  else: return 0

def test_var_noise():
  netz10.eval()
  netz25.eval()
  netz50.eval()
  hits_10 = 0
  hits_25 = 0
  hits_50 = 0
  for input, target in testing_data10:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_10 = netz10(input)
    target = Variable(target)
    if(classify(output_10, 0.5) == target): hits_10 += 1
  for input, target in testing_data25:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_25 = netz25(input)
    target = Variable(target)
    if(classify(output_25, 0.5) == target): hits_25 += 1
  for input, target in testing_data50:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_50 = netz50(input)
    target = Variable(target)
    if(classify(output_50, 0.5) == target): hits_50 += 1
  return (hits_10/len(testing_data10)), (hits_25/len(testing_data25)), (hits_50/len(testing_data50))

acc_10 = []
acc_25 = []
acc_50 = []
noise = []
b = 0

testing_data10 = get_train_dataset("../data/no_noise/unsorted/test_fin.txt", "../data/no_noise/unsorted/test_inf.txt", 1)
testing_data25 = get_train_dataset("../data/10x25/no_noise/test_fin_10x25.txt", "../data/10x25/no_noise/test_inf_10x25.txt", 1)
testing_data50 = get_train_dataset("../data/10x50/no_noise/test_fin_10x50.txt", "../data/10x50/no_noise/test_inf_10x50.txt", 1)
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  x10, x25, x50 = test_var_noise()
  acc_10.append(round(x10, 3))
  acc_25.append(round(x25, 3))
  acc_50.append(round(x50, 3))
  noise.append(b*100)
  b = round(b + 0.02, 3)

np.savez('../results/evaluation/Accuracy_Noise_size2.npz', acc_10=acc_10, acc_25=acc_25, acc_50=acc_50, noise=noise)

plt.plot(noise, acc_10, label = "10x10")
plt.plot(noise, acc_25, label = "10x25")
plt.plot(noise, acc_50, label = "10x50")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise_size2.png')
plt.show()

