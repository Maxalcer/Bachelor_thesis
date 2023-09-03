import sys
sys.path.append('../net_implementations')
sys.path.append('../..')

from hlp_fncs import *
from get_dataset import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

# Comparing the accuracy for different input sizes with more samples and mutations

netz10 = torch.load('../saved_nets/saved_fc_netz.py', map_location=torch.device('cpu'))
netz15 = torch.load('../saved_nets/saved_fc_netz_15x15.py', map_location=torch.device('cpu'))
netz25 = torch.load('../saved_nets/saved_fc_netz_25x25.py', map_location=torch.device('cpu'))

def classify(input, cutoff):
  if input > cutoff: return 1
  else: return 0

# Calculating the accuracy for the testing data
def test_var_noise():
  netz10.eval()
  netz15.eval()
  netz25.eval()
  hits_10 = 0
  hits_15 = 0
  hits_25 = 0
  for input, target in testing_data10:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_10 = netz10(input)
    target = Variable(target)
    if(classify(output_10, 0.5) == target): hits_10 += 1
  for input, target in testing_data15:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_15 = netz15(input)
    target = Variable(target)
    if(classify(output_15, 0.5) == target): hits_15 += 1
  for input, target in testing_data25:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_25 = netz25(input)
    target = Variable(target)
    if(classify(output_25, 0.5) == target): hits_25 += 1
  return (hits_10/len(testing_data10)), (hits_15/len(testing_data15)), (hits_25/len(testing_data25))

acc_10 = []
acc_15 = []
acc_25 = []
noise = []
b = 0

testing_data10 = get_train_dataset("../../data/10x10/no_noise/unsorted/test_fin.txt", "../../data/10x10/no_noise/unsorted/test_inf.txt", 1)
testing_data15 = get_train_dataset("../../data/15x15/no_noise/test_fin_15x15.txt", "../../data/15x15/no_noise/test_inf_15x15.txt", 1)
testing_data25 = get_train_dataset("../../data/25x25/no_noise/test_fin_25x25.txt", "../../data/25x25/no_noise/test_inf_25x25.txt", 1)
# Calculting the accuracy for different levels of beta
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  x10, x15, x25 = test_var_noise()
  acc_10.append(round(x10, 3))
  acc_15.append(round(x15, 3))
  acc_25.append(round(x25, 3))
  noise.append(b*100)
  b = round(b + 0.02, 3)

# Saving and ploting the results 
np.savez('../../results/evaluation/Accuracy_Noise_size.npz', acc_10=acc_10, acc_15=acc_15, acc_25=acc_25, noise=noise)

plt.plot(noise, acc_10, label = "10x10")
plt.plot(noise, acc_15, label = "15x15")
plt.plot(noise, acc_25, label = "25x25")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../../results/Accuracy_Noise_size.png')
plt.show()

