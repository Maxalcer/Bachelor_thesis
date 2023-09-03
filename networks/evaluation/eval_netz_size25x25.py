import sys
sys.path.append('../net_implementations')
sys.path.append('../..')

from hlp_fncs import *
from get_dataset import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

# Comparing the accuracy for 25x25 matrices with a normal and a big network
netz25_big = torch.load('../saved_nets/saved_fc_netz_25x25_big.py', map_location=torch.device('cpu'))
netz25 = torch.load('../saved_nets/saved_fc_netz_25x25.py', map_location=torch.device('cpu'))

def classify(input, cutoff):
  if input > cutoff: return 1
  else: return 0

# Calcultaing the accuracy for both networks for testing data
def test_var_noise():
  netz25_big.eval()
  netz25.eval()
  hits_25_big = 0
  hits_25 = 0
  for input, target in testing_data25:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    output_25 = netz25(input)
    output_25_big = netz25_big(input)
    target = Variable(target)
    if(classify(output_25, 0.5) == target): hits_25 += 1
    if(classify(output_25_big, 0.5) == target): hits_25_big += 1
  return (hits_25/len(testing_data25)), (hits_25_big/len(testing_data25))

acc_25_big = []
acc_25 = []
noise = []
b = 0

testing_data25 = get_train_dataset("../../data/25x25/no_noise/test_fin_25x25.txt", "../../data/25x25/no_noise/test_inf_25x25.txt", 1)
# Calculting the accuracy for different levels of beta
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  x25, x25_big = test_var_noise()
  acc_25_big.append(round(x25_big, 3))
  acc_25.append(round(x25, 3))
  noise.append(b*100)
  b = round(b + 0.02, 3)

# Saving and ploting the results 
np.savez('../../results/evaluation/Accuracy_Noise_size_25x25.npz', acc_25=acc_25, acc_25_big=acc_25_big, noise=noise)

plt.plot(noise, acc_25_big, label = "Hidden Layers with 625 Neurons")
plt.plot(noise, acc_25, label = "Hidden Layers with 100 Neurons")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../../results/evaluation/Accuracy_Noise_size_25x25.png')
plt.show()

