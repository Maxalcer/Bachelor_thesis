import sys
sys.path.append('../net_implementations')
sys.path.append('../..')

from hlp_fncs import *
from get_dataset import *
from convol_netz import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable 

# Evaluation of a network with a upstream connected adaptive average pooling layer

netzfc = torch.load('../saved_nets/saved_fc_netz.py', map_location=torch.device('cpu'))
netzfc15 = torch.load('../saved_nets/saved_fc_netz_15x15.py', map_location=torch.device('cpu'))

netzfc = netzfc.cpu() 
avgpool = nn.AdaptiveAvgPool2d(10)

# Calculates accuracy for the testing data
def test_var_noise():
  netzfc.eval()
  hits = 0
  hits_15 = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b))
    input = torch.tensor(sort_matrix(np.array(input)))
    output_15 = netzfc15(input)
    input = avgpool(input.repeat(1,1,1))
    input = input[0]
    output_fc = netzfc(input)
    target = Variable(target)
    if(torch.round(output_fc) == target): hits += 1
    if(torch.round(output_15) == target): hits_15 += 1
  return (hits/len(testing_data)), (hits_15/len(testing_data))

acc = []
acc_15 = []
noise = []

b = 0

testing_data = get_train_dataset("../../data/15x15/no_noise/test_fin_15x15.txt", "../../data/15x15/no_noise/test_inf_15x15.txt", 1)
# Calculates accuracy for different levels of beta
while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  net, net_15 = test_var_noise()
  acc.append(round(net, 3))
  acc_15.append(round(net_15, 3))
  noise.append(b*100)
  b = round(b+0.02, 3)

# Save and plot results
np.savez('../../results/evaluation/Accuracy_Noise_avgpl.npz', acc=acc, acc_15=acc_15, noise=noise)

plt.plot(noise, acc, label = "Average Pooling Layer")
plt.plot(noise, acc_15, label = "Adjusted Input Layer")
plt.ylabel('Accuracy')
plt.xlabel('Beta [%]')
plt.legend()
plt.savefig('../../results/Accuracy_Noise_avgpl.png')
plt.show()

