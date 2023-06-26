from hlp_fncs import *
from get_dataset import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

class Netz(nn.Module):
  def __init__(self):
    super(Netz, self).__init__()
    #self.drpout = nn.Dropout(0.1)
    self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1)    
    self.lin1 = nn.Linear(30, 16)
    self.lin2 = nn.Linear(16,2)


  def forward(self, x):
    #x = self.drpout(x)
    x = F.max_pool2d(self.conv1(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv2(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv3(x), 2, 2)
    x = F.relu(x)
    x = x.view(-1, 30)    
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    return F.softmax(x, dim=1)

netz = torch.load('conv_netz.py')
netz = netz.cuda()

def test():
  netz.eval()
  hits = 0
  hits_algo = 0
  for input, target in testing_data:
    is_inf = bool(np.array(target)[0,0])
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b))
    algo_erg = check_inf_sites(np.array(input))
    if (algo_erg == is_inf): hits_algo += 1
    expanded_input = torch.unsqueeze(input, 0).cuda()
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    target = Variable(target).cuda()
    if(torch.argmax(output) == torch.argmax(target)): hits += 1
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
  net, algo = test()
  print(net)
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

