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

def test():
  netz.eval()
  hits = 0
  total_loss = 0
  for input, target in testing_data:
    expanded_input = torch.unsqueeze(input, 0).cuda()
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    target = Variable(target).cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss += loss.item()
    for i in range(10):
      if(torch.argmax(output[i]) == torch.argmax(target[i])): hits += 1
  print("Accuarcy:", hits/(len(testing_data)*10))
  print("Loss:",(total_loss/(len(testing_data))))
  return hits/(len(testing_data)*10)

acc = []
acc_algo = []
noise = [0, 5, 15, 25]

testing_data = get_train_dataset("../data/test_fin_sorted.txt", "../data/test_inf_sorted.txt", 10)

testing_data_fin_0 = read_data_mat("../data/test_fin.txt")
testing_data_inf_0 = read_data_mat("../data/test_inf.txt")

acc.append(test())
acc_algo.append(check_inf_sites_list(testing_data_inf_0, testing_data_fin_0))

testing_data = get_train_dataset("../data/test_fin_noise_5_sorted.txt", "../data/test_inf_noise_5_sorted.txt", 10)

testing_data_fin_5 = read_data_mat("../data/test_fin_noise_5.txt")
testing_data_inf_5 = read_data_mat("../data/test_inf_noise_5.txt")

acc.append(test())
acc_algo.append(check_inf_sites_list(testing_data_inf_5, testing_data_fin_5))

testing_data = get_train_dataset("../data/test_fin_noise_15_sorted.txt", "../data/test_inf_noise_15_sorted.txt", 10)

testing_data_fin_15 = read_data_mat("../data/test_fin_noise_15.txt")
testing_data_inf_15 = read_data_mat("../data/test_inf_noise_15.txt")

acc.append(test())
acc_algo.append(check_inf_sites_list(testing_data_inf_15, testing_data_fin_15))

testing_data = get_train_dataset("../data/test_fin_noise_25_sorted.txt", "../data/test_inf_noise_25_sorted.txt", 10)

testing_data_fin_25 = read_data_mat("../data/test_fin_noise_25.txt")
testing_data_inf_25 = read_data_mat("../data/test_inf_noise_25.txt")

acc.append(test())
acc_algo.append(check_inf_sites_list(testing_data_inf_25, testing_data_fin_25))

plt.plot(noise, acc, label = "CNN")
plt.plot(noise, acc_algo, label = "Algorithm")
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.legend()
plt.savefig('../results/Accuracy_Noise.png')
plt.show()


