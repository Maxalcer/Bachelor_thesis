from hlp_fncs import *
from get_dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

class Netz(nn.Module):
  def __init__(self):
    super(Netz, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1)    
    self.lin1 = nn.Linear(30, 16)
    self.lin2 = nn.Linear(16, 1)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv2(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv3(x), 2, 2)
    x = F.relu(x)
    x = x.view(-1, 30)    
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    return F.sigmoid(x)

def test(netz, testing_data):
  netz.eval()
  hits = 0
  for input, target in testing_data:
    expanded_input = torch.unsqueeze(input, 0).cuda()
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    target = Variable(target).cuda()
    for i in range(10):
      if(torch.round(output[i]) == target[i]): hits += 1
  print(hits/(len(testing_data)*10))
  return (hits/(len(testing_data)*10))
        
def train(netz, training_data, optimizer):
  netz.train()
  total_loss = 0
  for input, target in training_data:
    input = torch.unsqueeze(input, 0).cuda()
    input = input.repeat(1,1,1,1)
    input = input.transpose(0, 1)
    input = Variable(input)
    output = netz(input)
    target = Variable(target).cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(total_loss/len(training_data))




