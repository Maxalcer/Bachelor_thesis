from hlp_fncs import *
from get_dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

class Netz(nn.Module):
  def __init__(self):
    super(Netz, self).__init__()
    self.avgpool = nn.AdaptiveAvgPool2d(10)
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)    
    self.lin1 = nn.Linear(64, 32)
    self.lin2 = nn.Linear(32, 1)

  def forward(self, x):
    x = self.avgpool(x) 
    x = F.max_pool2d(self.conv1(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv2(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv3(x), 2, 2)
    x = F.relu(x)
    x = x.view(-1, 64)    
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    return x

def test(netz, testing_data):
  netz.eval()
  total_loss = 0
  total_acc = 0
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
    total_acc += accuracy(output, target)
  print("test accuracy:", (total_acc/len(testing_data)), "test loss:", (total_loss/len(testing_data)))
  return (total_acc/len(testing_data)), (total_loss/len(testing_data))
        
def train(netz, training_data, optimizer):
  netz.train()
  total_loss = 0
  total_acc = 0
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
    total_acc += accuracy(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print("train accuracy:", (total_acc/len(training_data)), "train loss:", (total_loss/len(training_data)))
  return (total_acc/len(training_data)), (total_loss/len(training_data))




