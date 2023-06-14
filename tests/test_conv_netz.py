from hlp_fncs import *
from get_dataset import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim

class Netz(nn.Module):
  def __init__(self):
    super(Netz, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=1)
    self.drpout = nn.Dropout(0.2)
    self.lin1 = nn.Linear(30, 16)
    self.lin2 = nn.Linear(16,2)


  def forward(self, x):    
    x = F.max_pool2d(self.conv1(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv2(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv3(x), 2, 2)
    x = F.relu(x)
    x = x.view(-1, 30)
    x = self.drpout(x)
    x = F.relu(self.lin1(x))
    x = F.relu(self.lin2(x))
    return F.softmax(x, dim=1)

def test():
  netz.eval()
  hits = 0
  for input, target in test_data:
    expanded_input = torch.unsqueeze(input, 0).cuda()
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    
    if(torch.argmax(output[0]) == torch.argmax(target)):
      hits += 1

  print("hitrate: ", hits/len(test_data))

def train(epoch):
  netz.train()
  total_loss = 0
  hits = 0
  for input, target in training_data:
    expanded_input = torch.unsqueeze(input, 0).cuda()
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    target = Variable(target).cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss = total_loss + loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  #scheduler.step()
  print("loss:", total_loss/len(training_data))
  
training_data = get_train_dataset("../data/train_fin.txt", "../data/train_inf.txt", 64)

netz = Netz()

netz = netz.cuda()

optimizer = optim.Adam(netz.parameters(), lr = 0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

data_inf = read_data_tens("../data/test_inf.txt")

test_inf = []

for matrix in data_inf:
  test_inf.append((matrix, torch.Tensor([1,0])))

del data_inf

data_fin = read_data_tens("../data/test_fin.txt")

test_fin = []

for matrix in data_fin:
  test_fin.append((matrix, torch.Tensor([0,1])))

del data_fin

test_data = test_inf + test_fin
random.shuffle(test_data)
del test_fin, test_inf

for epoch in range(100):
  print("epoch:", epoch+1)
  train(epoch)
  test()


