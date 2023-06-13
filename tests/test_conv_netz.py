from hlp_fncs import *
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
    self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
    self.lin1 = nn.Linear(16,2)


  def forward(self, x):    
    x = F.max_pool2d(self.conv1(x), 2, 2)
    x = F.relu(x)
    x = F.max_pool2d(self.conv2(x), 2, 2)
    x = F.relu(x)
    x = x.view(-1, 16)
    x = F.relu(self.lin1(x))
    return F.softmax(x, dim=1)

def train(epoch):
  netz.train()
  total_loss = 0
  hits = 0
  for input, target in training_data:
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1)
    expanded_input = Variable(expanded_input)
    #expanded_input = expanded_input.cuda()
    output = netz(expanded_input)
    if(torch.argmax(output[0]) == torch.argmax(target)):
      hits += 1
    #target = target.cuda()
    target = Variable(target)
    criterion = nn.BCELoss()
    loss = criterion(output[0], target)
    total_loss = total_loss + loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  print("loss:", total_loss/len(training_data))
  print("hitrate:", hits/len(training_data))

  
def test():
  netz.eval()
  hits = 0
  for input, target in test_data:
    expanded_input = torch.unsqueeze(input, 0)
    expanded_input = expanded_input.repeat(1,1,1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    
    if(torch.argmax(output[0]) == torch.argmax(target)):
      hits += 1

  print("hitrate: ", hits/len(test_data))

data_inf = read_data_tens("../data/train_inf.txt")

train_inf = []

for matrix in data_inf:
  train_inf.append((matrix, torch.Tensor([1,0])))

del data_inf

data_fin = read_data_tens("../data/train_fin.txt")

train_fin = []

for matrix in data_fin:
  train_fin.append((matrix, torch.Tensor([0,1])))

del data_fin

training_data = train_inf + train_fin

del train_fin, train_inf

random.shuffle(training_data)

netz = Netz()
#netz = netz.cuda()
#training_data = training_data.cuda()
optimizer = optim.SGD(netz.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
  train(epoch)

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

test()
