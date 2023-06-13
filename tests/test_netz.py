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
    self.dropout = nn.Dropout(0.1)
    self.lin1 = nn.Linear(10,10)
    self.lin2 = nn.Linear(100,50)
    self.lin3 = nn.Linear(50,20)
    self.lin4 = nn.Linear(20,2)

  def forward(self, x):    
    x = F.relu(self.lin1(x))
    x = x.view(-1,100)
    x = self.dropout(x)
    x = F.relu(self.lin2(x))
    # x = self.dropout(x)
    x = F.relu(self.lin3(x))
    x = self.lin4(x)
    return F.softmax(x, dim=1)

def train(epoch):
  netz.train()
  total_loss = 0
  hits = 0
  for input, target in training_data:
    #input = input.cuda()
    input = Variable(input)
    output = netz(input)
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
    output = netz(Variable(input))
    
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

# netz = netz.cuda()

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
