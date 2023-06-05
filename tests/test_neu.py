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
    self.lin1 = nn.Linear(10,10)
    self.lin2 = nn.Linear(100,20)
    self.lin3 = nn.Linear(20,2)

  def forward(self, x):
    x = F.relu(self.lin1(x))
    x = x.view(-1, 100)
    x = F.relu(self.lin2(x))
    x = self.lin3(x)
    return x

def read_data(input):

  data = []

  file = open(input, "r")

  content = file.read()

  matrices = content.split('\n\n')

  for m in matrices:
    rows = m.split('\n')
    matrix = [list(map(float, list(row))) for row in rows]    
    data.append(torch.Tensor(matrix))

  return data

def train(epoch):
  netz.train()
  for input, target in training_data:
    output = netz(input)
    output = output[0]
    target_tensor = Variable(target)
    criterion = nn.NLLLoss()
    loss = criterion(output, target_tensor)
    netz.zero_grad()
    loss.backward()
    optimizer.step

def test():
  netz.eval()
  hits = 0
  for input, target in test_data:
    output = netz(input)
    output = output[0]
    
    if(torch.argmax(output) == torch.argmax(target)):
      hits += 1

  print("hitrate: ", hits/len(test_data))

data_inf = read_data("train_inf.txt")

train_inf = []

for matrix in data_inf:
  train_inf.append((matrix, torch.LongTensor([1,0])))

del data_inf

data_fin = read_data("train_fin.txt")

train_fin = []

for matrix in data_fin:
  train_fin.append((matrix, torch.LongTensor([0,1])))

del data_fin

training_data = train_inf + train_fin

del train_fin, train_inf

random.shuffle(training_data)

netz = Netz()

optimizer = optim.Adam(netz.parameters(), lr = 0.1)

for epoch in range(30):
  train(epoch)

data_inf = read_data("test_inf.txt")

test_inf = []

for matrix in data_inf:
  test_inf.append((matrix, torch.LongTensor([1,0])))

del data_inf

data_fin = read_data("test_fin.txt")

test_fin = []

for matrix in data_fin:
  test_fin.append((matrix, torch.LongTensor([0,1])))

del data_fin

test_data = test_inf + test_fin

del test_fin, test_inf

test()
