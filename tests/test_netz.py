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
    self.dropout = nn.Dropout(0.2)
    self.lin1 = nn.Linear(10,10)
    self.lin2 = nn.Linear(100,100)
    self.lin3 = nn.Linear(100,100)
    self.lin4 = nn.Linear(100,1)

  def forward(self, x):   
    x = F.relu(self.lin1(x))
    x = x.view(-1,100)    
    x = F.relu(self.lin2(x))
    x = self.dropout(x) 
    x = F.relu(self.lin3(x))
    x = self.dropout(x)
    x = F.sigmoid(self.lin4(x))
    return x

def train(epoch):
  netz.train()
  total_loss = 0
  hits = 0
  for input, target in training_data:
    input = Variable(input).cuda()
    output = netz(input)
    target = Variable(target).cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss = total_loss + loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  print("loss:", total_loss/len(training_data))
  
def test():
  netz.eval()
  hits = 0
  for input, target in testing_data:
    input = Variable(input).cuda()
    output = netz(input)
    target = target.cuda()
    for i in range(10):
      if(torch.round(output[i]) == target[i]): hits += 1
  print(hits/(len(testing_data)*10))
  return (hits/(len(testing_data)*10))

training_data = get_train_dataset("../data/train_fin_noise_sorted.txt", "../data/train_inf_noise_sorted.txt", 256)
testing_data = get_train_dataset("../data/test_fin_noise_5_sorted.txt", "../data/test_inf_noise_5_sorted.txt", 10)
netz = Netz()

netz = netz.cuda()

optimizer = optim.SGD(netz.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


for epoch in range(100):
  print("epoch: ", epoch+1)
  train(epoch)
  test()

torch.save(netz, 'saved_fc_netz.py')
