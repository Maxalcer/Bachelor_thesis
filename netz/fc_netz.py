from hlp_fncs import *
from get_dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FC_Netz(nn.Module):
  def __init__(self):
    super(FC_Netz, self).__init__()
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

def train(netz, training_data, optimizer):
  netz.train()
  total_loss = 0
  total_acc = 0
  for input, target in training_data:
    input = Variable(input).cuda()
    output = netz(input)
    target = Variable(target).cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss += loss.item()
    total_acc += accuracy(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return (total_acc/len(training_data)), (total_loss/len(training_data))
  
  
def test(netz, testing_data):
  netz.eval()
  total_loss = 0
  total_acc = 0
  for input, target in testing_data:
    input = Variable(input).cuda()
    output = netz(input)
    target = target.cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss += loss.item()
    total_acc += accuracy(output, target)
  return (total_acc/len(testing_data)), (total_loss/len(testing_data))