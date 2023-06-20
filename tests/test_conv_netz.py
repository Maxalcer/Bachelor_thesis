from hlp_fncs import *
from get_dataset import *
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim

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

#tp_t = []
#fp_t = []
#acc_t = []
#loss_t = []

tp_e = []
fp_e = []
acc_e = []
loss_e = []


def test():
  netz.eval()
  hits = 0
  #tp = 0
  #fp = 0
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
      #if((torch.argmax(output[i]) == torch.argmax(target[i])) & (torch.argmax(output[i]) == torch.argmax(torch.Tensor([1,0])))): tp += 1
      #if((torch.argmax(output[i]) != torch.argmax(target[i])) & (torch.argmax(output[i]) == torch.argmax(torch.Tensor([1,0])))): fp += 1
  print(hits/(len(testing_data)*10))
  #tp_e.append(tp/(len(testing_data)*10))
  #fp_e.append(fp/(len(testing_data)*10))
  acc_e.append(hits/(len(testing_data)*10))
  loss_e.append((total_loss/(len(testing_data))))
        


def train(epoch):
  netz.train()
  total_loss = 0
  #tp = 0
  #fp = 0
  #hits = 0
  for input, target in training_data:
    expanded_input = torch.unsqueeze(input, 0).cuda()
    expanded_input = expanded_input.repeat(1,1,1,1)
    expanded_input = expanded_input.transpose(0, 1)
    expanded_input = Variable(expanded_input)
    output = netz(expanded_input)
    target = Variable(target).cuda()
    criterion = nn.BCELoss()
    loss = criterion(output, target)
    total_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #for i in range(1000):
      #if(torch.argmax(output[i]) == torch.argmax(target[i])): hits += 1
      #if((torch.argmax(output[i]) == torch.argmax(target[i])) & (torch.argmax(output[i]) == torch.argmax(torch.Tensor([1,0])))): tp += 1
      #if((torch.argmax(output[i]) != torch.argmax(target[i])) & (torch.argmax(output[i]) == torch.argmax(torch.Tensor([1,0])))): fp += 1
  print(total_loss/len(training_data))
  #tp_t.append(tp/(len(testing_data)*1000))
  #fp_t.append(fp/(len(testing_data)*1000))
  #acc_t.append(hits/(len(testing_data)*1000))
  #loss_t.append((total_loss/(len(testing_data))))
  scheduler.step()
  
training_data = get_train_dataset("../data/train_fin_manip.txt", "../data/train_inf_manip.txt", 256)
testing_data = get_train_dataset("../data/test_fin_manip.txt", "../data/test_inf_manip.txt", 10)
netz = Netz()

netz = netz.cuda()

optimizer = optim.Adam(netz.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(100):
  print("epoch:", epoch+1)
  train(epoch)
  test()
"""
plt.plot(loss_t)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss for training data')
plt.savefig('../results/training_loss.png')

plt.plot(acc_t)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('accuracy for training data')
plt.savefig('../results/training_accuracy.png')

plt.plot(tp_t, fp_t)
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC for training data')
plt.savefig('../results/training_ROC.png')

"""
plt.plot(loss_e)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('loss for testing data')
plt.savefig('../results/training_loss.png')
plt.show()

plt.plot(acc_e)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('accuracy for testing data')
plt.savefig('../results/training_accuracy.png')
plt.show()

#plt.plot(tp_e, fp_e)
#plt.ylabel('TPR')
#plt.xlabel('FPR')
#plt.title('ROC for testing data')
#plt.savefig('../results/training_ROC.png')



