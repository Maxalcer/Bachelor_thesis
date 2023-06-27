from hlp_fncs import *
from get_dataset import *
from convol_netz import *
import torch.optim as optim

  
training_data = get_train_dataset("../data/train_fin_noise_sorted.txt", "../data/train_inf_noise_sorted.txt", 256)
testing_data = get_train_dataset("../data/test_fin_noise_5_sorted.txt", "../data/test_inf_noise_5_sorted.txt", 10)
netz = Netz()

netz = netz.cuda()

optimizer = optim.Adam(netz.parameters(), lr = 0.001)

for epoch in range(100):
  print("epoch:", epoch+1)
  train(netz, training_data, optimizer)
  test(netz, testing_data)

torch.save(netz, 'saved_netz.py')
