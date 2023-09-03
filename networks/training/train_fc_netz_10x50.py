import sys
sys.path.append('../net_implementations')
sys.path.append('../..')

from hlp_fncs import *
from get_dataset import *
from fc_netz_10x50 import *
import torch.optim as optim
import matplotlib.pyplot as plt

# training script for a FCNN for matices with size 10x50

training_data = get_train_dataset("../../data/10x50/douplets/noise/sorted/train_fin_10x50_douplets_noise_sorted.txt", "../../data/10x50/douplets/noise/sorted/train_inf_10x50_douplets_noise_sorted.txt", 256)
testing_data = get_train_dataset("../../data/10x50/douplets/noise/sorted/test_fin_10x50_douplets_1_noise_5_sorted.txt", "../../data/10x50/douplets/noise/sorted/test_inf_10x50_douplets_1_noise_5_sorted.txt", 10)

netz = FC_Netz()

netz = netz.cuda()

optimizer = optim.Adam(netz.parameters(), lr = 0.001)

train_acc = []
train_loss = []
test_acc = []
test_loss = []

for epoch in range(100):
  print("epoch:", epoch+1)
  tr_acc, tr_loss = train(netz, training_data, optimizer)
  te_acc, te_loss = test(netz, testing_data)
  train_acc.append(tr_acc)
  train_loss.append(tr_loss)
  test_acc.append(te_acc)
  test_loss.append(te_loss)

torch.save(netz, '../saved_nets/saved_fc_netz_10x50_douplets.py')

np.savez('../../results/learning_curves/Learning_Curves_10x50_douplets.npz', train_acc=train_acc, train_loss=train_loss, test_acc=test_acc, test_loss=test_loss)

plt.plot(train_acc, label = "Training Accuracy")
plt.plot(test_acc, label = "Testing Accuracy")
plt.plot(train_loss, label = "Training Loss")
plt.plot(test_loss, label = "Testing Loss")
plt.xlabel('Epoch')
plt.title('Learning Curves')
plt.legend()
plt.savefig('../../results/learning_curves/Learning_Curves_10x50_douplets.png')
plt.show()
