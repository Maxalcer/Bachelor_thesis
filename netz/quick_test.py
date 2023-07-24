from hlp_fncs import *
from get_dataset import *
from fc_netz import *

netz = torch.load('saved_fc_netz_won.py')
netz.eval()
testing_data = get_train_dataset("../data/test_fin_sorted.txt", "../data/test_inf_sorted.txt", 16)

acc = 0

for input, target in testing_data:
  input = input.cuda()
  target = target.cuda()
  output = netz(input)
  acc += accuracy(output, target)

print(acc/len(testing_data))
