from hlp_fncs import *
from get_dataset import *
from fc_netz import *
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

netzfc = torch.load('saved_fc_netz.py', map_location=torch.device('cpu'))

def test_var_noise():
  netzfc.eval()
  fn_n = 0
  fp_n = 0
  fn_a = 0
  fp_a = 0
  for input, target in testing_data:
    input = torch.tensor(noise_matrix(np.array(input[0]), a, b)).unsqueeze(0)
    input = sort_tensor(input)
    algo_erg = check_inf_sites(np.array(input[0]))
    if(algo_erg != int(target[0]) and algo_erg == 0): fn_a += 1
    if(algo_erg != int(target[0]) and algo_erg == 1): fp_a += 1
    output_fc = netzfc(input)
    target = Variable(target)
    if(torch.round(output_fc) != target and torch.round(output_fc) == 0): fn_n += 1
    if(torch.round(output_fc) != target and torch.round(output_fc) == 1): fp_n += 1
  return (fn_n/len(testing_data)), (fp_n/len(testing_data)), (fn_a/len(testing_data)), (fp_a/len(testing_data))


fnr_net = []
fpr_net = []
fnr_alg = []
fpr_alg = []
noise = []

b = 0

testing_data = get_train_dataset("../data/test_fin.txt", "../data/test_inf.txt", 1)

while (b <= 0.5):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  fnr_n, fpr_n, fnr_a, fpr_a = test_var_noise()
  fnr_net.append(round(fnr_n, 3))
  fpr_net.append(round(fpr_n, 3))
  fnr_alg.append(round(fnr_a, 3))
  fpr_alg.append(round(fpr_a, 3))
  noise.append(b*100)
  b = round(b + 0.005, 3)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(noise, fnr_net, label = "FNR")
plt.plot(noise, fpr_net, label = "FPR")
plt.xlabel('Noise Level [%]')
plt.title('Network')
plt.legend()
plt.subplot(1,2,2)
plt.plot(noise, fnr_alg, label = "FNR")
plt.plot(noise, fpr_alg, label = "FPR")
plt.xlabel('Noise Level [%]')
plt.title('Algorithm')
plt.legend()
plt.savefig('../results/FNR_FPR.png')
plt.show()

