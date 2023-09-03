from hlp_fncs import *
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset Class which inherits for torch Dataset
class Custom_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input_data, target_data = self.data[index]
        return input_data, target_data

    def __len__(self):
        return len(self.data)

# Gives shuffeled Dataset from file with data generated under and from file not generated under the ISM with given Batchsize     
def get_train_dataset(fin_file, inf_file, batch_size):

  data_inf = read_data_tens(inf_file)
  data_inf_targ = []

  for matrix in data_inf:
    data_inf_targ.append((matrix, torch.Tensor([1])))

  del data_inf

  data_fin = read_data_tens(fin_file)
  data_fin_targ = []

  for matrix in data_fin:
    data_fin_targ.append((matrix, torch.Tensor([0])))

  del data_fin

  training_data = data_inf_targ + data_fin_targ

  del data_inf_targ, data_fin_targ

  dataset = Custom_Dataset(training_data)

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return data_loader
    