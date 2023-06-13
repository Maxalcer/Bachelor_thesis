from hlp_fncs import *
import torch
from torch.utils.data import Dataset, DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input_data, target_data = self.data[index]
        return input_data, target_data

    def __len__(self):
        return len(self.data)
    
def get_train_dataset(fin_file, inf_file, batch_size):

  data_inf = read_data_tens(inf_file)

  train_inf = []

  for matrix in data_inf:
    train_inf.append((matrix, torch.Tensor([1,0])))

  del data_inf

  data_fin = read_data_tens(fin_file)

  train_fin = []

  for matrix in data_fin:
    train_fin.append((matrix, torch.Tensor([0,1])))

  del data_fin

  training_data = train_inf + train_fin

  del train_fin, train_inf

  dataset = Custom_Dataset(training_data)

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return data_loader
    