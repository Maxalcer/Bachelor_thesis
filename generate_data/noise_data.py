import sys
sys.path.append('../')

from hlp_fncs import *
import numpy as np

# Script for adding alpha and beta (with seperate levels) to the data

alpha = 10**(-5)

train_data_fin = read_data_mat("../data/10x50/douplets/no_noise/train_fin_10x50_douplets.txt")
train_data_inf = read_data_mat("../data/10x50/douplets/no_noise/train_inf_10x50_douplets.txt")

beta = 0.05

# Noises the Trainging Data with 0.05 <= beta <= 0.25
if (len(train_data_fin) == len(train_data_fin)):

  beta_step = 0.2/len(train_data_fin)

  for i in range(len(train_data_fin)):
    if((i % 5000) == 0): print(i/5000, "%")
    train_data_inf[i] = noise_matrix(train_data_inf[i], alpha, beta)
    train_data_fin[i] = noise_matrix(train_data_fin[i], alpha, beta)
    beta += beta_step

  write_data("../data/10x50/douplets/noise/unsorted/train_fin_10x50_douplets_noise.txt", train_data_fin)
  write_data("../data/10x50/douplets/noise/unsorted/train_inf_10x50_douplets_noise.txt", train_data_inf)

del train_data_inf, train_data_fin

test_data_inf = read_data_mat("../data/10x50/douplets/no_noise/test_inf_10x50_douplets_1.txt")
test_data_fin = read_data_mat("../data/10x50/douplets/no_noise/test_fin_10x50_douplets_1.txt")

# Noises the testing data with fixed beta
if (len(test_data_fin) == len(test_data_fin)):

  test_noise_5_inf = []
  test_noise_5_fin = []
  #test_noise_15_inf = []
  #test_noise_15_fin = []
  #test_noise_25_inf = []
  #test_noise_25_fin = []

  for fin, inf in zip(test_data_fin, test_data_inf):
    test_noise_5_inf.append(noise_matrix(inf, alpha, 0.05))
    test_noise_5_fin.append(noise_matrix(fin, alpha, 0.05))
    #test_noise_15_inf.append(noise_matrix(inf, alpha, 0.15))    
    #test_noise_15_fin.append(noise_matrix(fin, alpha, 0.15))
    #test_noise_25_inf.append(noise_matrix(inf, alpha, 0.25))
    #test_noise_25_fin.append(noise_matrix(fin, alpha, 0.25))

  write_data("../data/10x50/douplets/noise/unsorted/test_fin_10x50_douplets_1_noise_5.txt", test_noise_5_fin)
  write_data("../data/10x50/douplets/noise/unsorted/test_inf_10x50_douplets_1_noise_5.txt", test_noise_5_inf)
  #write_data("../data/test_fin_12x12_noise_15.txt", test_noise_15_fin)
  #write_data("../data/test_inf_12x12_noise_15.txt", test_noise_15_inf)
  #write_data("../data/test_fin_12x12_noise_25.txt", test_noise_25_fin)
  #write_data("../data/test_inf_12x12_noise_25.txt", test_noise_25_inf)
