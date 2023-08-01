from hlp_fncs import *
import numpy as np
alpha = 10**(-5)

train_data_fin = read_data_mat("../data/25x25/no_noise/train_fin_25x25.txt")
train_data_inf = read_data_mat("../data/25x25/no_noise/train_inf_25x25.txt")

beta = 0.05

if (len(train_data_fin) == len(train_data_fin)):

  beta_step = 0.2/len(train_data_fin)

  for i in range(len(train_data_fin)):
    if((i % 5000) == 0): print(i/5000, "%")
    train_data_inf[i] = noise_matrix(train_data_inf[i], alpha, beta)
    train_data_fin[i] = noise_matrix(train_data_fin[i], alpha, beta)
    beta += beta_step

  write_data("../data/25x25/noise/unsorted/train_fin_25x25_noise.txt", train_data_fin)
  write_data("../data/25x25/noise/unsorted/train_inf_25x25_noise.txt", train_data_inf)

del train_data_inf, train_data_fin

test_data_inf = read_data_mat("../data/25x25/no_noise/test_inf_25x25.txt")
test_data_fin = read_data_mat("../data/25x25/no_noise/test_fin_25x25.txt")

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

  write_data("../data/25x25/noise/unsorted/test_fin_25x25_noise_5.txt", test_noise_5_fin)
  write_data("../data/25x25/noise/unsorted/test_inf_25x25_noise_5.txt", test_noise_5_inf)
  #write_data("../data/test_fin_12x12_noise_15.txt", test_noise_15_fin)
  #write_data("../data/test_inf_12x12_noise_15.txt", test_noise_15_inf)
  #write_data("../data/test_fin_12x12_noise_25.txt", test_noise_25_fin)
  #write_data("../data/test_inf_12x12_noise_25.txt", test_noise_25_inf)
