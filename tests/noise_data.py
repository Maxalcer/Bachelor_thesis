from hlp_fncs import *
import numpy as np

train_data_fin = read_data_mat("../data/train_join.txt")
#train_data_inf = read_data_mat("../data/train_inf.txt")

alpha = 10**(-5)
beta = 0.05

if (len(train_data_fin) == len(train_data_fin)):

  beta_step = 0.2/len(train_data_fin)

  for i in range(len(train_data_fin)):
    if((i % 5000) == 0): print(i/5000, "%")
    #train_data_inf[i] = noise_matrix(train_data_inf[i], alpha, beta)
    train_data_fin[i] = noise_matrix(train_data_fin[i], alpha, beta)
    beta += beta_step

  write_data("../data/train_join_noise.txt", train_data_fin)
  #write_data("../data/train_inf_noise.txt", train_data_inf)

#del train_data_inf, train_data_fin

#test_data_inf = read_data_mat("../data/test_inf.txt")
test_data_fin = read_data_mat("../data/test_join.txt")

if (len(test_data_fin) == len(test_data_fin)):

  #test_noise_5_inf = []
  test_noise_5_fin = []
  #test_noise_15_inf = []
  test_noise_15_fin = []
  #test_noise_25_inf = []
  test_noise_25_fin = []

  for fin in test_data_fin:
    #test_noise_5_inf.append(noise_matrix(inf, alpha, 0.05))
    test_noise_5_fin.append(noise_matrix(fin, alpha, 0.05))
    #test_noise_15_inf.append(noise_matrix(inf, alpha, 0.15))    
    test_noise_15_fin.append(noise_matrix(fin, alpha, 0.15))
    #test_noise_25_inf.append(noise_matrix(inf, alpha, 0.25))
    test_noise_25_fin.append(noise_matrix(fin, alpha, 0.25))

  write_data("../data/test_join_noise_5.txt", test_noise_5_fin)
  #write_data("../data/test_inf_noise_5.txt", test_noise_5_inf)
  write_data("../data/test_join_noise_15.txt", test_noise_15_fin)
  #write_data("../data/test_inf_noise_15.txt", test_noise_15_inf)
  write_data("../data/test_join_noise_25.txt", test_noise_25_fin)
  #write_data("../data/test_inf_noise_25.txt", test_noise_25_inf)
