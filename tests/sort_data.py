from hlp_fncs import *

train_data_fin = read_data_mat("../data/train_fin_noise.txt")
train_data_inf = read_data_mat("../data/train_inf_noise.txt")

if (len(train_data_fin) == len(train_data_inf)):
  for i in range(len(train_data_inf)):
    if((i % 5000) == 0): print(i/5000, "%")
    train_data_inf[i] = sort_matrix(train_data_inf[i])
    train_data_fin[i] = sort_matrix(train_data_fin[i])

  write_data("../data/train_fin_noise_sorted.txt", train_data_fin)
  write_data("../data/train_inf_noise_sorted.txt", train_data_inf)

del train_data_inf, train_data_fin

test_data_inf_5 = read_data_mat("../data/test_inf_noise_5.txt")
test_data_fin_5 = read_data_mat("../data/test_fin_noise_5.txt")
test_data_inf_15 = read_data_mat("../data/test_inf_noise_15.txt")
test_data_fin_15 = read_data_mat("../data/test_fin_noise_15.txt")
test_data_inf_25 = read_data_mat("../data/test_inf_noise_25.txt")
test_data_fin_25 = read_data_mat("../data/test_fin_noise_25.txt")

if (len(test_data_inf_5) == len(test_data_fin_5)):
  for i in range(len(test_data_fin_5)):
    test_data_inf_5[i] = sort_matrix(test_data_inf_5[i])
    test_data_fin_5[i] = sort_matrix(test_data_fin_5[i])
    test_data_inf_15[i] = sort_matrix(test_data_inf_15[i])
    test_data_fin_15[i] = sort_matrix(test_data_fin_15[i])
    test_data_inf_25[i] = sort_matrix(test_data_inf_25[i])
    test_data_fin_25[i] = sort_matrix(test_data_fin_25[i])

  write_data("../data/test_fin_noise_5_sorted.txt", test_data_fin_5)
  write_data("../data/test_inf_noise_5_sorted.txt", test_data_inf_5)
  write_data("../data/test_fin_noise_15_sorted.txt", test_data_fin_15)
  write_data("../data/test_inf_noise_15_sorted.txt", test_data_inf_15)
  write_data("../data/test_fin_noise_25_sorted.txt", test_data_fin_25)
  write_data("../data/test_inf_noise_25_sorted.txt", test_data_inf_25)