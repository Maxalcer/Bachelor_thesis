from hlp_fncs import *

train_data_fin = read_data_mat("../data/train_fin.txt")
train_data_inf = read_data_mat("../data/train_inf.txt")

if (len(train_data_fin) == len(train_data_inf)):
  for i in range(len(train_data_inf)):
    if((i % 5000) == 0): print(i/5000, "%")
    train_data_inf[i] = sort_matrix(train_data_inf[i])
    train_data_fin[i] = sort_matrix(train_data_fin[i])

  write_data("../data/train_fin_manip.txt", train_data_fin)
  write_data("../data/train_inf_manip.txt", train_data_inf)

del train_data_inf, train_data_fin

test_data_inf = read_data_mat("../data/test_inf.txt")
test_data_fin = read_data_mat("../data/test_fin.txt")

if (len(test_data_fin) == len(test_data_inf)):
  for i in range(len(test_data_inf)):
    test_data_inf[i] = sort_matrix(test_data_inf[i])
    test_data_fin[i] = sort_matrix(test_data_fin[i])

  write_data("../data/test_fin_manip.txt", test_data_fin)
  write_data("../data/test_inf_manip.txt", test_data_inf)