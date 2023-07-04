from sklearn import tree
import numpy as np
import random
import matplotlib.pyplot as plt
from hlp_fncs import *

inf = read_data_mat("../data/train_inf_noise_sorted.txt")

inf = [(i, 1) for i in inf]

fin = read_data_mat("../data/train_inf_noise_sorted.txt")

fin = [(f, 0) for f in fin]

train_data = inf + fin

random.shuffle(train_data)

inputs = [np.reshape(d[0], 100) for d in train_data]
targets = [d[1] for d in train_data]

print("start fitting")
clf = tree.DecisionTreeClassifier()

clf = clf.fit(inputs, targets)

inf = read_data_mat("../data/test_inf_sorted.txt")
inf = [(i, 1) for i in inf]

fin = read_data_mat("../data/test_fin_sorted.txt")
fin = [(f, 0) for f in fin]

test_data = inf + fin
random.shuffle(test_data)

b = 0
acc = []
noise = []

print("start evaluation")

while (b < 0.51):
  print(b)
  if(b == 0): a = 0
  else: a = 10**(-5)
  hits = 0
  for input, target in test_data:
    input = noise_matrix(input, a, b)
    input = np.reshape(input, 100)
    input = np.reshape(input, (1, -1))
    output = clf.predict(input)
    if (output[0] == target): hits += 1
  acc.append(hits/len(test_data))
  noise.append(b*100)
  b += 0.005

plt.plot(noise, acc)
plt.ylabel('Accuracy')
plt.xlabel('Noise Level [%]')
plt.title('Accuracy for different Noise Levels')
plt.savefig('../results/Accuracy_Noise_DT.png')
plt.show()


