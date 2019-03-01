import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

np.random.seed(0)
training_data = pd.read_csv(
    'model/data/20190219_125466/classes_20180925_141203/4_2.csv', index_col=0, header=None)
print(training_data)

training_data_2 = pd.read_csv(
    'model/data/20190219_125466/classes_20180925_143754/4_2.csv', index_col=0, header=None)
print(training_data_2)

training_data_3 = pd.read_csv(
    'model/data/20190219_125466/classes_20180928_112016/4_2.csv', index_col=0, header=None)    
    

training_data = pd.concat([training_data, training_data_2], axis=0)
training_data = pd.concat([training_data, training_data_3], axis=0)
print(training_data)

count_dict = {}
for i in range(1, 7):
    data_counts = len(training_data[training_data[1] == i])
    print('data_counts')
    print(data_counts)
    count_category = {i: data_counts}
    count_dict.update(count_category)
print(count_dict)

prob_list = []
for k, v in sorted(count_dict.items()):
    print(k, v)
    prob_list.append(v)

prob_list = np.array(prob_list)
print(prob_list)

prob_list = prob_list / len(training_data)
print(prob_list)

training_data_list = [
    np.random.multinomial(
        i + 1,
        prob_list,
    ) for i in range(1000)
]
# print(training_data_list)

pattern = 'a'
alpha = [1, 1, 1, 1, 1, 1]

m = np.diag(np.ones(6))
print(m)
pi = np.random.dirichlet(
    [training_data_list[100][i] + alpha[i] for i in range(6)])

print(pi)

pred_list = []
for data in training_data_list:
    pi = np.random.dirichlet([data[i] + alpha[i] for i in range(6)])
    print('pi')
    print(pi)
    pred = [stats.multinomial.pmf(i, 1, [p for p in pi]) for i in m]
    print('pred')
    print(pred)
    pred_list.append(pred)

pred_list = np.array(pred_list)
print(pred_list)

count = 1000
plt.plot(
    np.arange(1, count + 1, 1), pred_list[:, 0], label='{}_1'.format(pattern))
plt.plot(
    np.arange(1, count + 1, 1), pred_list[:, 1], label='{}_2'.format(pattern))
plt.plot(
    np.arange(1, count + 1, 1), pred_list[:, 2], label='{}_3'.format(pattern))
plt.plot(
    np.arange(1, count + 1, 1), pred_list[:, 3], label='{}_4'.format(pattern))
plt.plot(
    np.arange(1, count + 1, 1), pred_list[:, 4], label='{}_5'.format(pattern))
plt.plot(
    np.arange(1, count + 1, 1), pred_list[:, 5], label='{}_6'.format(pattern))
plt.xlabel('count')
plt.ylabel('pred')
plt.ylim(0, 1.1)
plt.legend()
plt.show()

model = np.array([
    pred_list[-1][0],
    pred_list[-1][1],
    pred_list[-1][2],
    pred_list[-1][3],
    pred_list[-1][4],
    pred_list[-1][5],
])

with open('model/4_2.pkl', 'wb') as m:
    pickle.dump(model, m)
