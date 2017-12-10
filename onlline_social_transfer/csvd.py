import sys, time
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import sparse
import DYP
import test


train_male = pd.read_csv('input/male_train.csv', header=None).values
train_female = pd.read_csv('input/female_train.csv', header=None).values
male_set = set(train_male[:, 0]) & set(train_female[:, 0])
female_set = set(train_male[:, 1]) & set(train_female[:, 1])
male_index_dict = dict(zip(male_set, range(len(male_set))))
female_index_dict = dict(zip(female_set, range(len(female_set))))

train_male = np.array([[male_index_dict[i[0]], female_index_dict[i[1]], i[2]]
    for i in train_male if i[0] in male_index_dict and i[1] in female_index_dict])
train_female = np.array([[male_index_dict[i[0]], female_index_dict[i[1]], i[2]] 
    for i in train_female if i[0] in male_index_dict and i[1] in female_index_dict])
n_users, n_items = len(male_set), len(female_set)


model = DYP.CML(n_users, n_items, 5, 1)

begin = time.time()
l_loss = 0
i = 0
stop = 0
for i in tqdm(range(500)):
    c_loss = model.partial_fit(train_male[:, [0, 1]], train_male[:, [2]], 
                        train_female[:, [0, 1]], train_female[:, [2]])    
    if abs(c_loss - l_loss) == 0:
        stop += 1
    else:
        stop =0
    if stop >= 100:
        break
    l_loss = c_loss
    sys.stderr.write(("loss function %f" % (c_loss)))
    sys.stderr.flush()
    i += 1

male_prediction = model.prediction_matrix()

test_data = pd.read_csv('input/male_test.csv', header=None).values
# male_test_raw[:, 2] = 2 #计算单边auc


test_data = np.array([[male_index_dict[i[0]], female_index_dict[i[1]], i[2]]
    for i in test_data if i[0] in male_index_dict and i[1] in female_index_dict])
test.user_auc(male_prediction, train_male, test_data)

p_array = np.array(list(map(lambda x: male_prediction[x[0], x[1]], test_data)))
test_y = test_data[:, 2]
print(test.sample_auc(p_array, test_y, 1))


train_dict = test.data_to_dict(train_male, 0)
test_dict = test.data_to_dict(test_data, 0)

precision_list, recall_list = [], []
for k in [5, 10, 50]:
    precision, recall = test.precision_recall(male_prediction, test_dict, train_dict, top=k, mode='base').values[0]
    precision_list.append(precision)
    recall_list.append(recall)

plt.scatter(precision_list, recall_list)
plt.show()


