import sys, time
import pandas as pd
import numpy as np
import DYP
import test
from tqdm import tqdm
import matplotlib.pyplot as plt


train_male = pd.read_csv('../public_data/male_train.csv', header=None).values
train_female = pd.read_csv('../public_data/female_train.csv', header=None).values

male_train_match = train_male[train_male[:, 2]==2]

male_set = set(male_train_match[:, 0])
female_set = set(male_train_match[:, 1])
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
for i in tqdm(range(5000)):
    c_loss = model.partial_fit(train_male[:, [0, 1]], train_male[:, [2]], 
                        train_female[:, [0, 1]], train_female[:, [2]])    
    if abs(c_loss - l_loss) == 0:
        stop += 1
    else:
        stop =0
    if stop >= 100:
        break
    l_loss = c_loss
# =============================================================================
#     sys.stderr.write(("/r loss function %f" % (c_loss)))
#     sys.stderr.flush()
# =============================================================================
    i += 1

male_prediction = model.prediction_matrix()

test_data = pd.read_csv('../public_data/male_test.csv', header=None).values
# =============================================================================
# test_data[:, 2] = 2 #计算单边auc
# =============================================================================


test_data = np.array([[male_index_dict[i[0]], female_index_dict[i[1]], i[2]]
    for i in test_data if i[0] in male_index_dict and i[1] in female_index_dict])
test.user_auc(male_prediction, train_male, test_data)

# =============================================================================
# p_array = list()
# for row in test_data:
#     p_array.append(male_prediction[row[0], row[1]])
# p_array = np.array(p_array)
# 
# p_array = np.array(list(map(lambda x: male_prediction[np.asscalar(x[0]), np.asscalar(x[1])], test_data)))
# test_y = test_data[:, 2]
# print(test.sample_auc(p_array, test_y, 2))
# =============================================================================


train_dict = test.data_to_dict(train_male, 2)
test_dict = test.data_to_dict(test_data, 2)

precision_list, recall_list = [], []
for k in range(5, 100, 5):
    precision, recall = test.precision_recall(male_prediction, test_dict, train_dict, top=k, mode='base').values[0]
    precision_list.append(precision)
    recall_list.append(recall)
plt.scatter(precision_list, recall_list)

f1 = np.polyfit(precision_list, recall_list, 2)
# =============================================================================
# line_x = range(min(precision_list), max(precision_list), 0.01)
# =============================================================================
line_x = np.linspace(min(precision_list), max(precision_list), 10)
line_y = np.polyval(f1, line_x)
plt.plot(line_x, line_y)
# p1 = np.poly1d(f1)
# print(p1)



plt.show()

with open('../public_data/log.csv', 'a') as f:
    log = [precision_list[0], precision_list[1], precision_list[9], recall_list[0], recall_list[1], recall_list[9]]
    log_format = list(map(lambda x: float('%0.4f' % x), log))
    print(log_format)
    s = 'csvd,' + str(log_format)[1:-1]
    f.write(s)
    f.write('\n')


