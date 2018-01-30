# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:29:08 2017

@author: QYH
"""

# 內建库
import sys
sys.path.append('theano_bpr/')
# 第三方库
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
# 本地库
import utils
import bpr
import test

male_train_raw = pd.read_csv('../public_data/male_train.csv', header=None).values
# =============================================================================
# male_train_raw = male_train_raw[:, [1,0,2]]
# =============================================================================
# =============================================================================
# female_train_raw = pd.read_csv('../public_data/male_train.csv', header=None).values
# female_train_posi = female_train_raw[female_train_raw[:, 2]>=2]
# male_train_raw = np.r_[male_train_raw, female_train_posi]
# =============================================================================


male_set = set(male_train_raw[male_train_raw[:, 2]==2, 0])
#==============================================================================
female_set = set(male_train_raw[male_train_raw[:, 2]==2, 1])
#==============================================================================
# =============================================================================
# female_set = set(male_train_raw[:, 1])
# =============================================================================
male_to_index = dict(zip(male_set, range(len(male_set))))
female_to_index = dict(zip(female_set, range(len(female_set))))

male_train = np.array([[male_to_index[i[0]], female_to_index[i[1]], i[2]]
    for i in male_train_raw if i[0] in male_to_index and i[1] in female_to_index])

# =============================================================================
# male_train, male_to_index, female_to_index = utils.load_data_from_array(
#     male_train_raw, male_to_index, female_to_index)
# =============================================================================

male_bpr = bpr.BPR(rank=50, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=2)

male_bpr.train(male_train, epochs=3000)

male_prediction = male_bpr.prediction_to_matrix()


male_test_raw = pd.read_csv('../public_data/male_test.csv', header=None).values
# =============================================================================
# male_test_raw = male_test_raw[:, [1,0,2]]
# =============================================================================
# =============================================================================
# male_test_raw[:, 2] = 2 #计算单边auc
# =============================================================================

# =============================================================================
# male_test, male_to_index, female_to_index = utils.load_data_from_array(
#     male_test_raw, male_to_index, female_to_index)
# =============================================================================

male_test = np.array([[male_to_index[i[0]], female_to_index[i[1]], i[2]]
    for i in male_test_raw if i[0] in male_to_index and i[1] in female_to_index])

def auc(train_dict, rank_dict, test_dict):
    auc_values = []
    z = 0
    user_set = set(rank_dict.keys()) & set(test_dict.keys())
    for user in user_set:
        predictions = rank_dict[user]
        auc_for_user = 0.0
        n = 0
        pre_items = set(predictions.keys()) - set(train_dict[user].keys())
        pos_items = pre_items & set(test_dict[user].keys())
        neg_items = pre_items - pos_items
        for pos_item in pos_items:
            for neg_item in neg_items:
                n += 1
                if predictions[pos_item] > predictions[neg_item]:
                    auc_for_user += 1
                elif predictions[pos_item] == predictions[neg_item]:
                    auc_for_user += 0.5
        if n > 0:
            auc_for_user /= n
            auc_values.append(auc_for_user)
        z += 1
        if z % 100 == 0 and len(auc_values) > 0:
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z), np.mean(auc_values)))
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()
    return np.mean(auc_values)  



# test.user_auc(male_prediction, male_train, male_test)


# =============================================================================
# with open('../public_data/female_test.csv') as file:
#     test_data_ = pd.read_csv(file, header=None).values
# test_data = np.array([[male_to_index[i[0]], female_to_index[i[1]], i[2]]
#     for i in test_data_ if i[0] in male_to_index and i[1] in female_to_index])
# p_array = np.array(list(map(lambda x: male_prediction[x[0], x[1]], test_data)))
# test_y = test_data[:, 2]
# print(test.sample_auc(p_array, test_y, 2))
# =============================================================================

def data_to_dict(training_data, min_rate):
    train_dict = dict()
    for row in training_data:
        user, item, rate = row
        if rate >= min_rate:
            if user not in train_dict:
                train_dict[user] = dict()
            train_dict[user][item] = rate
    return train_dict

train_dict = data_to_dict(male_train, 2)
test_dict = data_to_dict(male_test, 2)
pre_dict = male_bpr.prediction_to_dict(100)

print(auc(train_dict, pre_dict, test_dict))

precision_list, recall_list = [], []
for k in range(5, 100, 5):
    precision, recall = test.precision_recall(pre_dict, test_dict, train_dict, top=k, mode='base').values[0]
    precision_list.append(precision)
    recall_list.append(recall)
    
cover_list = []
for k in [5, 10, 50]:
    cover = test.coverage(pre_dict, np.array(male_test), k)
    cover_list.append(cover)

plt.scatter(precision_list, recall_list)
plt.show()

with open('../public_data/log.csv', 'a') as f:
    log = [precision_list[0], precision_list[1], precision_list[9], recall_list[0], recall_list[1], recall_list[9]]
    log_format = list(map(lambda x: float('%0.4f' % x), log))
    print(log_format)
    s = 'algorithm,' + str(log_format)[1:-1]
    f.write(s)
    f.write('\n')



