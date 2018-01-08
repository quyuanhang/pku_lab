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
import heapq
# 本地库
import utils
import bpr
import test

male_train_raw = pd.read_csv('input/male_train.csv', header=None).values
female_train_raw = pd.read_csv('input/female_train.csv', header=None).values

male_train_match = male_train_raw[male_train_raw[:, 2]==2]

male_test_raw = pd.read_csv('input/male_test.csv', header=None).values
male_test_match = male_test_raw[male_test_raw[:, 2]==2]

male_set = set(male_train_raw[:, 0])
female_set = set(male_train_raw[:, 1])

male_to_index = dict(zip(male_set, range(len(male_set))))
female_to_index = dict(zip(female_set, range(len(female_set))))

male_train, male_to_index, female_to_index = utils.load_data_from_array(
    male_train_raw, male_to_index, female_to_index)
male_bpr = bpr.BPR(rank=50, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=1)
male_bpr.train(male_train, epochs=5000)

female_train, male_to_index, female_to_index = utils.load_data_from_array(
    female_train_raw, male_to_index, female_to_index)
female_bpr = bpr.BPR(rank=50, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=1)
female_bpr.train(female_train, epochs=5000)

male_prediction = male_bpr.prediction_to_matrix()
female_prediction = female_bpr.prediction_to_matrix()

male_prediction_scale = np.argsort(np.argsort(male_prediction, axis=1))
female_prediction_scale = np.argsort(np.argsort(female_prediction, axis=1))
male_prediction_plus_scale = male_prediction_scale + female_prediction_scale



# =============================================================================
# male_test_raw[:, 2] = 2 #计算单边auc
# =============================================================================

male_test, male_to_index, female_to_index = utils.load_data_from_array(
    male_test_raw, male_to_index, female_to_index)

test.user_auc(male_prediction, male_train, male_test)



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

def mat_to_dict(mat, topn):
    d = dict()
    for i, row in enumerate(mat):
        d[i] = dict(heapq.nlargest(topn, enumerate(row), key=lambda x: x[1]))
    return d

pre_dict = mat_to_dict(male_prediction, 50)


precision_list, recall_list = [], []
for k in [5, 10, 50]:
    precision, recall = test.precision_recall(pre_dict, test_dict, train_dict, top=k, mode='base').values[0]
    precision_list.append(precision)
    recall_list.append(recall)
    
cover_list = []
for k in [5, 10, 50]:
    cover = test.coverage(pre_dict, np.array(male_test), k)
    cover_list.append(cover)

plt.scatter(precision_list, recall_list)
plt.show()


