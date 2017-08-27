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

male_train_raw = pd.read_csv('input/male_train.csv', header=None).values
male_test_raw = pd.read_csv('input/male_test.csv', header=None).values
female_train_raw = pd.read_csv('input/female_train.csv', header=None).values
female_test_raw = pd.read_csv('input/female_test.csv', header=None).values

male_train, male_to_index, female_to_index = utils.load_data_from_array(
    male_train_raw)
male_test, male_to_index, female_to_index = utils.load_data_from_array(
    male_test_raw, male_to_index, female_to_index)
female_train, female_to_index, male_to_index = utils.load_data_from_array(
    female_train_raw, female_to_index, male_to_index)
female_test, female_to_index, male_to_index = utils.load_data_from_array(
    female_test_raw, female_to_index, male_to_index)

male_bpr = bpr.BPR(rank=20, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=1)

male_bpr.train(male_train, epochs=500)

male_prediction = male_bpr.prediction_to_matrix()

female_bpr = bpr.BPR(rank=20, n_users=len(female_to_index),
              n_items=len(male_to_index), match_weight=1)

female_bpr.train(female_train, epochs=500)

female_prediction = female_bpr.prediction_to_matrix()

male_prediction_plus = male_prediction + female_prediction.T

def auc_test(prediction_mat, train_data, test_data):
    def _data_to_dict(self, data):
        data_dict = dict()
        items = set()
        for (user, item) in data:
            if user not in data_dict:
                data_dict
            data_dict[user].append(item)
            items.add(item)
        return data_dict, set(data_dict.keys()), items 
    train_pos_dict, train_users, train_items = _data_to_dict(train_data)    
    test_pos_dict, test_users, test_items = _data_to_dict(test_data)
    auc_values = []
    z = 0
    for user in test_users:
        auc_for_user = 0.0
        n = 0
        predictions = prediction_mat[user]
        pos_items = set(test_pos_dict[user]) - set(train_pos_dict[user])
        neg_items = train_items - set(test_pos_dict[user]) - set(train_pos_dict[user])
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
        if z % 10 == 0 and len(auc_values) > 0:
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.3f" % (str(z), np.mean(auc_values)))
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()
    return np.mean(auc_values)

auc_test(male_prediction_plus, male_train, male_test)
# def data_to_dict(training_data, min_rate):
#     train_dict = dict()
#     for row in training_data:
#         user, item, rate = row
#         if rate >= min_rate:
#             if user not in train_dict:
#                 train_dict[user] = dict()
#             train_dict[user][item] = 1
#     return train_dict

# train_dict = data_to_dict(training_data, 2)
# test_dict = data_to_dict(testing_data, 2)

# def evaluate(recommend_dict, lable_dict, train_dict, top=1000, mode='base'):
#     tp, fp, fn = 0, 0, 0
#     precision_recall_list = list()
#     for exp, job_rank_dict in recommend_dict.items():
#         if exp in lable_dict:
#             job_rank = sorted(job_rank_dict.items(),
#                               key=lambda x: x[1], reverse=True)
#             rec = [j_r[0] for j_r in job_rank[:top]]
#             rec_set = set(rec)
#             positive_set = set(lable_dict[exp].keys())
#             tp += len(rec_set & positive_set)
#             fp += len(rec_set - positive_set)
#             fn += len(positive_set - rec_set)
#             if mode == 'max':
#                 precision = 1 if rec_set & positive_set else 0
#                 recall = 1 if rec_set & positive_set else 0
#             else:
#                 precision = len(rec_set & positive_set) / len(rec_set)
#                 recall = len(rec_set & positive_set) / len(positive_set)
#             precision_recall_list.append([precision, recall])
#     if (mode == 'base') or (mode == 'max'):
#         df = pd.DataFrame(precision_recall_list, columns=[
#                           'precision', 'recall'])
#         return pd.DataFrame([df.mean(), df.std()], index=['mean', 'std'])
#     elif mode == 'sum':
#         return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp + fn))))

# precision_list, recall_list = [], []
# for k in range(1, 100, 5):
#     precision, recall = evaluate(prediction, test_dict, train_dict, top=k, mode='base').values[0]
#     precision_list.append(precision)
#     recall_list.append(recall)

# plt.scatter(precision_list, recall_list)

# def evaluate(recommend_dict, lable_dict, train_dict, top=1000, mode='base'):
#     tp, fp, fn = 0, 0, 0
#     precision_recall_list = list()
#     for exp, job_rank_dict in recommend_dict.items():
#         if exp in set(lable_dict.keys()) & set(train_dict.keys()):
#             job_rank = sorted(job_rank_dict.items(),
#                               key=lambda x: x[1], reverse=True)
#             rec = [j_r[0] for j_r in job_rank if j_r[0] not in train_dict[exp]][:top]
#             rec_set = set(rec)
#             positive_set = set(lable_dict[exp].keys()) - set(train_dict[exp].keys())
#             tp += len(rec_set & positive_set)
#             fp += len(rec_set - positive_set)
#             fn += len(positive_set - rec_set)
#             if len(positive_set) > 0:
#                 if mode == 'max':
#                     precision = 1 if rec_set & positive_set else 0
#                     recall = 1 if rec_set & positive_set else 0
#                 else:
#                     precision = len(rec_set & positive_set) / len(rec_set)
#                     recall = len(rec_set & positive_set) / len(positive_set)
#                 precision_recall_list.append([precision, recall])
#     if (mode == 'base') or (mode == 'max'):
#         df = pd.DataFrame(precision_recall_list, columns=[
#                           'precision', 'recall'])
#         return pd.DataFrame([df.mean(), df.std()], index=['mean', 'std'])
#     elif mode == 'sum':
#         return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp + fn))))

# precision_list, recall_list = [], []
# for k in range(1, 100, 5):
#     precision, recall = evaluate(prediction, test_dict, train_dict, top=k, mode='base').values[0]
#     precision_list.append(precision)
#     recall_list.append(recall)

# plt.scatter(precision_list, recall_list)
# plt.show()

# bpr.test(testing_data)

