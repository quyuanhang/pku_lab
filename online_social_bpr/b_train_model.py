# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:29:08 2017

@author: QYH
"""

import sys
sys.path.append('theano_bpr/')
from sklearn.model_selection import train_test_split
import utils
import bpr

with open('input/ratings.dat') as file:
    data = [row.split(',')[:2] for row in file if int(row.split(',')[2]) >= 3]
    data = data[:1000]
    data_train, data_test = train_test_split(
        data, test_size=0.2, random_state=0)

training_data, users_to_index, items_to_index = utils.load_data_from_array(
    data_train)
testing_data, users_to_index, items_to_index = utils.load_data_from_array(
    data_test, users_to_index, items_to_index)

bpr = bpr.BPR(10, len(users_to_index.keys()), len(users_to_index.keys()))

bpr.train(training_data, epochs=50)

# bpr.test(testing_data)

# prediction_dict_tmp = bpr.prediction_to_dict()


# import pandas as pd

# def data_to_dict(training_data):
#     train_dict = dict()
#     for row in training_data:
#         user, item = row
#         if user not in train_dict:
#             train_dict[user] = dict()
#         train_dict[user][item] = 1
#     return train_dict

# train_dict = data_to_dict(training_data)
# test_dict = data_to_dict(testing_data)

# def evaluate(recommend_dict, lable_dict, top=1000, mode='base', train_dict=0):
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
# return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp +
# fn))))

# print('标准测试')
# print(evaluate(prediction_dict_tmp, test_dict, 100, 'base', train_dict))