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
from sklearn import preprocessing
# 本地库
import utils
import bpr

male_train_raw = pd.read_csv('input/train.csv', header=None).values
male_test_raw = pd.read_csv('input/test.csv', header=None).values

male_train, male_to_index, female_to_index = utils.load_data_from_array(
    male_train_raw)
male_test, male_to_index, female_to_index = utils.load_data_from_array(
    male_test_raw, male_to_index, female_to_index)

male_bpr = bpr.BPR(rank=10, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=1)

male_bpr.train(male_train, epochs=1000)

male_prediction = male_bpr.prediction_to_matrix()

def auc_test(prediction_mat, train_data, test_data, s=0.3):
    def _data_to_dict(data):
        pos_dict = dict()
        match_dict = dict()
        all_items = set()
        for (user, item, rate) in data:
            if rate == 3:
                if user not in match_dict:
                    match_dict[user] = list()
                match_dict[user].append(item)
            if rate == 2:
                if user not in pos_dict:
                    pos_dict[user] = list()
                pos_dict[user].append(item)
            all_items.add(item)
        return match_dict, pos_dict, (set(match_dict.keys()) & set(pos_dict.keys())), all_items
    train_match_dict, train_pos_dict, train_users, train_items = _data_to_dict(train_data)    
    test_match_dict, test_pos_dict, test_users, test_items = _data_to_dict(test_data)
    auc_values = []
    z = 0
    user_array = np.array(list(test_users & train_users))
    user_sample = user_array[np.random.randint(len(user_array), size=round(s * len(user_array)))]
# =============================================================================
#     user_sample = user_array
# =============================================================================
    for user in user_sample:
        auc_for_user = 0.0
        n = 0
        predictions = prediction_mat[user]
        match_items = set(test_match_dict[user]) & train_items - set(train_match_dict[user])
        pos_items = set(test_pos_dict[user]) & train_items - set(train_pos_dict[user])
        neg_items = train_items - match_items - pos_items - set(train_pos_dict[user]) - set(train_match_dict[user])
        for match_item in match_items:
            for other_item in pos_items | neg_items:
                n += 1
                if predictions[match_item] > predictions[other_item]:
                    auc_for_user += 1
                elif predictions[match_item] == predictions[other_item]:
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


auc_test(male_prediction, male_train, male_test)
