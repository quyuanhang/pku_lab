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

male_train_raw = pd.read_csv('input/male_train.csv', header=None).values
female_train_raw = pd.read_csv('input/female_train.csv', header=None).values

male_train_raw[:, 2] = np.array(list(map(lambda x: max(x // 50, 1), male_train_raw[:, 2])))
female_train_raw[:, 2] = np.array(list(map(lambda x: max(x // 50, 1), female_train_raw[:, 2])))

male_set = set(male_train_raw[:, 0]) & set(female_train_raw[:, 0])
female_set = set(male_train_raw[:, 1]) & set(female_train_raw[:, 1])
male_to_index = dict(zip(male_set, range(len(male_set))))
female_to_index = dict(zip(female_set, range(len(female_set))))

male_train, male_to_index, female_to_index = utils.load_data_from_array(
    male_train_raw, male_to_index, female_to_index)
female_train, male_to_index, female_to_index = utils.load_data_from_array(
    female_train_raw, male_to_index, female_to_index)

male_bpr = bpr.BPR(rank=50, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=1)

male_bpr.train(male_train, epochs=2000)

female_bpr = bpr.BPR(rank=50, n_users=len(male_to_index),
              n_items=len(female_to_index), match_weight=1)

female_bpr.train(female_train, epochs=2000)

male_prediction = male_bpr.prediction_to_matrix()
female_prediction = female_bpr.prediction_to_matrix()
male_prediction_plus = male_prediction + female_prediction

def auc_test(prediction_mat, train_data, test_data, s=0.3):
    def _data_to_dict(data):
        pos_dict = dict()
        match_dict = dict()
        all_items = set()
        for (user, item, rate) in data:
            if rate == 2:
                if user not in match_dict:
                    match_dict[user] = list()
                match_dict[user].append(item)
            if rate == 1:
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

male_test_raw = pd.read_csv('input/male_test.csv', header=None).values
female_test_raw = pd.read_csv('input/female_test.csv', header=None).values

male_test_raw[:, 2] = np.array(list(map(lambda x: max(x // 50, 1), male_test_raw[:, 2])))
female_test_raw[:, 2] = np.array(list(map(lambda x: max(x // 50, 1), female_test_raw[:, 2])))

male_test, male_to_index, female_to_index = utils.load_data_from_array(
    male_test_raw, male_to_index, female_to_index)
female_test, male_to_index, female_to_index = utils.load_data_from_array(
    female_test_raw, male_to_index, female_to_index)

auc_test(male_prediction, male_train, male_test)
auc_test(female_prediction, female_train, female_test)
auc_test(male_prediction_plus, male_train, male_test, 1)

# =============================================================================
# male_prediction_scale = preprocessing.scale(male_prediction, axis=1)
# female_prediction_scale = preprocessing.scale(female_prediction, axis=1)
# male_prediction_plus_scale = male_prediction_scale + female_prediction_scale
# 
# auc_test(male_prediction_scale, male_train, male_test)
# auc_test(female_prediction_scale, female_train, female_test)
# auc_test(male_prediction_plus_scale, male_train, male_test)
# =============================================================================

male_prediction_scale = np.argsort(np.argsort(male_prediction, axis=1))
female_prediction_scale = np.argsort(np.argsort(female_prediction, axis=1))
male_prediction_plus_scale = male_prediction_scale + female_prediction_scale

auc_test(male_prediction_scale, male_train, male_test)
auc_test(female_prediction_scale, female_train, female_test)
auc_test(male_prediction_plus_scale, male_train, male_test, 1)

def auc(p_array, test_y, split):
    positive_index = [i[0] for i in enumerate(test_y) if i[1] >= split]
    negative_index = [i[0] for i in enumerate(test_y) if i[1] < split]
    positive_score = p_array[positive_index]
    negative_score = p_array[negative_index]
    auc = 0.0
    for pos_s in positive_score:
        for neg_s in negative_score:
            if pos_s > neg_s:
                auc += 1
            if pos_s == neg_s:
                auc += 0.5
    auc /= (len(positive_score) * len(negative_score))
    return auc

with open('input/male_test.csv') as file:
    test_data_ = pd.read_csv(file, header=None).values
test_data = np.array([[male_to_index[i[0]], female_to_index[i[1]], i[2] // 100]
    for i in test_data_ if i[0] in male_to_index and i[1] in female_to_index])
p_array = np.array(list(map(lambda x: male_prediction_plus_scale[x[0], x[1]], test_data)))
test_y = test_data[:, 2]
print(auc(p_array, test_y, 1))

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

def evaluate(recommend_dict, lable_dict, train_dict, top=1000, mode='base', sam=0.3):
    tp, fp, fn = 0, 0, 0
    precision_recall_list = list()
    user_array = np.array(list(set(lable_dict.keys()) & set(train_dict.keys())))
    user_sample = user_array[np.random.randint(len(user_array), size=round(sam * len(user_array)))]
    for exp in user_sample:
        job_rank_dict = recommend_dict[exp]
        job_rank = sorted(job_rank_dict.items(),
                            key=lambda x: x[1], reverse=True)
        rec = [j_r[0] for j_r in job_rank if j_r[0] not in train_dict[exp]][:top]
        rec_set = set(rec)
        positive_set = set(lable_dict[exp].keys()) - set(train_dict[exp].keys())
        tp += len(rec_set & positive_set)
        fp += len(rec_set - positive_set)
        fn += len(positive_set - rec_set)
        if len(positive_set) > 0:
            if mode == 'max':
                precision = 1 if rec_set & positive_set else 0
                recall = 1 if rec_set & positive_set else 0
            else:
                precision = len(rec_set & positive_set) / len(rec_set)
                recall = len(rec_set & positive_set) / len(positive_set)
            precision_recall_list.append([precision, recall])
    if (mode == 'base') or (mode == 'max'):
        df = pd.DataFrame(precision_recall_list, columns=[
                          'precision', 'recall'])
        return pd.DataFrame([df.mean(), df.std()], index=['mean', 'std'])
    elif mode == 'sum':
        return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp + fn))))

precision_list, recall_list = [], []
for k in range(1, 100, 5):
    precision, recall = evaluate(pre_dict, test_dict, train_dict, top=k, mode='base').values[0]
    precision_list.append(precision)
    recall_list.append(recall)

plt.scatter(precision_list, recall_list)
plt.show()

# bpr.test(testing_data)

