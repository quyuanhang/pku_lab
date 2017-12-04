# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:31:44 2017

@author: QYH
"""

import sys
import time
import tensorflow as tf
import pandas as pd
import numpy as np

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


class CML(object):

    def __init__(self, n_users, n_items, embed_dim=50, master_learning_rate=0.1):

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.input1 = tf.placeholder(tf.int32, [None, 2])
        self.input2 = tf.placeholder(tf.int32, [None, 2])
        self.input1_score = tf.placeholder(tf.float32, [None, 1])
        self.input2_score = tf.placeholder(tf.float32, [None, 1])

        self.master_learning_rate = master_learning_rate

        self.W = tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                              stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        self.V = tf.Variable(tf.random_normal([n_items, embed_dim],
                                              stddev=1 / (embed_dim ** 0.5), dtype=tf.float32))

        self.sigma1 = tf.Variable(tf.random_normal(
            [self.embed_dim], dtype=tf.float32))
        self.sigma2 = tf.Variable(tf.random_normal(
            [self.embed_dim], dtype=tf.float32))

        self.W_1_1 = tf.nn.embedding_lookup(self.W, self.input1[:, 0])
        self.V_1_1 = tf.nn.embedding_lookup(self.V, self.input1[:, 1])
        self.loss_supvise1 = self.input1_score[:, 0]
        self.A1 = tf.multiply(tf.multiply(self.W_1_1, self.V_1_1), self.sigma1)
        self.A1 = tf.reduce_sum(self.A1, axis=1)
        self.loss1 = (tf.reduce_mean(
            tf.squared_difference(self.loss_supvise1, self.A1)))

        self.W_1_2 = tf.nn.embedding_lookup(self.W, self.input2[:, 0])
        self.V_1_2 = tf.nn.embedding_lookup(self.V, self.input2[:, 1])
        self.loss_supvise2 = self.input2_score[:, 0]
        self.A2 = tf.multiply(tf.multiply(self.W_1_2, self.V_1_2), self.sigma2)
        self.A2 = tf.reduce_sum(self.A2, axis=1)
        self.loss2 = (tf.reduce_mean(
            tf.squared_difference(self.loss_supvise2, self.A2)))

        self.loss = self.loss1 + self.loss2

        # self.optimize1 = tf.train.AdadeltaOptimizer(
        #     self.master_learning_rate).minimize(self.loss,var_list=[self.W])
        # self.optimize2 = tf.train.AdadeltaOptimizer(
        #     self.master_learning_rate).minimize(self.loss,var_list=[self.V])
        # self.optimize3 = tf.train.AdadeltaOptimizer(
        #     self.master_learning_rate).minimize(self.loss,var_list=[self.sigma1,self.sigma2])
        self.optimize = tf.train.AdadeltaOptimizer(
            self.master_learning_rate).minimize(self.loss)        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # def partial_fit1(self, X1, X2, Y1, Y2):
    #     opt, loss = self.sess.run((self.optimize1, self.loss), feed_dict={
    #                               self.input1: X1, self.input1_score: X2, self.input2: Y1, self.input2_score: Y2})
    #     return loss
    # def partial_fit2(self, X1, X2, Y1, Y2):
    #     opt, loss = self.sess.run((self.optimize2, self.loss), feed_dict={
    #                           self.input1: X1, self.input1_score: X2, self.input2: Y1, self.input2_score: Y2})
    #     return loss
    # def partial_fit3(self, X1, X2, Y1, Y2):
    #     opt, loss = self.sess.run((self.optimize3, self.loss), feed_dict={
    #                           self.input1: X1, self.input1_score: X2, self.input2: Y1, self.input2_score: Y2})
    #     return loss

    def partial_fit(self, X1, X2, Y1, Y2):
        opt, loss = self.sess.run((self.optimize, self.loss), feed_dict={
                              self.input1: X1, self.input1_score: X2, self.input2: Y1, self.input2_score: Y2})
        return loss

    def prediction(self, m, f):
        vec_m = self.W[m]
        vec_f = self.V[f]
        vec = tf.multiply(tf.multiply(vec_m, vec_f), self.sigma1)
        vec_ = tf.multiply(tf.multiply(vec_m, vec_f), self.sigma2)
        pr = (tf.reduce_sum(vec) + tf.reduce_sum(vec_)) / 2
        return self.sess.run(pr)

    def prediction_matrix(self):
        p = tf.matmul(tf.matmul(self.W, tf.diag(self.sigma1)), tf.transpose(self.V))
        p_ = tf.matmul(tf.matmul(self.W, tf.diag(self.sigma2)), tf.transpose(self.V))
        p = (p + p_) / 2
        return self.sess.run(p)

def print_schedule(begin, i, s_=None):
    if not s_:
        return 0
    if i % 1 == 0:
        sum_time = '%0.2f' % (time.time() - begin)
        sys.stderr.write(("\r%s %d sum time %s" % (s_, i, sum_time)))
        sys.stderr.flush()


def complete_schedual():
    sys.stderr.write("\n")
    sys.stderr.flush()


model = CML(n_users, n_items, 5, 1)

def optmiz(loss, step, words, begin):
    l_loss = 0
    i = 0
    stop = 0
    while i < step:
        c_loss = loss
        sum_time = '%0.2f' % (time.time() - begin)
        sys.stderr.write(("\r%s %0.1f %d sum time %s" % (words, c_loss, i, sum_time)))
        sys.stderr.flush()
        if abs(c_loss - l_loss) == 0:
            stop += 1
        else:
            stop = 0
        if stop >= 10:
            break
        i += 1
    return loss
# loss_funtion_v = model.partial_fit2(train_male[:, [0, 1]], train_male[:, [2]], 
#                     train_female[:, [0, 1]], train_female[:, [2]])
# loss_funtion_sigma = model.partial_fit3(train_male[:, [0, 1]], train_male[:, [2]], 
#                 train_female[:, [0, 1]], train_female[:, [2]])

begin = time.time()
l_loss = 0
i = 0
stop = 0
while i < 5000:
    c_loss = model.partial_fit(train_male[:, [0, 1]], train_male[:, [2]], 
                        train_female[:, [0, 1]], train_female[:, [2]])    
    if abs(c_loss - l_loss) == 0:
        stop += 1
    else:
        stop =0
    if stop >= 100:
        break
    l_loss = c_loss
    print_schedule(begin, i, ('loss function %0.1f' % c_loss))
    i += 1


with open('input/test.csv') as file:
    test_data_ = pd.read_csv(file, header=None).values[:1000, :]
test_data = np.array([[male_index_dict[i[0]], female_index_dict[i[1]], i[2] // 100]
    for i in test_data_ if i[0] in male_index_dict and i[1] in female_index_dict])

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

# p_array = [model.prediction(i[0], i[1]) for i in test_data]
mat = model.prediction_matrix()
p_array = np.array(list(map(lambda x: mat[x[0], x[1]], test_data)))
test_y = test_data[:, 2]
print(auc(p_array, test_y, 1))

def auc_test(prediction_mat, train_data, test_data, s=0.3):
    def _data_to_dict(data):
        pos_dict = dict()
        match_dict = dict()
        all_items = set()
        for (user, item, rate) in data:
            if rate == 100:
                if user not in match_dict:
                    match_dict[user] = list()
                match_dict[user].append(item)
            else:
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

test_male = pd.read_csv('input/male_test.csv', header=None).values
test_male = np.array([[male_index_dict[i[0]], female_index_dict[i[1]], i[2]]
    for i in test_male if i[0] in male_index_dict and i[1] in female_index_dict])
male_prediction = model.prediction_matrix()
auc_test(male_prediction, train_male, test_male)