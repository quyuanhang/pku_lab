import heapq
from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf

import test


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

        self.optimize = tf.train.AdadeltaOptimizer(
            self.master_learning_rate).minimize(self.loss)        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def partial_fit(self, X1, X2, Y1, Y2):
        opt, loss = self.sess.run((self.optimize, self.loss), feed_dict={
                              self.input1: X1, self.input1_score: X2, self.input2: Y1, self.input2_score: Y2})
        return loss


    def prediction_matrix(self):
        p = tf.matmul(tf.matmul(self.W, tf.diag(self.sigma1)), tf.transpose(self.V))
        p_ = tf.matmul(tf.matmul(self.W, tf.diag(self.sigma2)), tf.transpose(self.V))
        p = (p + p_) / 2
        return self.sess.run(p)

class CSVD(object):
    def __init__(self, user_train_frame, item_train_frame):
        user_train_raw = user_train_frame.values
        item_train_raw = item_train_frame.values
        user_train_raw = user_train_raw[user_train_raw[:, 2]==2]
        user_list = list(set(user_train_raw[:, 0]))
        item_list = list(set(user_train_raw[:, 1]))
        self.nu = len(user_list)
        self.ni = len(item_list)
        user_to_index = dict(zip(user_list, range(self.nu)))
        item_to_index = dict(zip(item_list, range(self.ni)))
        self.user_to_index, self.item_to_index = user_to_index, item_to_index
        self.index_to_user = dict(zip(range(self.nu), user_list))
        self.index_to_item = dict(zip(range(self.ni), item_list))
        self.train_user = np.array([[user_to_index[i[0]], item_to_index[i[1]], i[2]]
            for i in user_train_raw if i[0] in user_to_index and i[1] in item_to_index])
        self.train_item = np.array([[user_to_index[i[0]], item_to_index[i[1]], i[2]] 
            for i in item_train_raw if i[0] in user_to_index and i[1] in item_to_index])
        self.model = CML(len(user_to_index), len(item_to_index)) 

    def train(self, steps=5000):
        l_loss = 0
        stop = 0
        for i in tqdm(range(steps)):
            c_loss = self.model.partial_fit(self.train_user[:, [0, 1]], self.train_user[:, [2]], 
                                self.train_item[:, [0, 1]], self.train_item[:, [2]])    
            if abs(c_loss - l_loss) == 0:
                stop += 1
            else:
                stop =0
            if stop >= 100:
                break
            l_loss = c_loss
        return True

    def predict(self, topn=False):
        if not topn:
            topn = self.ni
        rank_dict = dict()
        user_prediction = self.model.prediction_matrix()
        print('generating csvd prediction dict')
        for user in tqdm(range(self.nu)):
            rank_list = user_prediction[user]
            item_rank = map(lambda x: (self.index_to_item[x[0]], x[1]), enumerate(rank_list))
            if not topn:
                topn_item_rank = item_rank
            else:
                topn_item_rank = heapq.nlargest(topn, item_rank, key=lambda x: x[1])
            rank_dict[self.index_to_user[user]] = dict(topn_item_rank)
        return rank_dict

def main():
    pass

if __name__ == '__main__':
    user_train_frame = pd.read_csv('../data/male_train.csv')    
    item_train_frame = pd.read_csv('../data/female_train.csv')
    csvd = CSVD(user_train_frame, item_train_frame)
    csvd.train(steps=5000)
    csvd_rec = csvd.predict(100)

    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(user_train_frame, min_rate=2)

    auc = test.auc(train_dict, test_dict, csvd_rec)
    print('auc:%0.2f'%(auc))

    precision_list, recall_list = test.precision_recall_list(
        csvd_rec, test_dict, train_dict, range(5, 100, 5))
    frame = pd.DataFrame(precision_list + recall_list).T
    frame.index=['csvd']
    test.p_r_curve(frame, line=True, point=True)
