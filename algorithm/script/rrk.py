import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

import test

class Ranking():
    def __init__(self, data, n_features=50, lambda_all=0.01, learning_rate=0.1, beta=None, gama=None):
        self._n_features = n_features
        self._lambda = lambda_all
        self._learning_rate = learning_rate
        self._beta = beta
        self._gama = gama
        self._match_dict, self._pos_dict = self.load_from_sparse_matrix(data)        


    def load_from_sparse_matrix(self, data):
        user_set = set(data[data[:, 2]==2, 0]) & set(data[data[:, 2]==1, 0])
        item_set = set(data[:, 1])

        self._n_users = len(user_set)
        self._users = list(range(self._n_users))
        self._n_items = len(item_set)
        self._items = list(range(self._n_items))
        
        self._user_index = dict(zip(user_set, list(range(len(user_set)))))
        self._index_user = dict(zip(list(range(len(user_set))), user_set))
        self._item_index = dict(zip(item_set, list(range(len(item_set)))))
        self._index_item = dict(zip(list(range(len(item_set))), item_set))

        pos_dict = dict()
        match_dict = dict()
        for (user, item, rate) in data:
            if not (user in self._user_index and item in self._item_index):
                continue
            user = self._user_index[user]
            item = self._item_index[item]
            if rate == 2:
                if user not in match_dict:
                    match_dict[user] = list()
                match_dict[user].append(item)
            if rate == 1:
                if user not in pos_dict:
                    pos_dict[user] = list()
                pos_dict[user].append(item)
        # for u in set(match_dict.keys()) - set(pos_dict.keys()):
        #     match_dict.pop(u)
        # for u in set(pos_dict.keys()) - set(match_dict.keys()):
        #     pos_dict.pop(u)
        return match_dict, pos_dict
    
    def uniform_user_sampling(self, n_samples):
        sgd_users = np.array(self._users)[
            np.random.randint(len(self._users), size=n_samples)]
        samples = list()
        # for sgd_user in tqdm(sgd_users):
        for sgd_user in sgd_users:            
            # 生成三类item
            match_item = self._match_dict[sgd_user][np.random.randint(len(self._match_dict[sgd_user]))]
            pos_item = self._pos_dict[sgd_user][np.random.randint(len(self._pos_dict[sgd_user]))]
            dis_neg_pool = set(self._match_dict[sgd_user]) | set(self._pos_dict[sgd_user])
            neg_item = np.random.randint(self._n_items)
            while neg_item in dis_neg_pool:
                neg_item = np.random.randint(self._n_items)
            # 写入样本列表
            samples.append([sgd_user, match_item, pos_item, neg_item])
        return np.array(samples)

    def init_super_weight(self):
        if self._beta != None:
            self._beta = tf.constant([self._beta] * self._n_users, dtype=tf.float32)
            self._gama = tf.constant([self._gama] * self._n_users, dtype=tf.float32)
        else:            
            beta = []
            gama = []
            for user in self._users:
                m = len(self._match_dict[user])
                p = len(self._pos_dict[user])
                beta.append(m /(m + p))
                gama.append(p /(m + p))
                beta.append
            self._beta = tf.constant(beta, dtype=tf.float32)
            self._gama = tf.constant(gama, dtype=tf.float32)
        return
        
    def init_ranking(self):
        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        j = tf.placeholder(tf.int32, [None])
        k = tf.placeholder(tf.int32, [None])

        self._user_emb_w = tf.Variable(tf.random_normal([self._n_users, self._n_features], stddev=1 / (self._n_features ** 0.5)))
        self._item_emb_w = tf.Variable(tf.random_normal([self._n_items, self._n_features], stddev=1 / (self._n_features ** 0.5)))
        self._item_b = tf.Variable(tf.zeros([self._n_items]))

        u_emb = tf.nn.embedding_lookup(self._user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(self._item_emb_w, i)
        i_b = tf.nn.embedding_lookup(self._item_b, i)
        j_emb = tf.nn.embedding_lookup(self._item_emb_w, j)
        j_b = tf.nn.embedding_lookup(self._item_b, j)
        k_emb = tf.nn.embedding_lookup(self._item_emb_w, k)
        k_b = tf.nn.embedding_lookup(self._item_b, k)
        
        beta = tf.nn.embedding_lookup(self._beta, u)
        gama = tf.nn.embedding_lookup(self._gama, u)

        qij = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)
        qjk = j_b - k_b + tf.reduce_sum(tf.multiply(u_emb, (j_emb - k_emb)), 1, keep_dims=True)
        qik = i_b - k_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - k_emb)), 1, keep_dims=True)

        # obj = tf.log(tf.sigmoid(qik)) + tf.multiply(beta, tf.log(tf.sigmoid(qij))) + tf.multiply(gama, tf.log(tf.sigmoid(qjk)))
        # obj = tf.log(tf.sigmoid(qik) + tf.multiply(beta, tf.sigmoid(qij)) + tf.multiply(gama, tf.sigmoid(qjk)))

        # l2 = tf.add_n([
        #     tf.reduce_sum(tf.square(u_emb)), 
        #     tf.reduce_sum(tf.square(i_emb)),
        #     tf.reduce_sum(tf.square(j_emb)),
        #     tf.reduce_sum(tf.square(k_emb))，
        #     tf.reduce_sum(tf.square(i_b)),
        #     tf.reduce_sum(tf.square(j_b)),
        #     tf.reduce_sum(tf.square(k_b))
        # ])

        obj = tf.log(tf.sigmoid(qik))
        l2 = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)), 
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(k_emb, k_emb)),
            tf.reduce_sum(tf.square(i_b)),
            tf.reduce_sum(tf.square(k_b))            
        ])

        ranking_loss = l2 * self._lambda - tf.reduce_mean(obj)        


        return u, i, j, k, ranking_loss


    def train_model(self, banch_size, steps, epoches):
        steps_per_epoche = steps // epoches
        self.init_super_weight()
        u, i, j, k, loss = self.init_ranking()

        sgd = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(loss)        
        ada = tf.train.AdamOptimizer().minimize(loss)

        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())        
        
        for epoch in tqdm(range(epoches)):
#        for epoch in range(epoches):
            for step in range(steps_per_epoche):
                sample_banch = self.uniform_user_sampling(banch_size)
                _sgd, _loss = self._session.run([sgd, loss], feed_dict={
                        u: sample_banch[:, 0],
                        i: sample_banch[:, 1],
                        j: sample_banch[:, 2],
                        k: sample_banch[:, 3]})
#            print('sgd epoche:', epoch, 'loss', _loss)

#        for epoch in tqdm(range(epoches // 2)):    
##        for epoch in range(epoches // 2):
#            for step in range(steps_per_epoche):
#                sample_banch = self.uniform_user_sampling(banch_size)
#                _ada, _loss = self._session.run([ada, loss], feed_dict={
#                        u: sample_banch[:, 0],
#                        i: sample_banch[:, 1],
#                        j: sample_banch[:, 2],
#                        k: sample_banch[:, 3]})
#            print('ada epoche:', epoch, 'loss', _loss)            
        
        return


    def predict(self, topn=False):
        if not topn:
            topn = self._n_items
        rank_matrix = tf.add(
            tf.matmul(
                self._user_emb_w, 
                tf.transpose(self._item_emb_w, perm=(1,0))),
            tf.tile([self._item_b], (self._n_users, 1)))
        top_values, top_index = self._session.run(tf.nn.top_k(rank_matrix, topn))
        rank_dict = dict()
        print('generating rrk prediction dict')
        for user in tqdm(self._users):
            rank_dict[self._index_user[user]] = dict(zip([self._index_item[item] for item in top_index[user]], top_values[user]))
        return rank_dict



if __name__ == '__main__':

    def rec_test(train_dict, test_dict, rank_dict, topn, auc_list, alg_name):
        precision_list, recall_list = test.precision_recall_list(
            rank_dict, test_dict, train_dict, range(5, topn, 5))
        frame = pd.DataFrame(precision_list + recall_list).T
        frame.index = [alg_name]
        auc = test.auc(train_dict, rank_dict, test_dict)
        auc_list.append(auc)
        return frame

    train_frame = pd.read_csv('../data/male_train.csv')  
    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)
    auc_list = []
    
    # algorithm = Ranking(train_frame.values, beta=0.1, gama=0.9)      
    # algorithm.train_model(banch_size=100, steps=5000, epoches=100)
    # alg_rec = algorithm.predict(topn=50)

    # f1 = rec_test(train_dict, test_dict, alg_rec, 50, auc_list, 'alg')

    base = Ranking(train_frame.values, beta=0, gama=0)
    base.train_model(banch_size=1000, steps=5000, epoches=100)
    base_rec = base.predict()

    f2 = rec_test(train_dict, test_dict, base_rec, 50, auc_list, 'base')
    test.p_r_curve(f2, point=True)

    # test.p_r_curve(pd.concat([f1, f2]), line=True, point=True)    
    