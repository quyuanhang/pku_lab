from tqdm import tqdm
from collections import defaultdict
import heapq
import math
import time
import sys
import numpy
import theano
import theano.tensor as T
# =============================================================================
# import theano_lstm
# =============================================================================


class BPR(object):

    def __init__(self, rank, n_users, n_items, base_weight=1, match_weight=1, posi_weight=1, lambda_all=0.01, learning_rate=0.1, sgd_weight=1):
        self._rank = rank
        self._base_weight = base_weight
        self._beta = match_weight
        self._gama = posi_weight
        self._n_users = n_users
        self._n_items = n_items
        self._lambda = lambda_all
        self._learning_rate = learning_rate
        self._sgd_weight = sgd_weight
        self.configure_theano()
        self.generate_train_model_function()

    def configure_theano(self):
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'        

    def generate_train_model_function(self):
        u = T.lvector('u')
        v = T.lvector('v')
        w = T.lvector('w')
        i = T.lvector('i')
        j = T.lvector('j')
        k = T.lvector('k')
        

        self.W = theano.shared(numpy.random.random(
            (self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random(
            (self._n_items, self._rank)).astype('float32'), name='H')

        self.IB = theano.shared(numpy.zeros(
            self._n_items).astype('float32'), name='IB')

        self.UB = theano.shared(numpy.zeros(
            self._n_users).astype('float32'), name='UB')


        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + self.UB[u] + self.IB[i]
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + self.UB[u] + self.IB[j]
        x_uk = T.dot(self.W[u], self.H[k].T).diagonal() + self.UB[u] + self.IB[k]
        x_vi = T.dot(self.W[v], self.H[i].T).diagonal() + self.UB[v] + self.IB[i]
        x_wi = T.dot(self.W[w], self.H[i].T).diagonal() + self.UB[w] + self.IB[i]

# 增加posi权重1=================================================================
        x_uijk = (
# =============================================================================
#                 T.log(T.nnet.sigmoid(x_uj - x_uk)) * 0.5 + 
#                 T.log(T.nnet.sigmoid(x_vi - x_vi)) * 0.5 +
# =============================================================================
                T.log(T.nnet.sigmoid(x_ui - x_uk)) + 
                T.log(T.nnet.sigmoid(x_ui - x_wi)))
# =============================================================================
        
# listwise损失=================================================================
        # e_ui = T.exp(x_ui)
        # e_uj = T.exp(x_uj)
        # e_uk = T.exp(x_uk)
        # e_vi = T.exp(x_vi)
        # e_wi = T.exp(x_wi)
        # x_uijk = T.log(e_ui/(e_ui + e_uj + e_uk + e_vi + e_wi) * e_uj/(e_uj + e_uk) * e_vi/(e_vi + e_wi))        
# =============================================================================

        l2 = ((self.W[u] ** 2).sum(axis=1) +
            (self.H[i] ** 2).sum(axis=1) +
            (self.H[j] ** 2).sum(axis=1) +
            (self.H[k] ** 2).sum(axis=1) +
            self.UB[u] ** 2 + self.UB[v] ** 2 + self.UB[w] ** 2 +
            self.IB[i] ** 2 + self.IB[j] ** 2 + self.IB[k] ** 2)
        

        cost = - T.sum(x_uijk - self._lambda * l2)

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_IB = T.grad(cost=cost, wrt=self.IB)
        g_cost_UB = T.grad(cost=cost, wrt=self.UB)
        sgd_updates = [(self.W, self.W - self._learning_rate * g_cost_W),
                       (self.H, self.H - self._learning_rate * g_cost_H),
                       (self.UB, self.UB - self._learning_rate * g_cost_UB),
                       (self.IB, self.IB - self._learning_rate * g_cost_IB)]
        self.train_sgd = theano.function(
            # inputs=[u, i, j, k, beta, gama], outputs=cost, updates=sgd_updates)
            inputs=[u, v, w, i, j, k], outputs=cost, updates=sgd_updates, on_unused_input='warn')
            # inputs=[u, i, k], outputs=cost, updates=sgd_updates)

# =============================================================================
#         ada_updates, gsums, xsums, lr, max_norm = theano_lstm.create_optimization_updates(
#             cost, [self.W, self.H, self.B], method="adadelta")
#         self.train_ada = theano.function(
#             # inputs=[u, i, j, k, beta, gama], outputs=cost, updates=ada_updates)
#             inputs=[u, i, j, k], outputs=cost, updates=ada_updates, on_unused_input='warn')
#             # inputs=[u, i, k], outputs=cost, updates=ada_updates)
# =============================================================================

        return True

    def _data_to_dict(self, data):
        m_pos_dict, m_match_dict, f_pos_dict, f_match_dict = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        self._match_array = data[data[:, 2] == 2]
        for (user, item, rate) in self._match_array:
            m_match_dict[user].append(item)
            f_match_dict[item].append(user)
        self._posi_array = data[data[:, 2] == 1]
        for (user, item, rate) in self._posi_array:
            m_pos_dict[user].append(item)
            f_pos_dict[item].append(user)
        self._m_match_dict, self._m_pos_dict, self._f_match_dict, self._f_pos_dict = (
            m_match_dict, m_pos_dict, f_match_dict, f_pos_dict)
        return

    def _uniform_user_sampling(self, n_samples):
        sys.stderr.write(
            "Generating %s random training samples\n" % str(n_samples))
        
        sgd_ui = numpy.array(list(self._match_array))[
            numpy.random.randint(len(self._match_array), size=n_samples)]
        samples = []
        for u, i, r in tqdm(sgd_ui):
            # user行为关联的item
            un_neg_items = set(self._m_match_dict[u])
            if u in self._m_pos_dict:
                arr = self._m_pos_dict[u]
                j = arr[numpy.random.randint(len(arr))]
                un_neg_items |= set(arr)
            else:
                j = i
            k = numpy.random.randint(self._n_items)
            while k in un_neg_items:
                k = numpy.random.randint(self._n_items)
            # item行为关联的user
            un_neg_users = set(self._f_match_dict[i])
            if i in self._f_pos_dict:
                arr = self._f_pos_dict[i]
                v = arr[numpy.random.randint(len(arr))]
                un_neg_users |= set(arr)
            else:
                v = u
            w = numpy.random.randint(self._n_users)
            while w in un_neg_users:
                w = numpy.random.randint(self._n_users)
            # 写入样本列表
            samples.append([u, v, w, i, j, k])
        samples = numpy.array(samples)
        return samples


    def train(self, train_data, epochs=1, batch_size=100):
        if len(train_data) < batch_size:
            sys.stderr.write(
                "WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(train_data)
        self._data_to_dict(train_data)
        n_sgd_samples = self._n_users * epochs
        samples = self._uniform_user_sampling(n_sgd_samples)
        z = 0
        print('\rsgd Processed')
        for z in tqdm(range(math.floor(n_sgd_samples * self._sgd_weight / batch_size - 1))):
            low, high = z * batch_size, (z + 1) * batch_size            
            self.train_sgd(
                samples[low:high, 0], samples[low:high, 1], samples[low:high, 2], 
                samples[low:high, 3], samples[low:high, 4], samples[low:high, 5])
# =============================================================================
#         print('\rada Processed')
#         _z = z
#         for z in tqdm(range(_z, math.floor(n_sgd_samples / batch_size)-2)):
#             low, high = z * batch_size, (z + 1) * batch_size
#             sgd_user = sgd_users[low: high]
#             self.train_ada(
#                 sgd_user,
#                 sgd_match_items[low: high],
#                 sgd_pos_items[low: high],
#                 sgd_neg_items[low: high],
#                 # self._beta[sgd_user], 
#                 # self._gama[sgd_user]
#             )
# =============================================================================
        
        return


    def predictions(self, user_index):
        w = self.W.get_value()
        h = self.H.get_value()
        ib = self.IB.get_value()
        user_vector = w[user_index, :]
        return user_vector.dot(h.T) + ib


    def prediction_to_matrix(self):
        rank_lists = list()
        for user in range(self._n_users):
            rank_list = self.predictions(user)
            rank_lists.append(rank_list)
        return numpy.array(rank_lists)

    def prediction_to_dict(self, topn, iu, ii):
        rank_dict = dict()
        z = 0
        for user in range(self._n_users):
            rank_list = self.top_predictions(user, topn)
            rank_list = map(lambda x: (ii[x[0]], x[1]), rank_list)
            rank_dict[iu[user]] = dict(rank_list)
            z += 1
            if z % 10 == 0:
                sys.stderr.write("\rgenerate %d predictions" % z)
                sys.stderr.flush()            
        return rank_dict


    def top_predictions(self, user_index, topn=False):
        rank_list = enumerate(self.predictions(user_index))
        if not topn:
            top_list = rank_list
        else:
            top_list = heapq.nlargest(topn, rank_list, key=lambda x: x[1])
        return top_list

