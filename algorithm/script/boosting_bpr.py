from tqdm import tqdm
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
        self._train_users = set()
        self._train_items = set()
        self._match_dict = {}
        self._pos_dict = {}
        self._configure_theano()
        self._generate_train_model_function()

    def _configure_theano(self):
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _init_super_weight(self):
        if self._beta != None:
            self._beta = [self._beta] * self._n_users
            self._gama = [self._gama] * self._n_users
        else:
            self._beta = []
            self._gama = []
            for user in range(self._n_users):
                m = len(self._match_dict[user])
                p = len(self._pos_dict[user])
                self._beta.append(m /(m + p))
                self._gama.append(p /(m + p))
        self._beta = numpy.array(self._beta)
        self._gama = numpy.array(self._gama)
        return
        

    def _generate_train_model_function(self):
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')
        k = T.lvector('k')

        self.W = theano.shared(numpy.random.random(
            (self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random(
            (self._n_items, self._rank)).astype('float32'), name='H')

        self.B = theano.shared(numpy.zeros(
            self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + self.B[i]
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + self.B[j]
        x_uk = T.dot(self.W[u], self.H[k].T).diagonal() + self.B[k]

# 增加posi权重1=================================================================
        # x_uijk = 0.5 * T.log(T.nnet.sigmoid(x_uj - x_uk)) + T.log( T.nnet.sigmoid(x_ui - x_uk))
# =============================================================================
        
# 基本bpr======================================================================
        # x_uijk = T.log(T.nnet.sigmoid(x_ui - x_uk))        
# ============================================================================

# user 个性化权重 =============================================================
        # x_uijk = T.log(T.dot(beta, T.nnet.sigmoid(x_ui - x_uj)) + 
        #                T.dot(gama, T.nnet.sigmoid(x_uj - x_uk)) + 
        #                T.nnet.sigmoid(x_ui - x_uk))
# ============================================================================

# listwise损失=================================================================
        exp_x_ui = T.exp(x_ui)
        exp_x_uj = T.exp(x_uj)
        exp_x_uk = T.exp(x_uk)
        x_uijk = T.log(exp_x_ui/(exp_x_ui + exp_x_uj + exp_x_uk) * exp_x_uj/(exp_x_uj + exp_x_uk))
# =============================================================================

        # obj_uij = T.mean(x_uijk +
        #                 self._lambda * (self.W[u] ** 2).sum(axis=1) -
        #                 self._lambda * (self.H[i] ** 2).sum(axis=1) -
        #                 self._lambda * (self.H[j] ** 2).sum(axis=1) -
        #                 self._lambda * (self.H[k] ** 2).sum(axis=1) -
        #                 self._lambda * (self.B[i] ** 2 + self.B[j] ** 2 + self.B[k] ** 2))
        #                 # self._lambda * (self.B[i] ** 2 + self.B[k] ** 2))
        # cost = - obj_uij

        l2 = ((self.W[u] ** 2).sum(axis=1) +
            (self.H[i] ** 2).sum(axis=1) +
            (self.H[j] ** 2).sum(axis=1) +
            (self.H[k] ** 2).sum(axis=1) +
            (self.B[i] ** 2 + self.B[j] ** 2 + self.B[k] ** 2))
        

        cost = - T.sum(x_uijk - self._lambda * l2)

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)
        sgd_updates = [(self.W, self.W - self._learning_rate * g_cost_W),
                       (self.H, self.H - self._learning_rate * g_cost_H),
                       (self.B, self.B - self._learning_rate * g_cost_B)]
        self.train_sgd = theano.function(
            # inputs=[u, i, j, k, beta, gama], outputs=cost, updates=sgd_updates)
            inputs=[u, i, j, k], outputs=cost, updates=sgd_updates, on_unused_input='warn')
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

    def train(self, train_data, epochs=1, batch_size=100):
        if len(train_data) < batch_size:
            sys.stderr.write(
                "WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(train_data)
        self._match_dict, self._pos_dict, self._train_users, self._train_items = self._data_to_dict(
            train_data)
        self._init_super_weight()        
        n_sgd_samples = len(self._train_users) * epochs
        sgd_users, sgd_match_items, sgd_pos_items, sgd_neg_items = self._uniform_user_sampling(
            n_sgd_samples)
        z = 0
        t0 = time.time()
        print('\rsgd Processed')
        for z in tqdm(range(math.floor(n_sgd_samples * self._sgd_weight / batch_size - 1))):
            low, high = z * batch_size, (z + 1) * batch_size
            sgd_user = sgd_users[low: high]
            self.train_sgd(
                sgd_user,
                sgd_match_items[low: high],
                sgd_pos_items[low: high],
                sgd_neg_items[low: high],
                # self._beta[sgd_user], 
                # self._gama[sgd_user]
            )
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

        if n_sgd_samples > 0:
            t2 = time.time()
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (
                t2 - t0, (t2 - t0) / n_sgd_samples))
            sys.stderr.flush()

    def _uniform_user_sampling(self, n_samples):
        sys.stderr.write(
            "Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self._train_users))[
            numpy.random.randint(len(self._train_users), size=n_samples)]
        sgd_users_update, sgd_match_items, sgd_pos_items, sgd_neg_items = [], [], [], []
        for sgd_user in tqdm(sgd_users):
            # 生成三类item
            match_item = self._match_dict[sgd_user][numpy.random.randint(len(self._match_dict[sgd_user]))]
            pos_item = self._pos_dict[sgd_user][numpy.random.randint(len(self._pos_dict[sgd_user]))]
            dis_neg_pool = set(self._match_dict[sgd_user])
            dis_neg_pool |= set(self._pos_dict[sgd_user])
            neg_item = numpy.random.randint(self._n_items)
            while neg_item in dis_neg_pool:
                neg_item = numpy.random.randint(self._n_items)
            # 写入样本列表
            sgd_users_update.append(sgd_user)
            sgd_match_items.append(match_item)
            sgd_pos_items.append(pos_item)
            sgd_neg_items.append(neg_item)
        return sgd_users_update, sgd_match_items, sgd_pos_items, sgd_neg_items

    def predictions(self, user_index):
        w = self.W.get_value()
        h = self.H.get_value()
        b = self.B.get_value()
        user_vector = w[user_index, :]
        return user_vector.dot(h.T) + b


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

    def _data_to_dict(self, data):
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
