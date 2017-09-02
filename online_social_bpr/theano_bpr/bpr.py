# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import theano
import numpy
import theano.tensor as T
import theano_lstm
import time
import sys
from collections import defaultdict


class BPR(object):

    def __init__(self, rank, n_users, n_items, match_weight=2, lambda_all=0.01, learning_rate=0.1, sgd_weight=0.8):
        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) Matrix Factorisation 
          model, as described by Rendle et al. in:

            http://arxiv.org/abs/1205.2618

          This model tries to predict a ranking of items for each user
          from a viewing history.  
          It's also used in a variety of other use-cases, such
          as matrix completion, link prediction and tag recommendation.

          `rank` is the number of latent features in the matrix
          factorisation model.

          `n_users` is the number of users and `n_items` is the
          number of items.

          The regularisation parameters can be overridden using
          `lambda_u`, `lambda_i` and `lambda_j`. They correspond
          to each three types of updates.

          The learning rate can be overridden using `learning_rate`.

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/

          Example use (10 latent dimensions, 100 users, 50 items) for
          training:

          >>> from theano_bpr import BPR
          >>> bpr = BPR(10, 100, 50) 
          >>> from numpy.random import randint
          >>> train_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.train(train_data)

          This object also has a method for testing, which will return
          the Area Under Curve for a test set.

          >>> test_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.test(test_data)

          (This should give an AUC of around 0.5 as the training and
          testing set are chosen at random)
        """
        self._rank = rank
        self._match_weight = match_weight
        self._n_users = n_users
        self._n_items = n_items
        self._lambda_u = lambda_all
        self._lambda_i = lambda_all
        self._lambda_j = lambda_all
        self._lambda_bias = lambda_all
        self._learning_rate = learning_rate
        self._sgd_weight = sgd_weight
        self._train_users = set()
        self._train_items = set()
        self._match_dict = {}
        self._pos_dict = {}
        self._pos_ed_dict = {}
        self._configure_theano()
        self._generate_train_model_function()

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats. 
        """
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _generate_train_model_function(self):
        """
          Generates the train model function in Theano.
          This is a straight port of the objective function
          described in the BPR paper.

          We want to learn a matrix factorisation

            U = W.H^T

          where U is the user-item matrix, W is a user-factor
          matrix and H is an item-factor matrix, so that
          it maximises the difference between
          W[u,:].H[i,:]^T and W[u,:].H[j,:]^T, 
          where `i` is a positive item
          (one the user `u` has watched) and `j` a negative item
          (one the user `u` hasn't watched).
        """
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random(
            (self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random(
            (self._n_items, self._rank)).astype('float32'), name='H')

        self.B = theano.shared(numpy.zeros(
            self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal() + self.B[i]
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal() + self.B[j]

        x_uij = x_ui - x_uj

        obj_uij = T.sum(T.log(T.nnet.sigmoid(x_uij)) -
                        self._lambda_u * (self.W[u] ** 2).sum(axis=1) -
                        self._lambda_i * (self.H[i] ** 2).sum(axis=1) -
                        self._lambda_j * (self.H[j] ** 2).sum(axis=1) -
                        self._lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj_uij

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)
        sgd_updates = [(self.W, self.W - self._learning_rate * g_cost_W),
                       (self.H, self.H - self._learning_rate * g_cost_H),
                       (self.B, self.B - self._learning_rate * g_cost_B)]
        self.train_sgd = theano.function(
            inputs=[u, i, j], outputs=cost, updates=sgd_updates)

        ada_updates, gsums, xsums, lr, max_norm = theano_lstm.create_optimization_updates(
            cost, [self.W, self.H, self.B], method="adadelta")
        self.train_ada = theano.function(
            inputs=[u, i, j], outputs=cost, updates=ada_updates)

    def train(self, train_data, epochs=1, batch_size=100):
        """
          Trains the BPR Matrix Factorisation model using Stochastic
          Gradient Descent and minibatches over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.

          We first create a set of random samples from `train_data` for 
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples by
          batches of length `batch_size`, and run one iteration of gradient
          descent for the batch.
        """
        if len(train_data) < batch_size:
            sys.stderr.write(
                "WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            batch_size = len(train_data)
        self._match_dict, self._pos_dict, self._pos_ed_dict, self._train_users, self._train_items = self._data_to_dict(
            train_data)
        n_sgd_samples = len(self._train_users) * epochs
        sgd_users, sgd_pos_items, sgd_neg_items = self._uniform_user_sampling(
            n_sgd_samples)
        z = 0
        t2 = t1 = t0 = time.time()
        while (z + 1) * batch_size < n_sgd_samples * (self._sgd_weight):
            self.train_sgd(
                sgd_users[z * batch_size: (z + 1) * batch_size],
                sgd_pos_items[z * batch_size: (z + 1) * batch_size],
                sgd_neg_items[z * batch_size: (z + 1) * batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rsgd Processed %s ( %.2f%% ) in %.4f seconds" % (
                str(z * batch_size), 100.0 * float(z * batch_size) / n_sgd_samples, t2 - t1))
            sys.stderr.flush()

        while (z + 1) * batch_size < n_sgd_samples:
            self.train_ada(
                sgd_users[z * batch_size: (z + 1) * batch_size],
                sgd_pos_items[z * batch_size: (z + 1) * batch_size],
                sgd_neg_items[z * batch_size: (z + 1) * batch_size]
            )
            z += 1
            t2 = time.time()
            sys.stderr.write("\rada Processed %s ( %.2f%% ) in %.4f seconds" % (
                str(z * batch_size), 100.0 * float(z * batch_size) / n_sgd_samples, t2 - t1))
            sys.stderr.flush()

        if n_sgd_samples > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (
                t2 - t0, (t2 - t0) / n_sgd_samples))
            sys.stderr.flush()

    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, 
          and then sample a positive and a negative item for each 
          user sample.
        """
        sys.stderr.write(
            "Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self._train_users))[
            numpy.random.randint(len(self._train_users), size=n_samples)]
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            match_item = self._match_dict[sgd_user][
                numpy.random.randint(len(self._match_dict[sgd_user]))]
            pos_item = self._pos_dict[sgd_user][
                numpy.random.randint(len(self._pos_dict[sgd_user]))]
            pos_ed_item = self._pos_ed_dict[sgd_user][
                numpy.random.randint(len(self._pos_ed_dict[sgd_user]))]
            neg_item = numpy.random.randint(self._n_items)
            while neg_item in set(self._pos_dict[sgd_user]) | set(self._match_dict[sgd_user]) | set(self._pos_ed_dict[sgd_user]) :
                neg_item = numpy.random.randint(self._n_items)
            sgd_pos_items.extend([match_item, pos_item, pos_ed_item])
            sgd_neg_items.extend([pos_item, pos_ed_item, neg_item])
        return sgd_users, sgd_pos_items, sgd_neg_items

    def predictions(self, user_index):
        """
          Computes item predictions for `user_index`.
          Returns an array of prediction values for each item
          in the dataset.
        """
        w = self.W.get_value()
        h = self.H.get_value()
        b = self.B.get_value()
        user_vector = w[user_index, :]
        return user_vector.dot(h.T) + b

    def prediction(self, user_index, item_index):
        """
          Predicts the preference of a given `user_index`
          for a gven `item_index`.
        """
        return self.predictions(user_index)[item_index]


    def prediction_to_matrix(self):
        rank_lists = list()
        for user in range(self._n_users):
            rank_list = self.predictions(user)
            rank_lists.append(rank_list)
        return numpy.array(rank_lists)

    def prediction_to_dict(self, topn):
        rank_dict = dict()
        z = 0
        for user in range(self._n_users):
            rank_list = self.top_predictions(user, topn)
            try:
                rank_dict[user] = dict(rank_list)
            except:
                import pdb
                pdb.set_trace()
            z += 1
            if z % 10 == 0:
                sys.stderr.write("\rgenerate %d predictions" % z)
                sys.stderr.flush()            
        return rank_dict


    def top_predictions(self, user_index, topn=10):
        """
          Returns the item indices of the top predictions
          for `user_index`. The number of predictions to return
          can be set via `topn`.
          This won't return any of the items associated with `user_index`
          in the training set.
        """
        rank_list = enumerate(self.predictions(user_index))
        top_list = heapq.nlargest(topn, rank_list, key=lambda x: x[1])
        return top_list

    def test(self, test_data):
        """
          Computes the Area Under Curve (AUC) on `test_data`.

          `test_data` is an array of (user_index, item_index) tuples.

          During this computation we ignore users and items
          that didn't appear in the training data, to allow
          for non-overlapping training and testing sets.
        """
        test_match_dict, test_pos_dict, test_pos_ed_dict, test_users, test_items = self._data_to_dict(
            test_data)
        auc_values, auc_match_values, auc_pos_values = [], [], []
        z = 0
        for user in test_users & self._train_users:
            auc_for_user, auc_match, auc_pos = 0.0, 0.0, 0.0
            n, n_match, n_pos = 0, 0, 0
            predictions = self.predictions(user)
            match_items = set(
                test_match_dict[user]) & self._train_items - set(self._match_dict[user])
            pos_items = set(
                test_pos_dict[user]) & self._train_items - set(self._pos_dict[user])
            neg_items = self._train_items - match_items - pos_items - \
                set(self._pos_dict[user]) - set(self._match_dict[user])
            for match_item in match_items:
                for other_item in pos_items | neg_items:
                    n += 1
                    n_match += 1
                    if predictions[match_item] > predictions[other_item]:
                        auc_for_user += 1
                        auc_match += 1
                    elif predictions[match_item] == predictions[other_item]:
                        auc_for_user += 0.5
                        auc_match += 0.5
            for pos_item in match_items | pos_items:
                for neg_item in neg_items:
                    n += 1
                    n_pos += 1
                    if predictions[pos_item] > predictions[neg_item]:
                        auc_for_user += 1
                        auc_pos += 1
                    elif predictions[pos_item] == predictions[neg_item]:
                        auc_for_user += 0.5
                        auc_pos += 0.5
            if n > 0:
                auc_for_user /= n
                auc_values.append(auc_for_user)
            if n_match > 0:
                auc_match /= n_match
                auc_match_values.append(auc_match)
            if n_pos > 0:
                auc_pos /= n_pos
                auc_pos_values.append(auc_pos)
            z += 1
            if z % 10 == 0 and len(auc_values) > 0:
                sys.stderr.write("\rCurrent AUC mean (%s samples): %0.3f, %0.3f, %0.3f" % (
                    str(z), numpy.mean(auc_values), numpy.mean(auc_match_values), numpy.mean(auc_pos_values)))
                sys.stderr.flush()
        sys.stderr.write("\n")
        sys.stderr.flush()
        return numpy.mean(auc_values)

    def _data_to_dict(self, data):
        pos_dict = dict()
        match_dict = dict()
        pos_ed_dict = dict()
        all_items = set()
        for (user, item, rate) in data:
            if rate == 3:
                if user not in match_dict:
                    match_dict[user] = list()
                match_dict[user].append(item)
            elif rate == 2:
                if user not in pos_dict:
                    pos_dict[user] = list()
                pos_dict[user].append(item)
            elif rate == 1:
                if user not in pos_ed_dict:
                    pos_ed_dict[user] = list()
                pos_ed_dict[user].append(item)
            all_items.add(item)
        return match_dict, pos_dict, pos_ed_dict, (set(match_dict.keys()) & set(pos_dict.keys()) & set(pos_ed_dict.keys())), all_items
