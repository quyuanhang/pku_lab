# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:31:44 2017

@author: QYH
"""

import tensorflow as tf

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