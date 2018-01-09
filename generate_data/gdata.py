# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class sampleGenerator(object):
    def __init__(self, n_user=1000, n_item=1000, n_feature=50, mu=0.5, sigma=0.2, hedge=0.26):
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.mu = mu
        self.sigma = sigma
        self.hedge = hedge
        self.user_attr = self.random_matrix(self.n_user, self.n_feature)
        self.user_prefer = self.random_matrix(self.n_user, self.n_feature)
        self.item_attr = self.random_matrix(self.n_item, self.n_feature)
        self.item_prefer = self.random_matrix(self.n_item, self.n_feature)
        self.noise = np.random.normal(0, 0.01, self.n_user*self.n_item).reshape(self.n_user, self.n_item)
        self.u_i_link = self.generate_link(self.user_prefer, self.item_attr, self.hedge)
        self.i_u_link = self.generate_link(self.item_prefer, self.user_attr, self.hedge)
        # self.friendship = np.multiply(self.u_i_link, self.i_u_link)

    
    def random_matrix(self, row, column):
        return np.random.normal(
            loc=self.mu, scale=self.sigma, size=(row * column)).reshape(row, column)
    
    def generate_link(self, m1, m2, hedge):
        mat = (np.matmul(m1, m2.T)) / self.n_feature + self.noise
        func = np.frompyfunc(lambda x: 1 if x > hedge else 0, 1, 1)
        return func(mat)

    def generate_sample(self):
        friend_list = list()
        u_i_list = list()
        i_u_list = list()
        m_f, f_m, friend =0, 0, 0
        for i in range(self.n_user):
            for j in range(self.n_item):
                u_like_i = (self.u_i_link[i, j] == 1)
                i_like_u = (self.i_u_link[j, i] == 1)
                if u_like_i and i_like_u:
                    friend += 1
                    friend_list.append(['m' + str(i), 'f' + str(j), 2])
                elif u_like_i:
                    m_f += 1
                    u_i_list.append(['m' + str(i), 'f' + str(j), 1])
                elif i_like_u:
                    f_m += 1
                    i_u_list.append(['m' + str(i), 'f' + str(j), 1])
        print(m_f, f_m, friend)                
        return u_i_list, i_u_list, friend_list

if __name__ == '__main__':
    sample_generator = sampleGenerator()
    u_i_list, i_u_list, friend_list = sample_generator.generate_sample()
    friend_train, friend_test = train_test_split(friend_list, test_size=0.2)
    u_i_train, u_i_test = train_test_split(u_i_list, test_size=0.2)
    i_u_train, i_u_test = train_test_split(i_u_list, test_size=0.2)
    pd.DataFrame(friend_train + u_i_train).to_csv('../public_data/male_train.csv', index=False, columns=None)
    pd.DataFrame(friend_test + u_i_test).to_csv('../public_data/male_test.csv', index=False, columns=None)
    pd.DataFrame(friend_train + i_u_train).to_csv('../public_data/female_train.csv', index=False, columns=None)
    pd.DataFrame(friend_test + i_u_test).to_csv('../public_data/female_test.csv', index=False, columns=None)
    
