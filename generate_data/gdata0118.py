# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class sampleGenerator(object):
    def __init__(self, n_user=1000, n_item=1000, n_feature=50, mu=0.5, sigma=0.2, sparseness=0.01):
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.mu = mu
        self.sigma = sigma
        self.sparseness = sparseness
        self.user_attr = self.random_matrix(self.n_user, self.n_feature)
        self.item_attr = self.random_matrix(self.n_item, self.n_feature)
        self.user_prefer = self.random_matrix(self.n_user, self.n_feature)
        self.item_prefer = self.random_matrix(self.n_item, self.n_feature)        
        self.noise = np.random.normal(loc=0, scale=0.1, size=(self.n_user, self.n_item)) 
        self.q_matrix = np.matmul(self.user_prefer, self.item_attr.T) + np.matmul(self.item_prefer, self.user_attr.T) + self.noise

    def random_matrix(self, row, column):
        # return np.random.normal(
            # loc=self.mu, scale=self.sigma, size=(row * column)).reshape(row, column)
        return np.random.randint(low=0, high=2, size=(row, column))

    def generate_sample(self):
        m = self.q_matrix
        friend_list, u_i_list, i_u_list = list(), list(), list()
        match_hedge = np.percentile(m, 100-self.sparseness*100)
        like_hedge = np.percentile(m, 100-((self.sparseness)**(1/2))*100)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                qij = m[i, j]
                if qij > match_hedge:
                    friend_list.append(['m' + str(i), 'f' + str(j), 2])
                elif qij > like_hedge:
                    if np.random.rand() < 0.5:
                        u_i_list.append(['m' + str(i), 'f' + str(j), 1])
                    else:
                        i_u_list.append(['m' + str(i), 'f' + str(j), 1])
        return pd.DataFrame(friend_list), pd.DataFrame(u_i_list), pd.DataFrame(i_u_list)

if __name__ == '__main__':
    sample_generator = sampleGenerator(sparseness=0.001)
    friend_list, u_i_list, i_u_list = sample_generator.generate_sample()
    # u_i_list_tmp, i_u_list_tmp, friend_list_tmp = sample_generator.generate_sample(0.5, 0.28, 0.25)
    # sample_filter = sampleFilter(u_i_list_tmp, i_u_list_tmp, friend_list_tmp)
    # u_i_list, i_u_list, friend_list = sample_filter._filter(hedge=3)
    friend_train, friend_test = train_test_split(friend_list, test_size=0.5)
    u_i_train, u_i_test = train_test_split(u_i_list, test_size=0.5)
    i_u_train, i_u_test = train_test_split(i_u_list, test_size=0.5)
    pd.concat([friend_train, u_i_train]).to_csv('../public_data/male_train.csv', index=False, columns=None)
    pd.concat([friend_test, u_i_test]).to_csv('../public_data/male_test.csv', index=False, columns=None)
    pd.concat([friend_train, i_u_train]).to_csv('../public_data/female_train.csv', index=False, columns=None)
    pd.concat([friend_test, i_u_test]).to_csv('../public_data/female_test.csv', index=False, columns=None)
                        

