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
        self.user_attr = self.random_matrix(self.n_user, self.n_feature, one_hot=True)
        self.item_attr = self.random_matrix(self.n_item, self.n_feature, one_hot=True)
        self.user_prefer = self.random_matrix(self.n_user, self.n_feature, one_hot=True)
        self.item_prefer = self.random_matrix(self.n_item, self.n_feature, one_hot=True)        
        self.noise = np.random.normal(loc=0, scale=0.1, size=(self.n_user, self.n_item)) 
        self.user_rank = np.matmul(self.user_prefer, self.item_attr.T)
        self.item_rank = np.matmul(self.item_prefer, self.user_attr.T)
        self.q_matrix = (self.user_rank + self.item_rank) / 2 + self.noise

    def random_matrix(self, row, column, one_hot=False):
        # return np.random.normal(loc=self.mu, scale=self.sigma, size=(row * column)).reshape(row, column)
        if one_hot:
            return np.random.randint(low=0, high=2, size=(row, column))
        else:
            return np.random.rand(row, column)

    def generate_sample(self):
        m = self.q_matrix.copy()
        m[self.user_rank < self.mu] = 0
        m[self.item_rank < self.mu] = 0        
        friend_list, u_i_list, i_u_list = list(), list(), list()
        hedge = np.percentile(m, 100-self.sparseness*100)
        like_hedge = np.percentile(m, 100-((self.sparseness)**(1/4))*100)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                qij = m[i, j]
                if qij > hedge:
                    friend_list.append(['m' + str(i), 'f' + str(j), 2])
                # elif qij > like_hedge:
                else:
                    uij = self.user_rank[i, j]
                    iij = self.item_rank[i, j]
                    if (uij > hedge):
                        u_i_list.append(['m' + str(i), 'f' + str(j), 1])
                    elif (iij > hedge):
                        i_u_list.append(['m' + str(i), 'f' + str(j), 1])
        print('u-i sample'.ljust(20) + 'i-u sample'.ljust(20) + 'friend sample'.ljust(20))
        print(str(len(u_i_list)).ljust(20) + str(len(i_u_list)).ljust(20) + str(len(friend_list)).ljust(20))
                        
        return pd.DataFrame(friend_list), pd.DataFrame(u_i_list), pd.DataFrame(i_u_list)

    def train_test_split(self, t_size=0.5, samples=None):
        if samples:
            friend_list, u_i_list, i_u_list = samples
        else:
            friend_list, u_i_list, i_u_list = self.generate_sample()
        friend_train, friend_test = train_test_split(friend_list, test_size=t_size)
        u_i_train, u_i_test = train_test_split(u_i_list, test_size=t_size)
        i_u_train, i_u_test = train_test_split(i_u_list, test_size=t_size)
        u_train = pd.concat([friend_train, u_i_train])
        u_test = pd.concat([friend_test, u_i_test])
        i_train = pd.concat([friend_train, i_u_train])
        i_test = pd.concat([friend_test, i_u_test])
        return u_train, u_test, i_train, i_test

    def sava_sample(self, t_size, save_path, samples=None):
        if samples:
            u_train, u_test, i_train, i_test = samples
        else:
            u_train, u_test, i_train, i_test = self.train_test_split(t_size)
        u_train.to_csv(save_path + 'male_train.csv', index=False)
        u_test.to_csv(save_path + 'male_test.csv', index=False)
        i_train.to_csv(save_path + 'female_train.csv', index=False)
        i_test.to_csv(save_path + 'female_test.csv', index=False)
        return True        

if __name__ == '__main__':
    sample_generator = sampleGenerator(sparseness=0.001)
    sample_generator.sava_sample(t_size=0.5, save_path = '../data/')
                        

