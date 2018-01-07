# coding=utf-8

import numpy as np
import pandas as pd

class sampleGenerator(object):
    def __init__(self, n_user=1000, n_item=1000, n_feature=50, mu=0.5, sigma=0.1, hedge=0.26):
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
        self.u_i_link = self.generate_link(self.user_prefer, self.item_attr, self.hedge)
        self.i_u_link = self.generate_link(self.item_prefer, self.user_attr, self.hedge)
        # self.friendship = np.multiply(self.u_i_link, self.i_u_link)

    
    def random_matrix(self, row, column):
        return np.random.normal(
            loc=self.mu, scale=self.sigma, size=(row * column)).reshape(row, column)
    
    def generate_link(self, m1, m2, hedge):
        mat = np.matmul(m1, m2.T) / self.n_feature
        func = np.frompyfunc(lambda x: 1 if x > hedge else 0, 1, 1)
        return func(mat)

    def generate_sample(self):
        sample_list = list()
        m_f, f_m, friend =0, 0, 0
        for i in range(self.n_user):
            for j in range(self.n_item):
                if self.u_i_link[i, j] == 1:
                    sample_list.append(['m' + str(i), 'f' + str(j)])
                    m_f += 1
                if self.i_u_link[j, i] == 1:
                    sample_list.append(['f' + str(j), 'm' + str(i)])
                    f_m += 1
                if self.u_i_link[i, j] == 1 and self.i_u_link[j, i] == 1:
                    friend += 1
        print(m_f, f_m, friend)                
        return sample_list

def main():
    sample_generator = sampleGenerator()
    sample = sample_generator.generate_sample()
    frame = pd.DataFrame(sample)
    frame.to_csv('data.csv', columns=None, index=False)

if __name__ == '__main__':
    main()


