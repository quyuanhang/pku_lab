# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class sampleGenerator(object):
    def __init__(self, n_user=1000, n_item=10000, n_feature=50, mu=0.5, sigma=0.2, hedge=0.28):
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
        self.u_i_rank, self.u_i_link = self.generate_link(self.user_prefer, self.item_attr, self.hedge)
        self.i_u_rank, self.i_u_link = self.generate_link(self.item_prefer, self.user_attr, self.hedge)
        # self.friendship = np.multiply(self.u_i_link, self.i_u_link)

    
    def random_matrix(self, row, column):
        return np.random.normal(
            loc=self.mu, scale=self.sigma, size=(row * column)).reshape(row, column)
    
    def generate_link(self, m1, m2, hedge):
        noise = np.random.normal(0, 0.01, self.n_user*self.n_item).reshape(m1.shape[0], m2.shape[0])
        mat = (np.matmul(m1, m2.T)) / self.n_feature + noise
# =============================================================================
#         mat = (np.matmul(m1, m2.T)) / self.n_feature
# =============================================================================
        func = np.frompyfunc(lambda x: 1 if x > hedge else 0, 1, 1)
        return mat, func(mat)

    def generate_sample(self, sample_rate, invite=0.28, accept=0.26):
        friend_list = list()
        u_i_list = list()
        i_u_list = list()
        m_f, f_m, friend =0, 0, 0
        for i in range(self.n_user):
            for j in range(self.n_item):
                r = np.random.rand()
                if r > sample_rate:
                    continue                
                u_invite_i = (self.u_i_rank[i, j] >= invite)
                u_accept_i = (self.u_i_rank[i, j] >= accept)
                i_invite_u = (self.i_u_rank[j, i] >= invite)
                i_accept_u = (self.i_u_rank[j, i] >= accept)
                if (u_invite_i and i_accept_u) or (i_invite_u and u_accept_i) :
                    friend += 1
                    friend_list.append(['m' + str(i), 'f' + str(j), 2])
                elif u_invite_i:
                    m_f += 1
                    u_i_list.append(['m' + str(i), 'f' + str(j), 1])
                elif i_invite_u:
                    f_m += 1
                    i_u_list.append(['m' + str(i), 'f' + str(j), 1])
                elif r < 0.00001:
                    friend += 1
                    friend_list.append(['m' + str(i), 'f' + str(j), 2])

        print(m_f, f_m, friend, friend/(m_f*f_m))                
        return pd.DataFrame(u_i_list), pd.DataFrame(i_u_list), pd.DataFrame(friend_list)

class sampleFilter(object):
    def __init__(self, u_i_list, i_u_list, friend_list):
        self.u_i_list = u_i_list
        self.i_u_list = i_u_list
        self.friend_list = friend_list
        self.friend_num = self.friend_statistic()
        self.u_i_friend_with_num = pd.merge(pd.DataFrame(self.friend_list), self.friend_num, left_on=0, right_index=True)
        self.u_i_with_num = pd.merge(pd.DataFrame(self.u_i_list), self.friend_num, left_on=0, right_index=True)
        self.i_u_with_num = pd.merge(pd.DataFrame(self.i_u_list), self.friend_num, left_on=0, right_index=True)

    def friend_statistic(self):
        friend_frame = pd.DataFrame(self.friend_list)
        user_friend_num = pd.DataFrame(friend_frame.iloc[:, 0].value_counts())
        user_friend_num.columns = ['friend_num']
        return user_friend_num

    def _filter(self, hedge):
        friend = self.u_i_friend_with_num[self.u_i_friend_with_num['friend_num']>=hedge].iloc[:, :-1]
        iu = self.i_u_with_num[self.i_u_with_num['friend_num']>=hedge].iloc[:, :-1]
        ui = self.u_i_with_num[self.u_i_with_num['friend_num']>=hedge].iloc[:, :-1]
        print(len(ui), len(iu), len(friend), len(friend)/(len(ui)*len(iu)))
        return ui, iu, friend


if __name__ == '__main__':
    sample_generator = sampleGenerator()
    # u_i_list_tmp, i_u_list_tmp, friend_list_tmp = sample_generator.generate_sample(1)
    # sample_filter = sampleFilter(u_i_list_tmp, i_u_list_tmp, friend_list_tmp)
    # u_i_list, i_u_list, friend_list = sample_filter._filter(hedge=3)
    u_i_list, i_u_list, friend_list = sample_generator.generate_sample(0.5, 0.3, 0.25)
    friend_train, friend_test = train_test_split(friend_list, test_size=0.5)
    u_i_train, u_i_test = train_test_split(u_i_list, test_size=0.5)
    i_u_train, i_u_test = train_test_split(i_u_list, test_size=0.5)
    pd.concat([friend_train, u_i_train]).to_csv('../public_data/male_train.csv', index=False, columns=None)
    pd.concat([friend_test, u_i_test]).to_csv('../public_data/male_test.csv', index=False, columns=None)
    pd.concat([friend_train, i_u_train]).to_csv('../public_data/female_train.csv', index=False, columns=None)
    pd.concat([friend_test, i_u_test]).to_csv('../public_data/female_test.csv', index=False, columns=None)
    
