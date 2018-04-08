# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class sampleGenerator(object):
    def __init__(self, n_user=3000, n_item=3000, n_feature=50, mu=0.5, sigma=0.2, sparseness=0.01):
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
        like_hedge = np.percentile(m, 100-((self.sparseness) * 10)*100)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                qij = m[i, j]
                if qij > hedge:
                    friend_list.append(['m' + str(i), 'f' + str(j), 2])
                elif qij > like_hedge:
                    uij = self.user_rank[i, j]
                    iij = self.item_rank[i, j]
                    if uij > iij:
                        u_i_list.append(['m' + str(i), 'f' + str(j), 1])
                    elif iij > uij:
                        i_u_list.append(['m' + str(i), 'f' + str(j), 1])
        print('u-i sample'.ljust(20) + 'i-u sample'.ljust(20) + 'friend sample'.ljust(20))
        print(str(len(u_i_list)).ljust(20) + str(len(i_u_list)).ljust(20) + str(len(friend_list)).ljust(20))

# =============================================================================
#         u_i_list = pd.DataFrame(u_i_list).sample(n=4 * len(friend_list))
#         i_u_list = pd.DataFrame(i_u_list).sample(n=4 * len(friend_list))
# =============================================================================
                        
        return pd.DataFrame(friend_list), pd.DataFrame(u_i_list), pd.DataFrame(i_u_list)

class sampleCleaner():
    def __init__(self, friends, u_i, i_u):
        self._friends = friends
        self._uis = u_i
        self._ius = i_u

    def count_degree(self, frame, col):
        user = frame.columns[col]
        user_degree_series = frame.iloc[:, col]
        user_degree_frame = pd.DataFrame(user_degree_series.value_counts())
        user_degree_frame.columns = ['degree']
        user_degree_frame = pd.merge(frame, user_degree_frame,
                                    left_on=user, right_index=True)
        return user_degree_frame

    
    def filter_old(self, frame, N=0, M=100000):
        # 筛选老用户
        frame = self.count_degree(frame, 0)
        frame = self.count_degree(frame, 1)
        old_frame = frame[(frame['degree_x'] >= N) & (frame['degree_y'] >= N) & (frame['degree_x'] < M) & (frame['degree_y'] < M)]
        # print('rest users', len(set(old_frame.iloc[:, 0])))
        # print('rest items', len(set(old_frame.iloc[:, 1])))
        # print('rest matches', len(old_frame))
        return old_frame.iloc[:, :3]

    def iter_filter_old(self, N=0, M=100000, step=100):
        frame = self._friends
        for i in range(step):
            frame = self.filter_old(frame.iloc[:, :3], N, M)
            frame = self.count_degree(frame, 0)
            frame = self.count_degree(frame, 1)        
            if (frame['degree_x'].min() >= N and frame['degree_y'].min() >= N and
                frame['degree_x'].max() < M and frame['degree_y'].max() < M):
                print(frame.describe())
                break
        print('rest users', len(set(frame.iloc[:, 0])))
        print('rest items', len(set(frame.iloc[:, 1])))
        print('rest matches', len(frame))
        self._old_friends = frame.iloc[:, :3]
        return 

    def build_pos_data(self):
        # 只保留有匹配历史的用户
        uset = set(self._old_friends.iloc[:, 0])
        iset = set(self._old_friends.iloc[:, 1])
        print('nusers:', len(uset), 'nitems:', len(iset))
        self._old_uis = pd.DataFrame([row for row in self._uis.values if row[0] in uset and row[1] in iset])
        self._old_ius = pd.DataFrame([row for row in self._ius.values if row[0] in uset and row[1] in iset])
        print('match num:', len(self._old_friends), 
                'u i num:', len(self._old_uis), 
                'i u num', len(self._old_ius),
                'sparseness:', len(self._old_friends)/len(uset)/len(iset),
                len(self._old_uis)/len(uset)/len(iset), 
                len(self._old_ius)/len(uset)/len(iset))

        return self._old_friends, self._old_uis, self._old_ius

    def train_test_split(self, t_size=0.5, samples=None):
        friend_list, u_i_list, i_u_list = self.build_pos_data()
        friend_train, friend_test = train_test_split(friend_list, test_size=t_size)
        u_i_train, u_i_test = train_test_split(u_i_list, test_size=t_size)
        i_u_train, i_u_test = train_test_split(i_u_list, test_size=t_size)
        u_train = pd.concat([friend_train, u_i_train])
        u_test = pd.concat([friend_test, u_i_test])
        i_train = pd.concat([friend_train, i_u_train])
        i_test = pd.concat([friend_test, i_u_test])
        return u_train, u_test, i_train, i_test

    def sava_sample(self, t_size, save_path, samples=None):
        u_train, u_test, i_train, i_test = self.train_test_split(t_size)
        u_train.to_csv(save_path + 'male_train.csv', index=False)
        u_test.to_csv(save_path + 'male_test.csv', index=False)
        i_train.to_csv(save_path + 'female_train.csv', index=False)
        i_test.to_csv(save_path + 'female_test.csv', index=False)
        return  


if __name__ == '__main__':
    sample_generator = sampleGenerator(n_user=1000, n_item=1000, sparseness=0.001)
    f, u, i = sample_generator.generate_sample()
    sample_cleaner = sampleCleaner(f, u, i)
    sample_cleaner.iter_filter_old(N=3, M=100)
    sample_cleaner.sava_sample(t_size=0.5, save_path = '../data/')
                        

