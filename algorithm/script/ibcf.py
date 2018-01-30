#自建类
import collections
#第三方
import tqdm
import pandas as pd

import test


class ItemBasedCF:

    def __init__(self, train_data):
        self.train = train_data
        self.item_similarity()

    def item_similarity(self):
        # 建立物品-物品的共现矩阵
        C = dict()  # 物品-物品的共现矩阵
        N = dict()  # 物品被多少个不同用户购买
        for user, items in self.train.items():
            for i in items.keys():
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items.keys():
                    if i == j:
                        continue
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
        # 计算相似度矩阵
        self.W = dict()
        print('calculating item similarity')
        for i, related_items in tqdm.tqdm(C.items()):
            self.W.setdefault(i, {})
            for j, cij in related_items.items():
                self.W[i][j] = cij / ((N[i] * N[j]) ** 0.5)
        # return self.W

    # 给用户user推荐，前K个相关用户
    def recommend_one_user(self, user, K):
        rank = dict()
        action_item = self.train[user]  # 用户user产生过行为的item和评分
        for item, score in action_item.items():
            for j, wj in sorted(self.W[item].items(), key=lambda x: x[1], reverse=True)[0:K]:
                if j in action_item.keys():
                    continue
                rank.setdefault(j, 0)
                rank[j] += score * wj
        return collections.OrderedDict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:K])

    def recommend_all(self, K):
        rec_dict = dict()
        print('item based recommending')
        for user in tqdm.tqdm(self.train):
            rec_dict[user] = self.recommend_one_user(user, K)
        return rec_dict


class UserBasedCF(object):
    """docstring for UserBasedCF"""
    def __init__(self, train_data):
        self.train = train_data
        self.user_similarity()
    
    def user_similarity(self):
        print('calculating user similarity')        
        self.W = dict()
        for u, i_r in tqdm.tqdm(self.train.items()):
            for _u, _i_r in self.train.items():
                item_set = set(i_r.keys())
                _item_set = set(_i_r.keys())
                if not item_set & _item_set:
                    continue        
                self.W.setdefault(u, dict())
                self.W[u][_u] = len(item_set & _item_set) / (len(item_set) * len(_item_set)) ** (0.5)

    def recommend_one_user(self, user, K):
        rank = dict()
        action_item = self.train[user]
        for _u, w in self.W[user].items():
            for i, r in self.train[_u].items():
                if i not in action_item:
                    rank.setdefault(i, 0)
                    rank[i] += w * r
        return collections.OrderedDict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:K])

    def recommend_all(self, K):
        rec_dict = dict()
        print('user based recommending')
        for user in tqdm.tqdm(self.train):
            rec_dict[user] = self.recommend_one_user(user, K)
        return rec_dict


class IBCF(object):
    """docstring for IBCF"""
    def __init__(self, train, ucf=0, icf=0):
        self.train = self.data_format(train)
        if ucf == 0:
            self.ucf = UserBasedCF(self.train)
            self.icf = ItemBasedCF(self.train)
        else:
            self.ucf = ucf
            self.icf = icf

    def recommend_one_user(self, user, K):
        i_reco = self.icf.recommend_one_user(user, K)
        u_reco = self.ucf.recommend_one_user(user, K)
        candidate = set(i_reco.keys()) | set(u_reco.keys())
        rec_dict = dict()
        for c in candidate:
            i_rank = 0 if c not in i_reco else i_reco[c]
            u_rank = 0 if c not in u_reco else u_reco[c]
            rec_dict[c] = i_rank + u_rank
        rec_dict = collections.OrderedDict(sorted(rec_dict.items(), key=lambda x: x[1], reverse=True)[0:K])
        return rec_dict

    def recommend_all(self, K, user_list=0):
        if user_list == 0:
            user_list = self.train.keys()
        rec_dict = dict()
        print('ibcd recommending')
        for user in tqdm.tqdm(user_list):
            rec_dict[user] = self.recommend_one_user(user, K)
        return rec_dict

    @staticmethod
    def data_format(train_frame):
        def frame_to_dict(frame, user_index=0):
            match_dict = dict()
            for row in frame.iterrows():
                if user_index == 0:
                    user, item, rate = row[1]
                else:
                    item, user, rate = row[1]
                if user not in match_dict:
                    match_dict[user] = dict()
                match_dict[user][item] = rate
            return match_dict
        train_frame = train_frame[train_frame.iloc[:, 2]==2]
        train_data = frame_to_dict(train_frame, user_index=0)
        return train_data


if __name__ == '__main__':
    train_frame = pd.read_csv('../data/male_train.csv')
    my_ibcf = IBCF(train_frame)
    recommend = my_ibcf.recommend_all(100)

    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)

    auc = test.auc(train_dict, test_dict, recommend)
    print('auc:%0.2f'%(auc))

    precision_list, recall_list = test.precision_recall_list(
        recommend, test_dict, train_dict, range(5, 100, 5))
    frame = pd.DataFrame(precision_list + recall_list).T
    frame.index=['ibcf']
    test.p_r_curve(frame, line=True)
    
    



