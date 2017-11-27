import sys
import json
import collections
import numpy
import pandas as pd
import datetime
import tqdm

train_file = 'input/filter_match_dict_train.json'
test_file = 'input/filter_match_dict_test.json'
neighbor_file = 'output/neighbor.json'


def read_file(file_name):
    with open(file_name, encoding='utf8', errors='ignore') as file:
        obj = json.loads(file.read())
    return obj


def save_dict(obj, file_dir):
    f = open(file_dir, mode='w', encoding='utf8', errors='ignore')
    s = json.dumps(obj, indent=4, ensure_ascii=False)
    f.write(s)
    f.close()


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
                self.W[u][_u] = len(item_set & _item_set) / (len(item_set) * len(_item_set)) ** (0.5)

    def recommend_one_user(self, user, K):
        rank = dict()
        action_item = self.train[user]
        for _u, w in self.W[user]:
            for i, r in self.train[user].items():
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
    def __init__(self, train):
        self.train = train
        self.ucf = UserBasedCF(self.train)
        self.icf = ItemBasedCF(self.train)

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
            self.recommend_one_user(user, K)
        return rec_dict

class SRI(object):
    """docstring for SRI"""
    def __init__(self, train_data, test_data, rec_data):
        self.train_data = train_data
        self.test_data = test_data
        self.rec_data = rec_data
        self.user_set = set(train_data.keys()) & set(test_data.keys())
        self.BSR()

    def BSR(self):
        sr = 0
        for u, i_r in tqdm.tqdm(self.test_data):
            if u not in self.user_set():
                continue
            sr += len(set(i_r) & self.user_set)
        sr /= len(self.user_set) ** 2
        self.bsr = sr

    def SRI(self, data, K):
        s = 0
        r = 0
        for u, rec in self.rec_data.items():
            if u not in self.user_set:
                continue
            k_rec = set(list(rec.keys())[:K])
            s += len(k_rec & set(self.test_data[u].keys()))
            r += len(k_rec & self.user_set)
        sr = s / r
        sri = sr / self.bsr
        return sri 


        



        
        



        
        
