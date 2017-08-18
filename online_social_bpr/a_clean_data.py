# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:43:40 2017

@author: QYH
"""
import sys
import json
import time
import pandas as pd
# =============================================================================
# import matplotlib.pyplot as plt
# =============================================================================
from sklearn.cross_validation import train_test_split


# 数据文件 ==========================
rating_file = 'input/ratings.dat'
gender_file = 'input/gender.dat'
# 输出文件 ==========================
match_dict_file = 'input/match_dict.json'
filter_match_dict_file = 'input/filter_match_dict.json'
train_file = 'input/filter_match_dict_train.json'
test_file = 'input/filter_match_dict_test.json'


def read_dict(file_name):
    with open(file_name, encoding='utf8', errors='ignore') as file:
        obj = json.loads(file.read())
    return obj


def save_dict(obj, file_dir):
    f = open(file_dir, mode='w', encoding='utf8', errors='ignore')
    s = json.dumps(obj, indent=4, ensure_ascii=False)
    f.write(s)
    f.close()


def print_schedule(begin, i, s_=None):
    if not s_:
        return 0
    if i % 1000 == 0:
        sum_time = '%2f' % (time.time() - begin)
        sys.stderr.write(("\r%s %d sum time %s" % (s_, i, sum_time)))
        sys.stderr.flush()


def complete_schedual():
    sys.stderr.write("\n")
    sys.stderr.flush()


sys.stderr.write("\rruning\n")
sys.stderr.flush()

begin = time.time()

# 性别字典
with open(gender_file) as file:
    gender_dict = dict()
    for row in file:
        user_id, gender = row.strip().split(',')
        gender_dict[user_id] = gender

print('\rgender dict', time.time() - begin)


# 读取评分文件 构造字典
with open(rating_file) as file:
    male_rating_dict = dict()
    famale_rating_dict = dict()
    i = 0
    for row in file:
        user_id, item_id, rate = row.strip().split(',')
        rate = int(rate)
        if rate >= 5:
            user = gender_dict[user_id] + str(user_id)
            item = gender_dict[item_id] + str(item_id)
            if user[0] == 'M' and item[0] == 'F':
                # if user[0] == 'M':
                if user not in male_rating_dict:
                    male_rating_dict[user] = dict()
                male_rating_dict[user][item] = rate
            elif user[0] == 'F' and item[0] == 'M':
                # elif user[0] == 'F':
                if user not in famale_rating_dict:
                    famale_rating_dict[user] = dict()
                famale_rating_dict[user][item] = rate
            # else:
            #     print(user, item)
        print_schedule(begin, i, 'reading rating file')
        i += 1
        # if i > 500000:
        #     break
    complete_schedual()


print('rating dict', time.time() - begin)


# 从评分中构造匹配字典 匹配列表
match_dict = dict()
match_list = list()
female = 0
male = 0
i = 0
for user, item_rate_dict in male_rating_dict.items():
    for item, rate in item_rate_dict.items():
        if rate > 5:
            if item in famale_rating_dict:
                if user in famale_rating_dict[item]:
                    rate_ = famale_rating_dict[item][user]
                    if rate_ > 5:
                        if user not in match_dict:
                            match_dict[user] = dict()
                        match_dict[user][item] = rate * rate_
                        match_list.append([user, item, rate * rate_])
    print_schedule(begin, i, 'reading male dict')
    i += 1
complete_schedual()


# 构造匹配记录表
match_frame = pd.DataFrame(match_list, columns=['user', 'item', 'rate'])
print('users with matches', len(set(match_frame.iloc[:, 0])))


def filter_old(frame, N=0, M=100000):
    # 筛选老用户
    def count_degree(frame, col):
        try:
            user = frame.columns[col]
        except:
            import pdb
            pdb.set_trace()
        user_degree_series = frame.iloc[:, col]
        user_degree_frame = pd.DataFrame(user_degree_series.value_counts())
        user_degree_frame.columns = ['degree']
        user_degree_frame = pd.merge(frame, user_degree_frame,
                                     left_on=user, right_index=True)
        return user_degree_frame
    frame = count_degree(frame, 0)
    frame = count_degree(frame, 1)
    old_frame = frame[
        (frame['degree_x'] >= N) & (frame['degree_x'] <= M) &
        (frame['degree_y'] >= N) & (frame['degree_y'] <= M)]
    # print('delete', (len(set(frame.iloc[:, 0]) & set(frame.iloc[:, 1])) -
    #                  len(set(old_frame.iloc[:, 0]) & set(old_frame.iloc[:, 1]))), 'users')
    # print('delete', (len(frame) - len(old_frame)), 'matches')
    old_frame = count_degree(old_frame.iloc[:, :3], 0)
    old_frame = count_degree(old_frame, 1)
    return old_frame


def iter_filter_old(frame, N=0, M=100000, step=100):
    for i in range(step):
        frame = filter_old(frame.iloc[:, :3], N, M)
        if (frame['degree_x'].min() >= N and
                frame['degree_y'].min() >= N):
            break
    print('rest users', len(set(frame.iloc[:, 0])))
    print('rest items', len(set(frame.iloc[:, 1])))
    print('rest matches', len(frame))
    return frame.iloc[:, :3]


# 输出迭代消去的数据损失量
for i in [0, 1, 2, 3]:
    print('least match for old user', i)
    old_match_frame = iter_filter_old(match_frame, i)
    old_user_set = set(old_match_frame['user'])
    old_item_set = set(old_match_frame['item'])

# 根据匹配记录生成字典


def frame_to_dict(frame):
    match_dict = dict()
    for row in frame.iterrows():
        user, item, rate = row[1]
        if user not in match_dict:
            match_dict[user] = dict()
        match_dict[user][item] = rate
    return match_dict

male_match_dict = frame_to_dict(old_match_frame)

# 组合match和positive
positive_data = list()
i = 0
for user in old_user_set:
    items = male_rating_dict[user]
    for item in items:
        if item in old_item_set:
            if item in male_match_dict[user]:
                positive_data.append([user, item, 2])
            else:
                positive_data.append([user, item, 1])
    print_schedule(begin, i, 'combine match and positive users')
    i += 1
complete_schedual()


# 划分数据
data_train, data_test = train_test_split(
    pd.DataFrame(positive_data), test_size=0.2, random_state=0)

data_train.to_csv('input/train.csv', index=False, header=False)
data_test.to_csv('input/test.csv', index=False, header=False)


# 整理字典

# dict_train = frame_to_dict(data_train)
# dict_test = frame_to_dict(data_test)
# save_dict(dict_train, train_file)
# save_dict(dict_test, test_file)
