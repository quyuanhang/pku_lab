# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:43:40 2017

@author: QYH
"""

import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def statistic(ser):
    try:
        rating_user_count = ser.value_counts()
        rating_count_count = rating_user_count.value_counts()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(rating_user_count.values)
        ax2.plot(rating_count_count.values)
        print(rating_user_count.describe())
        print(rating_count_count.describe())
        return None
    except:
        return 1

begin = datetime.datetime.now()

# 性别字典
with open(gender_file) as file:
    gender_dict = dict()
    for row in file:
        user_id, gender = row.strip().split(',')
        gender_dict[user_id] = gender
 
print('gender dict', datetime.datetime.now() - begin)


# 读取评分文件 构造字典
with open(rating_file) as file:
    rating_dict = dict()
    for row in file:
        user_id, item_id, rate = row.strip().split(',')
        user = gender_dict[user_id] + str(user_id)
        item = gender_dict[item_id] + str(item_id)
        if user not in rating_dict:
            rating_dict[user] = dict()
        rating_dict[user][item] = int(rate)

print('rating dict', datetime.datetime.now() - begin)


# 从评分中构造匹配字典
match_dict = dict()
match_list = list()
female = 0
male = 0
i = 0
for user, item_rate_dict in rating_dict.items():
    # if i % 100 == 0:
    #     print(i, datetime.datetime.now() - begin)
    #     i += 1
    if user[0] == 'F':
        female += 1
    if user[0] == 'M':
        male += 1
    for item, rate in item_rate_dict.items():
        if rate > 5:
            if item in rating_dict:
                if user in rating_dict[item]:
                    rate_ = rating_dict[item][user]
                    if rate_ > 5:
                        if user not in match_dict:
                            match_dict[user] = dict()
                        match_dict[user][item] = rate * rate_
                        if item not in match_dict:
                            match_list.append([user, item, rate * rate_])
#==============================================================================
# save_dict(match_dict_file, match_dict)
#==============================================================================

# 构造匹配记录表
match_frame = pd.DataFrame(match_list, columns=['user_1', 'user_2', 'rate'])
print('users with matches', len(
    set(match_frame.iloc[:, 0]) | set(match_frame.iloc[:, 1])))


def filter_old(frame, N=0, M=100000):
    # 筛选老用户
    def count_degree(frame):
        degree_series = pd.concat([frame.iloc[:, 1], frame.iloc[:, 0]])
        degree_frame = pd.DataFrame(
            degree_series.value_counts(), columns=['degree'])
        user_1 = frame.columns[0]
        user_2 = frame.columns[1]
        user_degree_frame = pd.merge(frame, degree_frame,
                                     left_on=user_1, right_index=True)
        user_degree_frame = pd.merge(user_degree_frame, degree_frame,
                                     left_on=user_2, right_index=True)
        return user_degree_frame
    frame = count_degree(frame)
    old_frame = frame[
        (frame['degree_x'] >= N) & (frame['degree_x'] <= M) &
        (frame['degree_y'] >= N) & (frame['degree_y'] <= M)]
    # print('delete', (len(set(frame.iloc[:, 0]) & set(frame.iloc[:, 1])) -
    #                  len(set(old_frame.iloc[:, 0]) & set(old_frame.iloc[:, 1]))), 'users')
    # print('delete', (len(frame) - len(old_frame)), 'matches')
    old_frame = count_degree(old_frame.iloc[:, :3])
    return old_frame


def iter_filter_old(frame, N=0, M=100000, step=100):
    for i in range(step):
        frame = filter_old(frame.iloc[:, :3], N, M)
        if (frame['degree_x'].min() >= N and
                frame['degree_y'].min() >= N):
            break
    print('rest users', len(set(frame.iloc[:, 0]) | set(frame.iloc[:, 1])))
    print('rest matches', len(frame))
    return frame


# 输出迭代消去的数据损失量
for i in [10]:
    print('least match for old user', i)
    old_match_frame = iter_filter_old(match_frame, i)

# 统计指标
# user_match_count = pd.concat(
#     [old_match_frame['user_1'], old_match_frame['user_2']]).value_counts()


# 划分数据
data = old_match_frame.iloc[:, :2]
data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)

data_train.to_csv('input/train.csv', index=False, header=False)
data_test.to_csv('input/test.csv', index=False, header=False)


# 整理字典
# def frame_to_dict(frame):
#     match_dict = dict()
#     for row in frame.iterrows():
#         user_1, user_2, rate = row[1]
#         if user_1 not in match_dict:
#             match_dict[user_1] = dict()
#         if user_2 not in match_dict:
#             match_dict[user_2] = dict()
#         match_dict[user_1][user_2] = rate
#         match_dict[user_2][user_1] = rate
#     return match_dict

# dict_train = frame_to_dict(data_train)
# dict_test = frame_to_dict(data_test)
# save_dict(dict_train, train_file)
# save_dict(dict_test, test_file)
