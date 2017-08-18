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
    rating_list = list()
    i = 0
    for row in file:
        user_id, item_id, rate = row.strip().split(',')
        rate = int(rate)
        if rate >= 5:
            user = gender_dict[user_id] + str(user_id)
            item = gender_dict[item_id] + str(item_id)
            if user[0] == 'M' and item[0] == 'F':
                rating_list.append([user, item])
        print_schedule(begin, i, 'reading rating file')
        i += 1
        # if i > 500000:
        #     break
    complete_schedual()


print('rating dict', time.time() - begin)


# 划分数据
data_train, data_test = train_test_split(
    pd.DataFrame(rating_list), test_size=0.2, random_state=0)

data_train.to_csv('input/train.csv', index=False, header=False)
data_test.to_csv('input/test.csv', index=False, header=False)


# 整理字典

# dict_train = frame_to_dict(data_train)
# dict_test = frame_to_dict(data_test)
# save_dict(dict_train, train_file)
# save_dict(dict_test, test_file)
