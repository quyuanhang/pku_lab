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
from scipy import sparse

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
        sum_time = '%0.2f' % (time.time() - begin)
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

print('gender dict', time.time() - begin)
sys.stderr.flush()

# 读取评分文件 构造字典
with open(rating_file) as file:
    male_rating, female_rating = 0, 0
    male_rating_dict = dict()
    female_rating_dict = dict()
    i = 0
    for row in file:
        user_id, item_id, rate = row.strip().split(',')
        rate = int(rate)
        if rate >= 0:
            user = gender_dict[user_id] + str(user_id)
            item = gender_dict[item_id] + str(item_id)
            if user[0] == 'M' and item[0] == 'F':
                male_rating += 1
                # if user[0] == 'M':
                if user not in male_rating_dict:
                    male_rating_dict[user] = dict()
                male_rating_dict[user][item] = rate
            elif user[0] == 'F' and item[0] == 'M':
                female_rating += 1
                # elif user[0] == 'F':
                if item not in female_rating_dict:
                    female_rating_dict[item] = dict()
                female_rating_dict[item][user] = rate
            # else:
            #     print(user, item)
        print_schedule(begin, i, 'reading rating file')
        i += 1
    complete_schedual()
print('rating dict', time.time() - begin)


male_rating_frame = pd.DataFrame(male_rating_dict)
female_rating_frame = pd.DataFrame(female_rating_dict)

inter_female = list(set(male_rating_frame.index) & set(female_rating_frame.index))
inter_male = list(set(male_rating_frame.columns) & set(female_rating_frame.columns))

male_rating_frame = male_rating_frame.reindex(index=inter_female, columns=inter_male)
female_rating_frame = female_rating_frame.reindex(index=inter_female, columns=inter_male)

male_rating_frame_adj = male_rating_frame.fillna(0) * female_rating_frame.fillna(1)
female_rating_frame_adj = female_rating_frame.fillna(0) * male_rating_frame.fillna(1)

