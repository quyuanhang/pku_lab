import json
import collections
import pandas as pd
import datetime

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


def all_neighbor_dict(graph, stage):
    # 内部方法 获取一个用户的邻居
    def one_neighbor_dict(graph, user, stage):
        neighbor = dict()
        neighbor_not_in_train = dict()
        neighbor[user] = 1
        for s in range(stage):
            neighbor_tmp = dict()
            for user_, r in neighbor.items():
                for item, rui in graph[user_].items():
                    if item not in neighbor_tmp:
                        neighbor_tmp[item] = 0
                    neighbor_tmp[item] += r / len(graph[user_])
            neighbor = neighbor_tmp
        neighbor_not_in_train = dict()
        for key in neighbor:
            if key not in graph[user]:
                neighbor_not_in_train[key] = neighbor[key]
        if len(neighbor_not_in_train) > 0:
            return neighbor_not_in_train
    # 循环调用内部方法
    nd = dict()
    for user in graph:
        neighbor = one_neighbor_dict(graph, user, stage)
        if neighbor:
            nd[user] = neighbor
    return nd


def recommend(neighbor, top=0):
    recommend_dict = dict()
    for user_1 in neighbor:
        pr_list = list()
        for user_2 in neighbor[user_1]:
            if user_1 in neighbor[user_2]:
                rank = neighbor[user_1][user_2] * neighbor[user_2][user_1]
                pr_list.append([user_2, rank])
        if len(pr_list) > 0:
            top_list = sorted(pr_list, key=lambda x: x[1], reverse=True)[:top]
            recommend_dict[user_1] = collections.OrderedDict(top_list)
    return recommend_dict


def evaluate(recommend_dict, lable_dict, top=1000, mode='base', train_dict=0):
    tp, fp, fn = 0, 0, 0
    precision_recall_list = list()
    for exp, job_rank_dict in recommend_dict.items():
        if exp in lable_dict:
            job_rank = sorted(job_rank_dict.items(),
                              key=lambda x: x[1], reverse=True)
            rec = [j_r[0] for j_r in job_rank[:top]]
            rec_set = set(rec)
            positive_set = set(lable_dict[exp].keys())
            tp += len(rec_set & positive_set)
            fp += len(rec_set - positive_set)
            fn += len(positive_set - rec_set)
            if mode == 'max':
                precision = 1 if rec_set & positive_set else 0
                recall = 1 if rec_set & positive_set else 0
            else:
                precision = len(rec_set & positive_set) / len(rec_set)
                recall = len(rec_set & positive_set) / len(positive_set)
            precision_recall_list.append([precision, recall])
    if (mode == 'base') or (mode == 'max'):
        df = pd.DataFrame(precision_recall_list, columns=[
                          'precision', 'recall'])
        return pd.DataFrame([df.mean(), df.std()], index=['mean', 'std'])
    elif mode == 'sum':
        return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp + fn))))

begin = datetime.datetime.now()
# 评分过程
data_train = read_file(train_file)
neighbor = all_neighbor_dict(data_train, stage=5)
print(datetime.datetime.now() - begin)
prediction = recommend(neighbor, 5)
print(datetime.datetime.now() - begin)

# 测试过程
data_test = read_file(test_file)

print('标准测试')
print(evaluate(prediction, data_test, top=5,
               mode='base', train_dict=data_train))

print('总量测试')
print(evaluate(prediction, data_test, top=5,
               mode='sum', train_dict=data_train))
