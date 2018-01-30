'''
    主函数
'''

import time
import os

import pandas as pd

from script import gdata
from script import test
from script.ibcf import IBCF
from script.algorithm import Algorithm
from script.csvd import CSVD

def step():
    sample_generator = gdata.sampleGenerator(n_user=1000, n_item=1000, sparseness=0.001)
    # u_train, u_test, i_train, i_test = sample_generator.train_test_split(t_size=0.5)
    # sample_generator.sava_sample(t_size=0.5, save_path='data/', 
    #     samples=(u_train, u_test, i_train, i_test))
    sample_generator.sava_sample(t_size=0.5, save_path='data/')
    train_frame = pd.read_csv('data/male_train.csv')
    test_frame = pd.read_csv('data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)
    topn = 100
    auc_list = list()

    # ibcf
    my_ibcf = IBCF(train_frame)
    ibcf_rec = my_ibcf.recommend_all(topn)
    precision_list, recall_list = test.precision_recall_list(
        ibcf_rec, test_dict, train_dict, range(5, topn, 5))
    ibcf_frame = pd.DataFrame(precision_list + recall_list).T
    ibcf_frame.index=['ibcf']
    auc_list.append(0)

    # algorithm
    algorithm = Algorithm(train_frame, mweight=1, pweight=1, epochs=10000)
    alg_rec = algorithm.predict(mode='dict', top=topn)
    precision_list, recall_list = test.precision_recall_list(
        alg_rec, test_dict, train_dict, range(5, topn, 5))
    alg_frame = pd.DataFrame(precision_list + recall_list).T
    alg_frame.index=['algorithm']
    rank_dict = algorithm.predict(mode='dict')
    auc = test.auc(train_dict, rank_dict, test_dict)
    auc_list.append(auc)

    #bpr
    bpr = Algorithm(train_frame, mweight=1, pweight=0, epochs=10000)
    bpr_rec = bpr.predict(mode='dict', top=topn)
    precision_list, recall_list = test.precision_recall_list(
        bpr_rec, test_dict, train_dict, range(5, topn, 5))
    bpr_frame = pd.DataFrame(precision_list + recall_list).T
    bpr_frame.index=['bpr']
    rank_dict = bpr.predict(mode='dict')
    auc = test.auc(train_dict, rank_dict, test_dict)
    auc_list.append(auc)

    #csvd
    item_train_frame = pd.read_csv('data/female_train.csv')
    csvd = CSVD(train_frame, item_train_frame)
    csvd.train(steps=10000)
    csvd_rec = csvd.predict(topn)
    precision_list, recall_list = test.precision_recall_list(
        csvd_rec, test_dict, train_dict, range(5, topn, 5))
    csvd_frame = pd.DataFrame(precision_list + recall_list).T
    csvd_frame.index=['csvd']
    rank_dict = csvd.predict()
    auc = test.auc(train_dict, rank_dict, test_dict)
    auc_list.append(auc)

    frame = pd.concat([ibcf_frame, alg_frame, bpr_frame, csvd_frame])
    frame['auc'] = auc_list
    t = time.ctime()
    fname = t[4:16].replace(' ', '-').replace(':', '-')
    frame.to_csv('log/%s.csv'%(fname))
    test.p_r_curve(frame.iloc[:, :-1], line=True, save=('log/' + fname + '.png'))

    return True

def loop(n):
    for i in range(n):
        step()
    return True

def log_reduce():
    log_frames = list()
    log_files = [f for f in os.listdir('log/') if f[-3:]=='csv']
    for log_file in log_files:
        log_frames.append(pd.read_csv('log/'+log_file, index_col=0))
    frame = pd.concat(log_frames)
    reduce_list = list()
    for alg in ['algorithm', 'bpr', 'ibcf', 'csvd']:
        reduce_series = frame.loc[alg].mean()
        reduce_series.name = alg
        reduce_list.append(reduce_series)
    reduce_frame = pd.DataFrame(reduce_list)
    return reduce_frame

def filter_log(frame):
    rstart = (len(frame.columns)-1) / 2
    filtered = frame.iloc[:, [0, 1, 9, rstart, rstart+1, rstart+9, -1]]
    filtered.columns = ['prec 5', 'prec 10', 'prec 50', 'recall 5', 'recall 10', 'recall 50', 'auc']
    return filtered

if __name__ == '__main__':
    loop(5)

    frame = log_reduce()
    frame.to_csv('log/reduce.csv')
    test.p_r_curve(frame, line=True, save='log/reduce.png')
    filter_frame = filter_log(frame)
    filter_frame.to_csv('log/final.csv')


    