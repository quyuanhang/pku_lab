import time
import os
import shutil

import pandas as pd

import gdata
import cdata
import test
import basic_bpr
import boosting_bpr
from ibcf import IBCF
from algorithm import Algorithm
from csvd import CSVD

def rec_test(train_dict, test_dict, rank_dict, topn, auc_list, alg_name):
    precision_list, recall_list = test.precision_recall_list(
        rank_dict, test_dict, train_dict, range(5, topn, 5))
    frame = pd.DataFrame(precision_list + recall_list).T
    frame.index = [alg_name]
    auc = test.auc(train_dict, rank_dict, test_dict)
    auc_list.append(auc)
    return frame

def step():
#    sample_generator = gdata.sampleGenerator(n_user=1000, n_item=1000, sparseness=0.001)
#    sample_generator.sava_sample(t_size=0.5, save_path='../data/')

#    cdata.run()

    train_frame = pd.read_csv('../data/male_train.csv')
    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)
    topn = 100
    auc_list = list()

    # ibcf
    my_ibcf = IBCF(train_frame)
    ibcf_rec = my_ibcf.recommend_all(topn)
    ibcf_frame = rec_test(train_dict, test_dict, ibcf_rec, topn, auc_list, 'ibcf')

    # algorithm
    algorithm = Algorithm(train_frame, bweight=1, mweight=0.1, pweight=0.9, epochs=1000, model=boosting_bpr.BPR)
    alg_rec = algorithm.predict(mode='dict')
    alg_frame = rec_test(train_dict, test_dict, alg_rec, topn, auc_list, 'algorithm')

    #bpr
    # bpr = Algorithm(train_frame, bweight=0, mweight=1, pweight=0, epochs=1000, model=basic_bpr.BPR)
    # bpr_rec = bpr.predict(mode='dict')
    # bpr_frame = rec_test(train_dict, test_dict, bpr_rec, topn, auc_list, 'bpr')

    #csvd
    item_train_frame = pd.read_csv('../data/female_train.csv')
    csvd = CSVD(train_frame, item_train_frame)
    csvd.train(steps=1000)
    csvd_rec = csvd.predict()
    csvd_frame = rec_test(train_dict, test_dict, csvd_rec, topn, auc_list, 'csvd')

    # frame = pd.concat([ibcf_frame, alg_frame, bpr_frame, csvd_frame])
    frame = pd.concat([ibcf_frame, alg_frame, csvd_frame])
    frame['auc'] = auc_list
    t = time.ctime()
    fname = t[4:16].replace(' ', '-').replace(':', '-')
    frame.to_csv('../log/%s.csv'%(fname))
    test.p_r_curve(frame.iloc[:, :-1], line=True, save=('../log/' + fname + '.png'))

    return frame

def loop(n):
    for i in range(n):
        step()
    return True

def log_reduce():
    log_frames = list()
    log_files = [f for f in os.listdir('../log/') if f[-3:]=='csv']
    for log_file in log_files:
        log_frames.append(pd.read_csv('../log/'+log_file, index_col=0))
    frame = pd.concat(log_frames)
    reduce_list = list()
    if len(frame) == 4:
        return frame
    # for alg in ['algorithm', 'bpr', 'ibcf', 'csvd']:
    for alg in ['algorithm', 'ibcf', 'csvd']:        
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
     if os.path.exists('../log/'):
         shutil.rmtree('../log/')
     os.makedirs('../log/')

     loop(1)

     frame = log_reduce()
    #  frame = frame.reindex(index=['algorithm', 'bpr', 'ibcf', 'csvd'])
     frame = frame.reindex(index=['algorithm', 'ibcf', 'csvd'])     
     frame.to_csv('../log/reduce.csv')
     test.p_r_curve(frame.iloc[:, :-1], line=True, save='../log/reduce.png')

     filter_frame = filter_log(frame)
     filter_frame.to_csv('../log/final.csv')    

     test.top_f1(filter_frame.iloc[:, :-1], top_list=['top 5', 'top 10', 'top 50'], save='../log/f1.png')


#    train_frame = pd.read_csv('../data/male_train.csv')
#    test_frame = pd.read_csv('../data/male_test.csv')
#    test_dict = test.data_format(test_frame, min_rate=2)
#    train_dict = test.data_format(train_frame, min_rate=2)
#    topn = 100
#    auc_list = list()
#
#    # algorithm
#    algorithm = Algorithm(train_frame, bweight=1, mweight=0.4, pweight=0.1, epochs=1000, model=boosting_bpr.BPR)
#    alg_rec = algorithm.predict(mode='dict')
#    pr1=test.precision_recall(alg_rec, test_dict, train_dict, top=50)
#
#    #bpr
#    bpr = Algorithm(train_frame, bweight=0, mweight=1, pweight=0, epochs=1000, model=basic_bpr.BPR)
#    bpr_rec = bpr.predict(mode='dict')
#    pr2=test.precision_recall(bpr_rec, test_dict, train_dict, top=50)
