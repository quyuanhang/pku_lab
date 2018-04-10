import time
import os
import shutil

import numpy as np
import pandas as pd

import gdata
import cdata
import test
import basic_bpr
import boosting_bpr
from ibcf import IBCF
from algorithm import Algorithm
from csvd import CSVD

def rec_test(train_dict, test_dict, rank_dict, alg_name):
    prec_array, recall_array = test.precision_recall_list(
        rank_dict, test_dict, train_dict, (5, 10, 50))
    f1_array = 2 * prec_array * recall_array / (prec_array + recall_array)
    frame = pd.DataFrame(np.concatenate([prec_array, recall_array, f1_array])).T
    frame.index = [alg_name]
    frame['auc'] = test.auc(train_dict, rank_dict, test_dict)
    frame['ndcg'] = test.ndcg(train_dict, rank_dict, test_dict, k=5000)
    frame['mAP'] = test.mAP(train_dict, rank_dict, test_dict, k=5000)
    return frame

def step():
#    sample_generator = gdata.sampleGenerator(n_user=1000, n_item=1000, sparseness=0.001)
#    sample_generator.sava_sample(t_size=0.5, save_path='../data/')

#    cdata.run()
    global ibcf_rec, alg_rec, bpr_rec

    train_frame = pd.read_csv('../data/male_train.csv')
    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)
    topn = 100
    frame = pd.DataFrame()

    # ibcf
    my_ibcf = IBCF(train_frame)
    ibcf_rec = my_ibcf.recommend_all(topn)
    ibcf_frame = rec_test(train_dict, test_dict, ibcf_rec, 'ibcf')
    frame = pd.concat([frame, ibcf_frame])

    # algorithm
    algorithm = Algorithm(train_frame, bweight=1, mweight=0.1, pweight=0.9, epochs=3000, model=boosting_bpr.BPR)
    alg_rec = algorithm.predict(mode='dict')
    alg_frame = rec_test(train_dict, test_dict, alg_rec, 'algorithm')
    frame = pd.concat([frame, alg_frame])
    

    #bpr
    bpr = Algorithm(train_frame, bweight=0, mweight=1, pweight=0, epochs=1000, model=basic_bpr.BPR)
    bpr_rec = bpr.predict(mode='dict')
    bpr_frame = rec_test(train_dict, test_dict, bpr_rec, 'bpr')
    frame = pd.concat([frame, bpr_frame])

    #csvd
    item_train_frame = pd.read_csv('../data/female_train.csv')
    csvd = CSVD(train_frame, item_train_frame)
    csvd.train(steps=3000)
    csvd_rec = csvd.predict()
    csvd_frame = rec_test(train_dict, test_dict, csvd_rec, 'csvd')
    frame = pd.concat([frame, csvd_frame])

    t = time.ctime()
    fname = t[4:16].replace(' ', '-').replace(':', '-')
    frame.to_csv('../log/%s.csv'%(fname))
    # test.p_r_curve(frame.iloc[:, :-1], line=True, save=('../log/' + fname + '.png'))
    
    return frame

def loop(n):
    for i in range(n):
        frame = step()
    return frame

def log_reduce():
    log_frames = list()
    log_files = [f for f in os.listdir('../log/') if f[-3:]=='csv']
    for log_file in log_files:
        log_frames.append(pd.read_csv('../log/'+log_file, index_col=0))
    frame = pd.concat(log_frames)
    reduce_list = list()
    if len(frame) == 4:
        return frame
    for alg in ['algorithm', 'bpr', 'ibcf', 'csvd']:
    # for alg in ['algorithm', 'ibcf', 'csvd']:        
        reduce_series = frame.loc[alg].mean()
        reduce_series.name = alg
        reduce_list.append(reduce_series)
    reduce_frame = pd.DataFrame(reduce_list)
    return reduce_frame

def filter_log(frame, iloc_indexes, col_name):
    filtered = frame.iloc[:, iloc_indexes]
    filtered_ = frame.loc[:, ['auc', 'ndcg', 'map']]
    filtered = pd.concat([filtered, filtered_])
    filtered.columns = col_name
    return filtered

if __name__ == '__main__':
    if os.path.exists('../log/'):
        shutil.rmtree('../log/')
    os.makedirs('../log/')

    frame = loop(1)

    frame = frame.reindex(index=['bpr', 'ibcf', 'csvd', 'algorithm'])
    over_best = frame.loc['algorithm'] / frame.iloc[:-1].max() - 1
    over_best.name = 'over best'
    frame = frame.append(over_best)
    test.p_r_curve(frame.iloc[:-1, :6], line=True, save='../log/reduce.png')

    # frame = filter_log(frame)
    frame.columns = [
            'prec5', 'prec10', 'prec50', 
            'recall5', 'recall10', 'recall50',
            'f1-5', 'f1-10', 'f1-50',
            'auc', 'ndcg', 'mAP']
    frame.to_csv('../log/final.csv')    

    # test.top_f1(filter_frame.iloc[:, :-1], top_list=['top 5', 'top 10', 'top 50'], save='../log/f1.png')


    # train_frame = pd.read_csv('../data/male_train.csv')
    # test_frame = pd.read_csv('../data/male_test.csv')
    # test_dict = test.data_format(test_frame, min_rate=2)
    # train_dict = test.data_format(train_frame, min_rate=2)
    # topn = 100
    #  list()

    # # algorithm
    # algorithm = Algorithm(train_frame, bweight=1, mweight=0.4, pweight=0.1, epochs=1000, model=boosting_bpr.BPR)
    # alg_rec = algorithm.predict(mode='dict')
    # # pr1=test.precision_recall(alg_rec, test_dict, train_dict, top=50)
    # ndcg = test.ndcg(train_dict, alg_rec, test_dict)
    # mAP = test.mAP(train_dict, alg_rec, test_dict)

    # #bpr
    # bpr = Algorithm(train_frame, bweight=0, mweight=1, pweight=0, epochs=1000, model=basic_bpr.BPR)
    # bpr_rec = bpr.predict(mode='dict')
    # pr2=test.precision_recall(bpr_rec, test_dict, train_dict, top=50)
