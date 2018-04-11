# 內建库
import time
# 第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 本地库
import basic_bpr
import boosting_bpr
import update_bpr
import test
import gdata
import cdata

class Algorithm(object):
    def __init__(self, train_frame, bweight, mweight, pweight, epochs, model):
        male_train_raw = train_frame.values
        if mweight == None:
            mweight = len(male_train_raw[male_train_raw[:, 2]==2]) / len(male_train_raw)
            pweight = 1 - mweight
        male_list = list(
                set(male_train_raw[male_train_raw[:, 2]==2, 0]) & 
                set(male_train_raw[male_train_raw[:, 2]==1, 0]))
        female_list = list(set(male_train_raw[:, 1]))
        male_to_index = dict(zip(male_list, range(len(male_list))))
        female_to_index = dict(zip(female_list, range(len(female_list))))
        self.male_to_index, self.female_to_index = male_to_index, female_to_index
        self.index_to_male = dict(zip(range(len(male_list)), male_list))
        self.index_to_female = dict(zip(range(len(female_list)), female_list))
        self.male_train = np.array([[male_to_index[i[0]], female_to_index[i[1]], i[2]]
            for i in male_train_raw if i[0] in male_to_index and i[1] in female_to_index])
        print('\nproblem space:', len(male_list), len(female_list))
        self.model = model(rank=50, n_users=len(male_to_index),
                    n_items=len(female_to_index), base_weight=bweight, match_weight=mweight, posi_weight=pweight, sgd_weight=0.8)    
        self.model.train(self.male_train, epochs=epochs)

    def predict(self, mode='dict', top=False):
        if mode == 'dict':
            predict = self.model.prediction_to_dict(top, self.index_to_male, self.index_to_female)
        elif mode == 'matrix':
            predict = self.model.prediction_to_matrix()
        return predict

if __name__ == '__main__':
#==============================================================================
#     sample_generator = gdata.sampleGenerator(n_user=1000, n_item=1000, sparseness=0.001)
#     sample_generator.sava_sample(t_size=0.5, save_path='../data/')
#==============================================================================
    
    # cdata.run()
    
    train_frame = pd.read_csv('../data/male_train.csv')
    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)

    topn = 50

    frame = pd.DataFrame()

 
    # old alg
    algorithm = Algorithm(train_frame, bweight=1, mweight=0, pweight=0, epochs=1000, model=boosting_bpr.BPR)
    alg_rec = algorithm.predict(mode='dict')
    precision_list, recall_list = test.precision_recall_list(
        alg_rec, test_dict, train_dict, range(5, topn, 5))
    old_frame = pd.DataFrame(list(precision_list) + list(recall_list)).T
    old_frame.index=['old']
    frame = pd.concat([frame, old_frame])


    # bpr
    bpr = Algorithm(train_frame, bweight=1, mweight=1, pweight=0, epochs=1000, model=basic_bpr.BPR)
    bpr_rec = bpr.predict(mode='dict')
    precision_list, recall_list = test.precision_recall_list(
        bpr_rec, test_dict, train_dict, range(5, topn, 5))
    bpr_frame = pd.DataFrame(list(precision_list) + list(recall_list)).T
    bpr_frame.index=['bpr']
    frame = pd.concat([frame, bpr_frame])

    test.p_r_curve(frame, line=False, point=True)
    
