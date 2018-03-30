# 內建库
import time
# 第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 本地库
import basic_bpr
import boosting_bpr
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
                    n_items=len(female_to_index), base_weight=bweight, match_weight=mweight, posi_weight=pweight, sgd_weight=1)    
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

    # algorithm
    female = pd.read_csv('../data/female_train.csv')
    boosting_train_frame = pd.concat([train_frame, female]).drop_duplicates()
    algorithm = Algorithm(boosting_train_frame, bweight=1, mweight=0, pweight=0, epochs=1000, model=boosting_bpr.BPR)
#    algorithm = Algorithm(train_frame, bweight=1, mweight=0, pweight=0, epochs=1000, model=boosting_bpr.BPR)
    alg_rec = algorithm.predict(mode='dict')
    precision_list, recall_list = test.precision_recall_list(
        alg_rec, test_dict, train_dict, range(5, topn, 5))
    frame = pd.DataFrame(precision_list + recall_list).T
    frame.index=['algorithm']
 
    # bpr
    bpr = Algorithm(train_frame, bweight=1, mweight=1, pweight=0, epochs=1000, model=basic_bpr.BPR)
    bpr_rec = bpr.predict(mode='dict')
    precision_list, recall_list = test.precision_recall_list(
        bpr_rec, test_dict, train_dict, range(5, topn, 5))
    bpr_frame = pd.DataFrame(precision_list + recall_list).T
    bpr_frame.index=['bpr']
    frame = pd.concat([frame, bpr_frame])

    test.p_r_curve(frame, line=False, point=True)
    # test.top_f1(frame, range(5, topn, 5))
    
    def rec_test(train_dict, test_dict, rank_dict, topn, alg_name):
        precision_list, recall_list = test.precision_recall_list(
            rank_dict, test_dict, train_dict, range(5, topn, 5))
        frame = pd.DataFrame(precision_list + recall_list).T
        auc = test.auc(train_dict, rank_dict, test_dict)
        frame['auc'] = auc
        frame.index = [alg_name]
        return frame

    def refresh_data(test_size=0.5, test_rate=2):
# =============================================================================
#         sample_generator = gdata.sampleGenerator(n_user=1000, n_item=1000, sparseness=0.001)
#         sample_generator.sava_sample(t_size=0.5, save_path='../data/')
# =============================================================================

#        cdata.run()

        train_frame = pd.read_csv('../data/male_train.csv')    
        test_frame = pd.read_csv('../data/male_test.csv')
        test_dict = test.data_format(test_frame, min_rate=2)
        train_dict = test.data_format(train_frame, min_rate=2)

        return train_frame, train_dict, test_dict
  
#    def reduce_test(loop):
#        reducel = list()
#        for step in range(loop):
#            l = list()        
#            train_frame, train_dict, test_dict = refresh_data()
#            b_weight_list = [i/5 for i in range(6)]
#            m_weight_list = [i/5 for i in range(6)]
#            p_weight_list = [i/5 for i in range(6)]
#            # frame = pd.DataFrame(index=m_weight_list, columns=p_weight_list)
#            for k in b_weight_list:
#                for i in m_weight_list:
#                    for j in p_weight_list:
#                        algorithm = Algorithm(train_frame, bweight=k, mweight=i, pweight=j, epochs=1000, model=boosting_bpr.BPR)
#                        rank_dict = algorithm.predict(mode='dict')
#                        auc = test.auc(train_dict, rank_dict, test_dict)
#                        l.append(auc)
#            reducel.append(l)
#        r = np.array(reducel).mean(axis=0)
#        return r
        
    # def reduce_test(loop):
    #     reduce_dict = dict()
    #     for step in range(loop):
    #         train_frame, train_dict, test_dict = refresh_data()
    #         m_weight_list = [i/10 for i in range(0, 11)]
    #         p_weight_list = [i/10 for i in range(0, 11)]
    #         frame = pd.DataFrame(index=m_weight_list, columns=p_weight_list)
    #         for i in m_weight_list:
    #             for j in p_weight_list:
    #                 algorithm = Algorithm(train_frame, bweight=1, mweight=i, pweight=j, epochs=1000, model=boosting_bpr.BPR)
    #                 rank_dict = algorithm.predict(mode='dict')
    #                 auc = test.auc(train_dict, rank_dict, test_dict)
    #                 frame[j][i] = auc
    #         reduce_dict[step] = frame
    #     df = pd.Panel(reduce_dict).mean(axis=0)
    #     return df
    
    # frame = pd.DataFrame(reduce_test(3))
    # t = time.ctime()
    # fname = t[4:16].replace(' ', '-').replace(':', '-')
    # frame.to_csv('../savelog/mat%s.csv'%(fname))

