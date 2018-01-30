# 內建库

# 第三方库
import pandas as pd
import numpy as np
# 本地库
import basic_bpr
import test

class Algorithm(object):
    def __init__(self, train_frame, mweight, pweight, epochs):
        male_train_raw = train_frame.values
        male_list = list(set(male_train_raw[male_train_raw[:, 2]==2, 0]))
        female_list = list(set(male_train_raw[:, 1]))
        male_to_index = dict(zip(male_list, range(len(male_list))))
        female_to_index = dict(zip(female_list, range(len(female_list))))
        self.male_to_index, self.female_to_index = male_to_index, female_to_index
        self.index_to_male = dict(zip(range(len(male_list)), male_list))
        self.index_to_female = dict(zip(range(len(female_list)), female_list))
        self.male_train = np.array([[male_to_index[i[0]], female_to_index[i[1]], i[2]]
            for i in male_train_raw if i[0] in male_to_index and i[1] in female_to_index])
        self.model = basic_bpr.BPR(rank=50, n_users=len(male_to_index),
                    n_items=len(female_to_index), match_weight=mweight, posi_weight=pweight)    
        self.model.train(self.male_train, epochs=epochs)

    def predict(self, mode='dict', top=False):
        if not top:
            top = len(self.female_to_index)
        if mode == 'dict':
            predict = self.model.prediction_to_dict(top, self.index_to_male, self.index_to_female)
        elif mode == 'matrix':
            predict = self.model.prediction_to_matrix()
        return predict

if __name__ == '__main__':
    # train_frame = pd.read_csv('../../public_data/male_train.csv')    
    # test_frame = pd.read_csv('../../public_data/male_test.csv')
    train_frame = pd.read_csv('../data/male_train.csv')    
    test_frame = pd.read_csv('../data/male_test.csv')
    test_dict = test.data_format(test_frame, min_rate=2)
    train_dict = test.data_format(train_frame, min_rate=2)

    topn = 50

    # algorithm
    algorithm = Algorithm(train_frame, mweight=100, pweight=100, epochs=3000)
    alg_rec = algorithm.predict(mode='dict')
    precision_list, recall_list = test.precision_recall_list(
        alg_rec, test_dict, train_dict, range(5, topn, 5))
    frame = pd.DataFrame(precision_list + recall_list).T
    frame.index=['algorithm']
    # auc = test.auc(train_dict, alg_rec, test_dict)

    # bpr
    bpr = Algorithm(train_frame, mweight=100, pweight=0, epochs=3000)
    bpr = bpr.predict(mode='dict', top=topn)
    precision_list, recall_list = test.precision_recall_list(
        bpr, test_dict, train_dict, range(5, topn, 5))
    bpr_frame = pd.DataFrame(precision_list + recall_list).T
    bpr_frame.index=['bpr']
    frame = pd.concat([frame, bpr_frame])

    test.p_r_curve(frame, line=True, point=True)
    test.top_f1(frame, range(5, topn, 5))
    







