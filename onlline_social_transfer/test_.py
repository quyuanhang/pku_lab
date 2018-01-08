# 內建库
import sys
# 第三方库
import pandas as pd
import numpy as np

def user_auc(prediction_mat, train_data, test_data, s=0.3):
    def _data_to_dict(data):
        pos_dict = dict()
        match_dict = dict()
        all_items = set()
        for (user, item, rate) in data:
            if rate == 2:
                if user not in match_dict:
                    match_dict[user] = list()
                match_dict[user].append(item)
            else:
                if user not in pos_dict:
                    pos_dict[user] = list()
                pos_dict[user].append(item)
            all_items.add(item)
        return match_dict, pos_dict, set(match_dict.keys()), all_items
    train_match_dict, train_pos_dict, train_users, train_items = _data_to_dict(train_data)    
    test_match_dict, test_pos_dict, test_users, test_items = _data_to_dict(test_data)
    auc_values = []
    z = 0
    user_array = np.array(list(test_users & train_users))
    user_sample = user_array[np.random.randint(len(user_array), size=round(s * len(user_array)))]
# =============================================================================
#     user_sample = user_array
# =============================================================================
    for user in user_sample:
        auc_for_user = 0.0
        n = 0
        predictions = prediction_mat[user]
        match_items = set(test_match_dict[user]) & train_items - set(train_match_dict[user])
        pos_items = set(test_pos_dict[user]) & train_items - set(train_pos_dict[user])
        neg_items = train_items - match_items - pos_items - set(train_pos_dict[user]) - set(train_match_dict[user])
        for match_item in match_items:
            for other_item in pos_items | neg_items:
                n += 1
                if predictions[match_item] > predictions[other_item]:
                    auc_for_user += 1
                elif predictions[match_item] == predictions[other_item]:
                    auc_for_user += 0.5
        if n > 0:
            auc_for_user /= n
            auc_values.append(auc_for_user)
        z += 1
        if z % 10 == 0 and len(auc_values) > 0:
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.3f" % (str(z), np.mean(auc_values)))
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()
    return np.mean(auc_values)

def sample_auc(p_array, test_y, split):
    positive_index = [i[0] for i in enumerate(test_y) if i[1] >= split]
    negative_index = [i[0] for i in enumerate(test_y) if i[1] < split]
    positive_score = p_array[positive_index]
    negative_score = p_array[negative_index]
    auc = 0.0
    for pos_s in positive_score:
        for neg_s in negative_score:
            if pos_s > neg_s:
                auc += 1
            if pos_s == neg_s:
                auc += 0.5
    auc /= (len(positive_score) * len(negative_score))
    return auc

def precision_recall(recommend, lable_dict, train_dict, top=1000, mode='base', sam=0.3):
    tp, fp, fn = 0, 0, 0
    precision_recall_list = list()
    user_array = np.array(list(set(lable_dict.keys()) & set(train_dict.keys())))
    user_sample = user_array[np.random.randint(len(user_array), size=round(sam * len(user_array)))]
    for exp in user_sample:
        job_rank_list = recommend[exp]
        job_rank = sorted(enumerate(job_rank_list), key=lambda x: x[1], reverse=True)
        rec = [j_r[0] for j_r in job_rank if j_r[0] not in train_dict[exp]][:top]
        rec_set = set(rec)
        positive_set = set(lable_dict[exp].keys()) - set(train_dict[exp].keys())
        tp += len(rec_set & positive_set)
        fp += len(rec_set - positive_set)
        fn += len(positive_set - rec_set)
        if len(positive_set) > 0:
            if mode == 'max':
                precision = 1 if rec_set & positive_set else 0
                recall = 1 if rec_set & positive_set else 0
            else:
                precision = len(rec_set & positive_set) / (len(rec_set) + 0.01)
                recall = len(rec_set & positive_set) / (len(positive_set) + 0.01)
            precision_recall_list.append([precision, recall])
    if (mode == 'base') or (mode == 'max'):
        df = pd.DataFrame(precision_recall_list, columns=[
                          'precision', 'recall'])
        return pd.DataFrame([df.mean(), df.std()], index=['mean', 'std'])
    elif mode == 'sum':
        return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp + fn))))

def data_to_dict(training_data, min_rate):
    train_dict = dict()
    for row in training_data:
        user, item, rate = row
        if rate >= min_rate:
            if user not in train_dict:
                train_dict[user] = dict()
            train_dict[user][item] = rate
    return train_dict