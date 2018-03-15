# 內建库
import sys
import heapq
# 第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_format(train_frame, min_rate):
    def frame_to_dict(frame, user_index=0):
        match_dict = dict()
        for row in frame.iterrows():
            if user_index == 0:
                user, item, rate = row[1]
            else:
                item, user, rate = row[1]
            if user not in match_dict:
                match_dict[user] = dict()
            match_dict[user][item] = rate
        return match_dict
    train_frame = train_frame[train_frame.iloc[:, 2]>=min_rate]
    train_data = frame_to_dict(train_frame, user_index=0)
    return train_data


def auc(train_dict, rank_dict, test_dict):
    auc_values = []
    z = 0
    user_set = set(rank_dict.keys()) & set(test_dict.keys())
    for user in user_set:
        predictions = rank_dict[user]
        auc_for_user = 0.0
        n = 0
        pre_items = set(predictions.keys()) - set(train_dict[user].keys())
        pos_items = pre_items & set(test_dict[user].keys())
        neg_items = pre_items - pos_items
        for pos_item in pos_items:
            for neg_item in neg_items:
                n += 1
                if predictions[pos_item] > predictions[neg_item]:
                    auc_for_user += 1
                elif predictions[pos_item] == predictions[neg_item]:
                    auc_for_user += 0.5
        if n > 0:
            auc_for_user /= n
            auc_values.append(auc_for_user)
        z += 1
        if z % 100 == 0 and len(auc_values) > 0:
            sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z), np.mean(auc_values)))
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()
    return np.mean(auc_values)  

def precision_recall(recommend_dict, lable_dict, train_dict, top=1000, mode='base', sam=1):
    tp, p, r = 0, 0, 0
    precision_recall_list = list()
    user_array = np.array(list(set(lable_dict.keys()) & set(recommend_dict.keys())))
    if sam < 1:
        user_sample = user_array[np.random.randint(len(user_array), size=round(sam * len(user_array)))]
    else:
        user_sample = user_array
    for exp in user_sample:
        job_rank_dict = recommend_dict[exp]
# =============================================================================
#         job_rank = sorted(job_rank_dict.items(),key=lambda x: x[1], reverse=True)
#         rec = [j_r[0] for j_r in job_rank if j_r[0] not in train_dict[exp]][:top]
# =============================================================================
        rec = list()
        heap_top = 0
        while(len(rec)) < top:
            last_heap_top, heap_top = heap_top, heap_top + 2 * top
            job_rank = heapq.nlargest(heap_top, job_rank_dict.items(), key=lambda x: x[1])[last_heap_top : heap_top]
            rec += [j_r[0] for j_r in job_rank if j_r[0] not in train_dict[exp]]
            if heap_top > len(job_rank_dict):
                break            
        rec = rec[:top]
        rec_set = set(rec)
        positive_set = set(lable_dict[exp].keys()) - set(train_dict[exp].keys())
        if len(positive_set) > 0:
            if mode == 'max':
                precision = 1 if rec_set & positive_set else 0
                recall = 1 if rec_set & positive_set else 0
            else:
#                precision = len(rec_set & positive_set) / (len(rec_set)+0.1)
                precision = len(rec_set & positive_set) / top
                recall = len(rec_set & positive_set) / len(positive_set)
                tp+=len(rec_set & positive_set)
                p+=len(positive_set)
                r+=top
            precision_recall_list.append([precision, recall])
#    print(tp,p,r)
    if (mode == 'base') or (mode == 'max'):
        df = pd.DataFrame(precision_recall_list, columns=[
                          'precision', 'recall'])
        return pd.DataFrame([df.mean(), df.std()], index=['mean', 'std'])
    elif mode == 'sum':
        return ('precision, recall \n %f, %f' % ((tp / (tp + fp)), (tp / (tp + fn))))

def precision_recall_list(recommend_dict, lable_dict, train_dict, top_range, mode='base', sam=1):
    precision_list, recall_list = [], []
    for k in top_range:
        precision, recall = precision_recall(recommend_dict, lable_dict, train_dict, top=k, mode=mode, sam=sam).values[0]
        precision_list.append(precision)
        recall_list.append(recall)
    return precision_list, recall_list

def p_r_curve(frame, line=False, point=False, save=False):
    ls_list = map(lambda x: x, [ '-' , '--' , '-.' , ':' , 'steps'])
    l = int(len(frame.columns) / 2)
    for name, row in frame.iterrows():
        p = row[:l]
        r = row[l:]
        f = np.polyfit(p, r, 2)
        line_x = np.linspace(min(p), max(p), 10)
        line_y = np.polyval(f, line_x)
        if line:
            plt.plot(line_x, line_y, label=name, color='black', ls=next(ls_list))
        if point:
            plt.scatter(p, r, label=name)
    plt.legend()
    plt.xlabel('precision')
    plt.ylabel('recall')
    if save:
        plt.savefig(save)     
    plt.show()
    return True

def top_f1(frame, top_list, save=False):
    ls_list = map(lambda x: x, [ '-' , '--' , '-.' , ':' , 'steps'])    
    recall_start = int(len(frame.columns) / 2)
    for name, data in frame.iterrows():
        prec_array = data[:recall_start].values
        recall_array = data[recall_start:].values
        f1_list = 2 * prec_array * recall_array / (prec_array + recall_array)
        plt.plot(range(len(top_list)), f1_list, color='black', ls=next(ls_list), label=name)
        plt.scatter(range(len(top_list)), f1_list, color='black')
    plt.legend()
    plt.xticks(range(len(top_list)), top_list)
    plt.xlabel('top N')
    plt.ylabel('F1 score')
    if save:
        plt.savefig(save)
    plt.show()
    return True

def sri(train_dict, test_dict, rec_dict, topn):
    mean = lambda x: sum(x) / len(x)
    user_set = set(test_dict.keys()) & set(rec_dict.keys())
    sri_list, sr_list, bsr_list = [], [], []
    for user in user_set:
        job_rank = sorted(rec_dict[user].items(), key=lambda x: x[1], reverse=True)
        u_rec = [j_r[0] for j_r in job_rank if j_r[0] not in train_dict[user]][:topn]
        u_test = test_dict[user]
        t = set(u_test.keys())
        t_plus = set([i[0] for i in u_test.items() if i[1]==2])
        r = set(u_rec)
        sr = (len(t_plus & r) / (len(t & r) + 0.01))
        bsr = (len(t_plus) / len(t) + 0.01)
        sri = sr / bsr
        sri_list.append(sri)
        sr_list.append(sr)
        bsr_list.append(bsr)
    return mean(sri_list), mean(sr_list), mean(bsr_list)


if __name__ == '__main__':
    frame = pd.read_csv('../savelog/log_success_3000/final.csv', index_col=0)
    top_f1(frame.iloc[:, :-1], ['top 5', 'top 10', 'top 50'], save='../savelog/log_success_3000/f1.png')
        