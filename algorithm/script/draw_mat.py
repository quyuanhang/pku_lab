import pandas as pd
import numpy as np

data = pd.read_csv('../savelog/matMar--5-10-41.csv', index_col=0)
data_list = list(data.iloc[:, 0])

arr = np.empty((6,6,6))

b_weight_list = [i for i in range(6)]
m_weight_list = [i for i in range(6)]
p_weight_list = [i for i in range(6)]
# frame = pd.DataFrame(index=m_weight_list, columns=p_weight_list)
for bw in b_weight_list:
    for mw in m_weight_list:
        for pw in p_weight_list:
            arr[bw, mw, pw] = data_list.pop(0)
            
index = np.where(arr == np.max(arr))

bm = pd.DataFrame(arr.mean(axis=2))
mp = pd.DataFrame(arr.mean(axis=0))
bp = pd.DataFrame(arr.mean(axis=1))


bm.to_csv('bm.csv')
mp.to_csv('mp.csv')
bp.to_csv('bp.csv')



